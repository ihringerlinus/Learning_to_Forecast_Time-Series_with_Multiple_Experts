"""
Volatilitätsprognose: Streamlit-App

Lädt auf Knopfdruck die neuesten 5-Minuten-Kerzen von Binance, berechnet die tägliche
realisierte Varianz (RV) und prognostiziert sie für heute und in 5 Tagen, mit denselben
Experten und derselben statischen Kombination wie in den Notebooks.

Starten (aus dem Ordner app/, in der .venv):

    streamlit run app.py

Die Forschungsdaten in src/data/ und src/results/ werden nur gelesen, nie verändert.
Neue Daten und das Prognose-Logbuch liegen in src/data/live/.

t0 und Timestral (seit Herbst 2026) laufen nur live und separat mit: als eigene Modelle und in
einer zweiten Kombination. Die Hauptprognose und die Kursspanne verwenden sie nicht.
Installation und Prüfung: src/03_t0_timestral.ipynb.
"""

import io
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.special import ndtr

warnings.filterwarnings("ignore", module="arch")

# =====================================================================================
# Konfiguration (wie in den Notebooks)
# =====================================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "data"
RESULTS_DIR = ROOT / "src" / "results"
LIVE_DIR = DATA_DIR / "live"
LOG_PATH = LIVE_DIR / "forecast_log.parquet"

SYMBOLS = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana"}
HORIZONS = [1, 5]
QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
MIN_BARS_PER_DAY = 270          # Notebook 1
EST_WINDOW = 1000               # Notebook 1 (RW, HAR, GARCH)
CONTEXT = 1024                  # Kontext der Foundation Models
CHRONOS_MODEL = "amazon/chronos-2"
TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
T0_REPO = "theforecastingcompany/t0-gamma"
TIMESTRAL_REPO = "mistralai/Timestral-209M"

POOL = ["Chronos-2-full", "TimesFM-2.5", "HAR", "GARCH"]    # wie 01_kombination.ipynb
BASELINE = "RW"
COMBO = "Kombination"

# Neue Foundation Models: nur live, separat (möglicherweise kontaminiert, siehe 03_t0_timestral.ipynb)
NEW_MODELS = ["t0-gamma", "Timestral-209M"]
FOUNDATION = ["Chronos-2-full", "TimesFM-2.5"] + NEW_MODELS
POOL_NEW = POOL + NEW_MODELS
COMBO_NEW = "Kombination neu"
TRAIN_END = "2025-10-31"        # statische Gewichte nur aus train
TEST_END = "2026-08-31"         # Logbuch wertet nur Ziele danach aus (Testperiode bleibt unberührt)
ALPHA = 1.0

LABELS = {COMBO: "Kombination", "Chronos-2-full": "Chronos-2", "TimesFM-2.5": "TimesFM-2.5",
          "HAR": "HAR", "GARCH": "GARCH", BASELINE: "Random Walk",
          COMBO_NEW: "Kombination mit t0 und Timestral (neu)", "t0-gamma": "t0 (neu)",
          "Timestral-209M": "Timestral (neu)"}
MAIN_ORDER = [COMBO] + POOL + [BASELINE]
NEW_ORDER = [COMBO_NEW] + NEW_MODELS

API_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
DAILY_ZIP = "https://data.binance.vision/data/spot/daily/klines/{sym}/5m/{sym}-5m-{day}.zip"

# Farben (dataviz-Referenzpalette)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, BLUE_SOFT = "#2a78d6", "#cde2fb"
ORANGE = "#c2601d"                 # zweite Kombination (neu)

_LV = np.asarray(QUANTILES)
_QMASS = (np.r_[_LV[1:], 1.0] - np.r_[0.0, _LV[:-1]]) / 2
_QMASS = _QMASS / _QMASS.sum()


# =====================================================================================
# Daten: identisch zu Notebook 1
# =====================================================================================
def daily_rv(close):
    r = np.log(close).diff().dropna()
    day = r.index.floor("D").tz_localize(None)
    daily = pd.DataFrame({"rv": (r**2).groupby(day).sum(),
                          "ret": r.groupby(day).sum(),
                          "n_bars": r.groupby(day).size()})
    daily.index.name = "date"
    ok = (daily["n_bars"] >= MIN_BARS_PER_DAY) & (daily["rv"] > 0)
    return daily[ok]


def read_kline_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as zf, zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(f, header=None)
    if not str(df.iloc[0, 0]).strip().isdigit():
        df = df.iloc[1:]
    t = pd.to_numeric(df.iloc[:, 0]).astype("int64").to_numpy()
    unit = "us" if t.max() > 1e14 else "ms"
    return pd.Series(pd.to_numeric(df.iloc[:, 4]).to_numpy(float),
                     index=pd.to_datetime(t, unit=unit, utc=True), name="close")


def fetch_close_api(sym, start, end):
    # 5-Minuten-Schlusskurse [start, end) über die öffentliche Binance-API (ohne Schlüssel)
    last_err = None
    for host in API_HOSTS:
        try:
            rows, cursor = [], int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            while cursor < end_ms:
                r = requests.get(f"{host}/api/v3/klines", timeout=30, params={
                    "symbol": sym, "interval": "5m", "startTime": cursor,
                    "endTime": end_ms - 1, "limit": 1000})
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                rows += batch
                cursor = batch[-1][0] + 1
            t = np.array([b[0] for b in rows], dtype="int64")
            return pd.Series(np.array([float(b[4]) for b in rows]),
                             index=pd.to_datetime(t, unit="ms", utc=True), name="close")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Binance-API nicht erreichbar ({last_err})")


def fetch_close_zips(sym, first_day, last_day):
    # Fallback: Tagesdateien von data.binance.vision (wie Notebook 1, nur täglich statt monatlich)
    parts = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        r = requests.get(DAILY_ZIP.format(sym=sym, day=day.strftime("%Y-%m-%d")), timeout=60)
        if r.status_code in (403, 404):
            continue
        r.raise_for_status()
        parts.append(read_kline_zip(r.content))
    if not parts:
        raise RuntimeError("Keine Tagesdateien gefunden")
    close = pd.concat(parts).sort_index()
    return close[~close.index.duplicated()]


def load_daily(sym):
    live = LIVE_DIR / f"{sym}_daily.parquet"
    return pd.read_parquet(live if live.exists() else DATA_DIR / f"{sym}_daily.parquet")


def append_new_days(d, close, today):
    # neue vollständige Tage aus den Kursen berechnen und an die bestehenden anhängen
    last = d.index[-1]
    new = daily_rv(close)
    new = new[(new.index > last) & (new.index < today)]
    return pd.concat([d, new]), len(new)


def update_daily(sym, fetch_api=fetch_close_api, fetch_zips=fetch_close_zips, now=None):
    d = load_daily(sym)
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize().tz_localize(None)                  # heutiger UTC-Tag ist unvollständig
    last = d.index[-1]
    if last >= today - pd.Timedelta(days=1):
        return d, 0
    # ab der letzten Kerze des letzten bekannten Tages, damit die erste Rendite wie im Notebook entsteht
    start = (last + pd.Timedelta(hours=23, minutes=55)).tz_localize("UTC")
    end = today.tz_localize("UTC")
    try:
        close = fetch_api(sym, start, end)
    except Exception:
        close = fetch_zips(sym, last, today - pd.Timedelta(days=1))
    out, n_new = append_new_days(d, close, today)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LIVE_DIR / f"{sym}_daily.parquet")
    return out, n_new


# =====================================================================================
# Tageskurse (Schlusskurs eines UTC-Tages) für die Kursspanne
# =====================================================================================
ZIP_1D_DAY = "https://data.binance.vision/data/spot/daily/klines/{sym}/1d/{sym}-1d-{day}.zip"
ZIP_1D_MONTH = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip"
PRICE_DAYS = 120


def fetch_price_api(sym, first_day, last_day):
    last_err = None
    for host in API_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", timeout=30, params={
                "symbol": sym, "interval": "1d", "limit": 1000,
                "startTime": int(first_day.tz_localize("UTC").timestamp() * 1000),
                "endTime": int((last_day + pd.Timedelta(days=1)).tz_localize("UTC").timestamp() * 1000) - 1})
            r.raise_for_status()
            rows = r.json()
            idx = pd.to_datetime([b[0] for b in rows], unit="ms").normalize()
            return pd.Series([float(b[4]) for b in rows], index=idx, name="close")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Binance-API nicht erreichbar ({last_err})")


def fetch_price_zips(sym, first_day, last_day):
    parts = []
    for m in pd.period_range(first_day, last_day, freq="M"):
        if m < last_day.to_period("M"):
            urls = [ZIP_1D_MONTH.format(sym=sym, ym=m.strftime("%Y-%m"))]
        else:
            urls = [ZIP_1D_DAY.format(sym=sym, day=d.strftime("%Y-%m-%d"))
                    for d in pd.date_range(m.start_time, last_day, freq="D")]
        for url in urls:
            r = requests.get(url, timeout=60)
            if r.status_code in (403, 404):
                continue
            r.raise_for_status()
            parts.append(read_kline_zip(r.content))
    if not parts:
        raise RuntimeError("Keine Kursdateien gefunden")
    close = pd.concat(parts)
    close.index = close.index.floor("D").tz_localize(None)
    return close[~close.index.duplicated(keep="last")].sort_index()


def load_price(sym):
    path = LIVE_DIR / f"{sym}_price.parquet"
    return pd.read_parquet(path)["close"] if path.exists() else None


def update_price(sym, fetch_api=fetch_price_api, fetch_zips=fetch_price_zips, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    last = now.normalize().tz_localize(None) - pd.Timedelta(days=1)       # letzter vollständiger Tag
    old = load_price(sym)
    if old is not None and old.index[-1] >= last:
        return old
    first = last - pd.Timedelta(days=PRICE_DAYS)
    try:
        new = fetch_api(sym, first, last)
    except Exception:
        new = fetch_zips(sym, first, last)
    new = new[(new.index >= first) & (new.index <= last)]
    price = new if old is None else pd.concat([old, new])
    price = price[~price.index.duplicated(keep="last")].sort_index()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    price.rename("close").to_frame().to_parquet(LIVE_DIR / f"{sym}_price.parquet")
    return price


# =====================================================================================
# Experten: identisch zu Notebook 1 und 02_timesfm.ipynb, für den letzten Tag als Ursprung
# =====================================================================================
def dist_from_log_resid(log_point, resid):
    resid = resid[np.isfinite(resid)]
    q = np.exp(log_point + np.quantile(resid, _LV))
    mean = np.exp(log_point) * np.mean(np.exp(resid))
    return mean, q


def mean_from_quantiles(q):
    return q @ _QMASS


def rw_forecast(rv):
    L, t = np.log(rv), len(rv) - 1
    lo = max(0, t - EST_WINDOW + 1)
    out = {}
    for h in HORIZONS:
        resid = L[lo + h: t + 1] - L[lo: t + 1 - h]
        out[h] = dist_from_log_resid(L[t], resid)
    return out


def har_forecast(rv):
    s = pd.Series(rv)
    X = np.column_stack([np.ones(len(s)), np.log(s),
                         np.log(s.rolling(5).mean()), np.log(s.rolling(22).mean())])
    L, t = np.log(rv), len(rv) - 1
    lo = max(21, t - EST_WINDOW + 1)
    out = {}
    for h in HORIZONS:
        rows = np.arange(lo, t - h + 1)
        beta, *_ = np.linalg.lstsq(X[rows], L[rows + h], rcond=None)
        resid = L[rows + h] - X[rows] @ beta
        out[h] = dist_from_log_resid(X[t] @ beta, resid)
    return out


def garch_forecast(rv, ret):
    from arch import arch_model
    t = len(rv) - 1
    lo = max(0, t - EST_WINDOW + 1)
    am = arch_model(100 * ret[lo: t + 1], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    sig2_in = np.asarray(res.conditional_volatility) ** 2 / 1e4
    sig2_fc = res.forecast(horizon=max(HORIZONS), reindex=False).variance.to_numpy()[-1] / 1e4
    resid = np.log(rv[lo: t + 1]) - np.log(sig2_in)
    return {h: dist_from_log_resid(np.log(sig2_fc[h - 1]), resid) for h in HORIZONS}


_ND = NormalDist()
_Z_NATIVE = np.array([_ND.inv_cdf(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
_Z_TARGET = np.array([_ND.inv_cdf(p) for p in QUANTILES])


def expand_quantiles(logq9):
    # wie 02_timesfm.ipynb: 9 TimesFM-Quantile -> 21, in Logs auf der Normal-Skala
    q = np.asarray(logq9, float)
    out = np.empty(q.shape[:-1] + (len(QUANTILES),))
    lo_slope = np.maximum((q[..., 2] - q[..., 0]) / (_Z_NATIVE[2] - _Z_NATIVE[0]), 1e-6)
    hi_slope = np.maximum((q[..., 8] - q[..., 6]) / (_Z_NATIVE[8] - _Z_NATIVE[6]), 1e-6)
    for k, z in enumerate(_Z_TARGET):
        if z < _Z_NATIVE[0] - 1e-12:
            out[..., k] = q[..., 0] + lo_slope * (z - _Z_NATIVE[0])
        elif z > _Z_NATIVE[-1] + 1e-12:
            out[..., k] = q[..., 8] + hi_slope * (z - _Z_NATIVE[-1])
        else:
            j = int(np.clip(np.searchsorted(_Z_NATIVE, z), 1, 8))
            w = (z - _Z_NATIVE[j - 1]) / (_Z_NATIVE[j] - _Z_NATIVE[j - 1])
            out[..., k] = (1 - w) * q[..., j - 1] + w * q[..., j]
    return out


@st.cache_resource(show_spinner=False)
def load_chronos():
    import torch
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map="cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_timesfm():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT, max_horizon=max(HORIZONS), normalize_inputs=True,
        per_core_batch_size=4, use_continuous_quantile_head=True,
        fix_quantile_crossing=True, infer_is_positive=False))
    return model


def chronos_forecast(rv, pipe):
    ctx = np.log(rv).astype(np.float32)[-CONTEXT:]
    qs, _ = pipe.predict_quantiles([ctx], prediction_length=max(HORIZONS), quantile_levels=QUANTILES,
                                   batch_size=1, context_length=CONTEXT, cross_learning=False)
    log_q = qs[0][0].detach().float().cpu().numpy()
    out = {}
    for h in HORIZONS:
        q = np.sort(np.exp(log_q[h - 1]))
        out[h] = (mean_from_quantiles(q), q)
    return out


def timesfm_forecast(rv, model):
    ctx = np.log(rv).astype(np.float32)[-CONTEXT:]
    _, q = model.forecast(horizon=max(HORIZONS), inputs=[ctx])
    q9 = np.asarray(q)[0, :max(HORIZONS), 1:]
    out = {}
    for h in HORIZONS:
        qq = np.sort(np.exp(expand_quantiles(q9[h - 1])))
        out[h] = (mean_from_quantiles(qq), qq)
    return out


# Neue Foundation Models: Setup wie in 03_t0_timestral.ipynb (log-RV, Kontext 1024, 21 Quantile)
@st.cache_resource(show_spinner=False)
def load_t0():
    from t0 import T0Forecaster
    return T0Forecaster.from_pretrained(T0_REPO).eval()


@st.cache_resource(show_spinner=False)
def load_timestral():
    import torch
    from timestral import TimestralForecaster
    model = TimestralForecaster.from_pretrained(TIMESTRAL_REPO, device="cuda" if torch.cuda.is_available() else "cpu")
    levels = np.asarray(model.quantile_levels, float)
    if len(levels) != len(_LV) or not np.allclose(levels, _LV):
        raise ValueError(f"Timestral liefert andere Quantilstufen als erwartet: {tuple(levels)}")
    return model


def _as_numpy(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, float)


def _finish_log_quantiles(log_q):
    # (H, 21) Log-Quantile -> {h: (mean, q)}, Rücktransformation wie bei Chronos und TimesFM
    out = {}
    for h in HORIZONS:
        q = np.sort(np.exp(np.asarray(log_q[h - 1], float)))
        out[h] = (mean_from_quantiles(q), q)
    return out


def t0_forecast(rv, model):
    ctx = np.log(rv).astype(np.float32)[-CONTEXT:][None, :]
    res = model.predict(ctx, horizon=max(HORIZONS), quantile_levels=QUANTILES,
                        trim_leading_missing=True, symmetric_averaging=True,
                        clip_nonnegative=False, snap_integers=False)
    return _finish_log_quantiles(_as_numpy(res.quantiles).reshape(max(HORIZONS), len(QUANTILES)))


def timestral_forecast(rv, model):
    ctx = np.log(rv).astype(np.float32)[-CONTEXT:][None, None, :]
    res = model.forecast(ctx, prediction_length=max(HORIZONS), symmetric_averaging=True, clip_nonnegative=False)
    return _finish_log_quantiles(_as_numpy(res.quantiles)[0, 0])


MODEL_FNS = {"Chronos-2-full": chronos_forecast, "TimesFM-2.5": timesfm_forecast,
             "t0-gamma": t0_forecast, "Timestral-209M": timestral_forecast}
MODEL_LOADERS = {"Chronos-2-full": load_chronos, "TimesFM-2.5": load_timesfm,
                 "t0-gamma": load_t0, "Timestral-209M": load_timestral}


# =====================================================================================
# Statische Gewichte: wie "Statisch" in 01_kombination.ipynb, nur aus train
# =====================================================================================
def _softmax(b):
    e = np.exp(b - b.max())
    return e / e.sum()


@st.cache_data(show_spinner=False)
def static_weights():
    fc = pd.concat([pd.read_parquet(RESULTS_DIR / f"{sid}__{m}.parquet")
                    for sid in SYMBOLS for m in POOL], ignore_index=True)
    scale = pd.concat([load_research_scale(sid) for sid in SYMBOLS])
    weights = {}
    for h in HORIZONS:
        sub = fc[fc["h"] == h]
        wide = sub.set_index(["sid", "origin", "model"])[["y", "target_date", "mean"] + QCOLS].unstack("model")
        mean = wide["mean"][POOL].to_numpy()
        Q = np.stack([wide[c][POOL].to_numpy() for c in QCOLS], axis=-1)
        ok = np.isfinite(mean).all(1) & np.isfinite(Q).all((1, 2))
        y = wide["y"][POOL[0]].to_numpy()
        td = pd.to_datetime(wide["target_date"][POOL[0]].to_numpy())
        s = scale.reindex(wide.index).to_numpy()
        rows = ok & (td <= pd.Timestamp(TRAIN_END)) & np.isfinite(s)
        mean, Q, y, s = mean[rows], Q[rows], y[rows], s[rows]

        def loss(b):
            w = _softmax(b)
            f = mean @ w
            r = y / f
            d = y[:, None] - np.einsum("m,nmq->nq", w, Q)
            pin = np.maximum(_LV * d, (_LV - 1) * d).mean(1)
            return np.mean(r - np.log(r) - 1) + ALPHA * np.mean(pin / s)

        res = minimize(loss, np.zeros(len(POOL)), method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        weights[h] = pd.Series(_softmax(res.x), index=POOL)
    return weights


def new_weights(w_static, avail):
    # Startgewichte der zweiten Kombination: Der Anteil der Foundation Models aus den statischen
    # Gewichten wird gleichmässig auf alle verfügbaren Foundation Models verteilt (sie machen fast
    # dieselben Fehler), HAR und GARCH behalten ihr Gewicht. Nicht gelernt, nur gesetzt.
    fm_share = float(w_static[[m for m in FOUNDATION if m in w_static.index]].sum())
    fm_avail = [m for m in FOUNDATION if m in avail]
    w = pd.Series({m: (fm_share / len(fm_avail) if m in FOUNDATION else float(w_static.get(m, 0.0)))
                   for m in avail})
    return w / w.sum()


def load_research_scale(sid):
    d = pd.read_parquet(DATA_DIR / f"{sid}_daily.parquet")
    sc = d["rv"].rolling(22).mean()
    sc.index = pd.MultiIndex.from_arrays([np.repeat(sid, len(sc)), sc.index], names=["sid", "origin"])
    return sc


# =====================================================================================
# Prognose und Logbuch
# =====================================================================================
def forecast_coin(sym, d, models, price=None):
    rv, ret = d["rv"].to_numpy(float), d["ret"].to_numpy(float)
    results, errors = {BASELINE: rw_forecast(rv), "HAR": har_forecast(rv)}, {}
    try:
        results["GARCH"] = garch_forecast(rv, ret)
    except Exception as e:
        errors["GARCH"] = str(e)
    for name, fn in MODEL_FNS.items():
        if models.get(name) is None:
            errors[name] = models.get(f"{name}_error", "nicht geladen")
            continue
        try:
            results[name] = fn(rv, models[name])
        except Exception as e:
            errors[name] = str(e)

    origin = d.index[-1]
    w_all = static_weights()
    rows, combo_q_by_h = [], {}
    for h in HORIZONS:
        # Hauptprognose: nur der bisherige Pool, statische Gewichte (unverändert)
        avail = [m for m in POOL if m in results]
        w = w_all[h][avail]
        w = w / w.sum()                                              # falls ein Experte fehlt
        combo_mean = sum(w[m] * results[m][h][0] for m in avail)
        combo_q = sum(w[m] * results[m][h][1] for m in avail)
        combo_q_by_h[h] = np.asarray(combo_q, float)
        extra = [(COMBO, (combo_mean, combo_q))]

        # Zweite Kombination (separat): nur wenn mindestens eines der neuen Modelle gerechnet wurde
        avail_new = [m for m in POOL_NEW if m in results]
        w_new = pd.Series(dtype=float)
        if any(m in results for m in NEW_MODELS):
            w_new = new_weights(w_all[h], avail_new)
            extra.append((COMBO_NEW, (sum(w_new[m] * results[m][h][0] for m in avail_new),
                                      sum(w_new[m] * results[m][h][1] for m in avail_new))))

        for m, (mean, q) in [(m, results[m][h]) for m in results] + extra:
            rows.append({"sid": sym, "origin": origin, "h": h, "target_date": origin + pd.Timedelta(days=h),
                         "model": m, "mean": float(mean), **dict(zip(QCOLS, np.asarray(q, float))),
                         "weight": float(w.get(m, np.nan)) if m in POOL else np.nan,
                         "weight_new": float(w_new.get(m, np.nan)),
                         "n_experts": len(avail_new) if m == COMBO_NEW else len(avail)})

    price_rows = []
    if price is not None and origin in price.index:
        p0 = float(price.loc[origin])
        for h, pq in price_range(combo_q_by_h, p0).items():
            price_rows.append({"sid": sym, "origin": origin, "h": h, "target_date": origin + pd.Timedelta(days=h),
                               "p0": p0, **dict(zip(PRICE_COLS, pq))})
    return pd.DataFrame(rows), errors, pd.DataFrame(price_rows)


# =====================================================================================
# Kursspanne aus der Vola-Prognose
# =====================================================================================
PRICE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
PRICE_COLS = [f"p{q:.2f}" for q in PRICE_LEVELS]


def cumulative_variance(q1, q5, h):
    # Varianz der h-Tages-Rendite je RV-Quantil: Summe der Tages-RV über Tag 1..h.
    # Tage 2-4 werden in Logs zwischen der 1- und der 5-Tages-Prognose interpoliert;
    # gleiche Quantile werden addiert (RV ist stark persistent).
    l1, l5 = np.log(q1), np.log(q5)
    return sum(np.exp(l1 + (k - 1) / 4 * (l5 - l1)) for k in range(1, h + 1))


def return_quantiles(var_q, levels=PRICE_LEVELS):
    # Rendite | Varianz v ~ N(0, v); gemischt über die 21 Varianz-Quantile (Gewichte wie beim Mittelwert)
    sd = np.sqrt(np.asarray(var_q, float))
    grid = np.linspace(-8 * sd.max(), 8 * sd.max(), 8001)
    z = grid[:, None] / sd[None, :]
    weights = _QMASS if len(sd) == len(_QMASS) else np.full(len(sd), 1 / len(sd))
    cdf = ndtr(z) @ weights
    return np.interp(levels, cdf, grid)


def price_range(q_rv, p0):
    # q_rv: {h: 21 kombinierte RV-Quantile}; p0: Schlusskurs am Ursprung -> {h: Kurs-Quantile}
    return {h: p0 * np.exp(return_quantiles(cumulative_variance(q_rv[1], q_rv[5], h))) for h in HORIZONS}


def read_log():
    return pd.read_parquet(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()


def write_log(new):
    new = new.assign(made_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
    log = pd.concat([read_log(), new], ignore_index=True)
    log = log.drop_duplicates(["sid", "origin", "h", "model"], keep="last")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log.to_parquet(LOG_PATH)
    return log


def read_price_log():
    path = LIVE_DIR / "price_log.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_price_log(new):
    if new.empty:
        return read_price_log()
    log = pd.concat([read_price_log(), new.assign(made_at=pd.Timestamp.now(tz="UTC").tz_localize(None))],
                    ignore_index=True).drop_duplicates(["sid", "origin", "h"], keep="last")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log.to_parquet(LIVE_DIR / "price_log.parquet")
    return log


def vol(x):
    # Tagesvarianz -> annualisierte Volatilität in Prozent
    return np.sqrt(365 * np.asarray(x, float)) * 100


def qlike(y, f):
    r = y / f
    return r - np.log(r) - 1


def run_update(progress):
    models = {}
    progress.write("Lade Modelle (beim ersten Mal ca. 30–60 Sekunden) …")
    for name, loader in MODEL_LOADERS.items():
        try:
            models[name] = loader()
        except ModuleNotFoundError as e:
            models[name], models[f"{name}_error"] = None, f"Paket nicht installiert ({e.name}), siehe 03_t0_timestral.ipynb"
        except Exception as e:
            models[name], models[f"{name}_error"] = None, str(e)
    all_rows, price_rows, report, new_errors = [], [], [], {}
    for sym, label in SYMBOLS.items():
        progress.write(f"{label}: neue Daten laden …")
        try:
            d, n_new = update_daily(sym)
        except Exception as e:
            report.append(("error", f"{label}: Daten konnten nicht geladen werden ({e}). Es wird mit dem letzten Stand prognostiziert."))
            d, n_new = load_daily(sym), 0
        try:
            price = update_price(sym)
        except Exception as e:
            price = load_price(sym)
            report.append(("warning", f"{label}: Kurs konnte nicht geladen werden ({e})."))
        progress.write(f"{label}: Prognose berechnen …")
        fc, errors, prices = forecast_coin(sym, d, models, price)
        all_rows.append(fc)
        price_rows.append(prices)
        if prices.empty and price is not None:
            report.append(("warning", f"{label}: Für den {d.index[-1]:%d.%m.%Y} fehlt der Schlusskurs, "
                                      "deshalb gibt es diesmal keine Kursspanne."))
        report.append(("info", f"{label}: {n_new} neue Tage, Daten bis {d.index[-1]:%d.%m.%Y}."))
        for m, err in errors.items():
            if m in NEW_MODELS:
                new_errors.setdefault(m, err)
            else:
                report.append(("warning", f"{label}: {LABELS.get(m, m)} fehlt in dieser Prognose ({err}). "
                                          "Die Kombination wurde mit den übrigen Experten neu gewichtet."))
    for m, err in new_errors.items():
        report.append(("warning", f"{LABELS.get(m, m)} fehlt ({err}). Die Hauptprognose ist davon nicht betroffen."))
    write_log(pd.concat(all_rows, ignore_index=True))
    write_price_log(pd.concat(price_rows, ignore_index=True))
    return report


# =====================================================================================
# Oberfläche
# =====================================================================================
CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 980px; }
  [data-testid="stHeading"] h1 { font-weight: 650; letter-spacing: -0.01em;
      font-size: clamp(1.7rem, 7vw, 2.75rem) !important; word-break: normal !important; overflow-wrap: normal !important; }
  .lead { color: #52514e; font-size: 1.02rem; margin-top: -0.6rem; }
  .fc-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0;
             border-top: 1px solid #e1e0d9; border-bottom: 1px solid #e1e0d9; margin: 0.4rem 0 1.4rem; }
  .fc-cell { padding: 1rem 1.2rem 1.1rem 0; }
  .fc-cell + .fc-cell { padding-left: 1.2rem; border-left: 1px solid #e1e0d9; }
  .fc-when { color: #52514e; font-size: 0.92rem; }
  .fc-value { font-size: 2.6rem; font-weight: 650; line-height: 1.15; color: #0b0b0b; }
  .fc-value.is-forecast { color: #2a78d6; }
  .fc-value.is-range { font-size: 1.75rem; line-height: 1.35; white-space: nowrap; }
  .fc-range { color: #52514e; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
  @media (max-width: 640px) {
    .fc-grid { grid-template-columns: 1fr; }
    .fc-cell + .fc-cell { padding-left: 0; border-left: none; border-top: 1px solid #e1e0d9; }
  }
</style>
"""


def fmt_pct(x):
    return f"{x:.0f} %"


def pct(share):
    return f"{100 * share:.0f} %"


def headline(d, fc):
    last_day = d.index[-1]
    cells = [f"""<div class="fc-cell"><div class="fc-when">Zuletzt gemessen, {last_day:%d.%m.}</div>
                 <div class="fc-value">{fmt_pct(vol(d['rv'].iloc[-1]))}</div>
                 <div class="fc-range">Monatsschnitt {fmt_pct(vol(d['rv'].iloc[-22:].mean()))}</div></div>"""]
    for h, when in [(1, "Prognose 1 Tag voraus"), (5, "Prognose 5 Tage voraus")]:
        r = fc[(fc["model"] == COMBO) & (fc["h"] == h)].iloc[0]
        cells.append(f"""<div class="fc-cell"><div class="fc-when">{when}, {r['target_date']:%d.%m.}</div>
                 <div class="fc-value is-forecast">{fmt_pct(vol(r['mean']))}</div>
                 <div class="fc-range">90&nbsp;% wahrscheinlich zwischen {vol(r['q0.05']):.0f} und {vol(r['q0.95']):.0f}&nbsp;%</div></div>""")
    st.markdown(f'<div class="fc-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def history_chart(d, fc, days=90):
    hist = d.iloc[-days:].reset_index().rename(columns={"date": "Datum"})
    hist["Vola"] = vol(hist["rv"])
    c = fc[fc["model"] == COMBO].copy()
    c = c.assign(Datum=c["target_date"], Vola=vol(c["mean"]), Von=vol(c["q0.05"]), Bis=vol(c["q0.95"]),
                 Horizont=c["h"].map({1: "1 Tag voraus", 5: "5 Tage voraus"}))

    axis = dict(labelColor=MUTED, titleColor=INK_2, gridColor=GRID, domainColor="#c3c2b7", tickColor="#c3c2b7")
    x = alt.X("Datum:T", title=None, axis=alt.Axis(format="%d.%m.", **axis))
    y_scale = alt.Scale(zero=False, nice=True)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Datum"], empty=False)

    line = alt.Chart(hist).mark_line(color=INK_2, strokeWidth=2).encode(
        x=x, y=alt.Y("Vola:Q", title="Annualisierte Volatilität (%)", scale=y_scale, axis=alt.Axis(**axis)))
    hover = alt.Chart(hist).mark_point(size=60, filled=True, color=INK_2).encode(
        x=x, y="Vola:Q",
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip("Vola:Q", title="Gemessen (%)", format=".0f")],
    ).add_params(nearest)
    rule = alt.Chart(hist).mark_rule(color=GRID).encode(x=x).transform_filter(nearest)
    band = alt.Chart(c).mark_rule(color=BLUE, strokeWidth=2, opacity=0.55).encode(x=x, y="Von:Q", y2="Bis:Q")
    pts = alt.Chart(c).mark_point(color=BLUE, filled=True, size=90, opacity=1).encode(
        x=x, y="Vola:Q",
        tooltip=[alt.Tooltip("Horizont:N", title="Prognose"), alt.Tooltip("Datum:T", format="%d.%m.%Y"),
                 alt.Tooltip("Vola:Q", title="Erwartet (%)", format=".0f"),
                 alt.Tooltip("Von:Q", title="90 % ab (%)", format=".0f"),
                 alt.Tooltip("Bis:Q", title="90 % bis (%)", format=".0f")])
    label = alt.Chart(c[c["h"] == 5]).mark_text(align="left", dx=10, color=BLUE, fontSize=12).encode(
        x=x, y="Vola:Q", text=alt.value("Prognose"))
    return (rule + line + hover + band + pts + label).properties(height=320).configure_view(strokeWidth=0)


def model_chart(fc, h, order=MAIN_ORDER, weight_col="weight"):
    c = fc[(fc["h"] == h) & fc["model"].isin(order)].copy()
    role = np.select([c["model"] == COMBO, c["model"] == COMBO_NEW], ["Kombination", "Kombination neu"], "Experte")
    c = c.assign(Modell=c["model"].map(LABELS), Vola=vol(c["mean"]), Von=vol(c["q0.05"]), Bis=vol(c["q0.95"]),
                 Rolle=role)
    wcol = c[weight_col] if weight_col in c else pd.Series(np.nan, index=c.index)
    c["Gewicht"] = wcol.map(lambda w: "–" if pd.isna(w) else pct(w))
    sort = [LABELS[m] for m in order if m in set(c["model"])]
    axis = dict(labelColor=MUTED, titleColor=INK_2, gridColor=GRID, domainColor="#c3c2b7", tickColor="#c3c2b7")
    y = alt.Y("Modell:N", sort=sort, title=None, axis=alt.Axis(labelColor=INK_2, labelFontSize=12, domain=False,
                                                                ticks=False, labelLimit=320))
    color = alt.Color("Rolle:N", scale=alt.Scale(domain=["Kombination", "Kombination neu", "Experte"],
                                                 range=[BLUE, ORANGE, MUTED]), legend=None)
    tip = [alt.Tooltip("Modell:N"), alt.Tooltip("Vola:Q", title="Erwartet (%)", format=".0f"),
           alt.Tooltip("Von:Q", title="90 % ab (%)", format=".0f"), alt.Tooltip("Bis:Q", title="90 % bis (%)", format=".0f"),
           alt.Tooltip("Gewicht:N", title="Gewicht")]
    rng = alt.Chart(c).mark_rule(strokeWidth=2).encode(
        y=y, x=alt.X("Von:Q", title="Annualisierte Volatilität (%)", scale=alt.Scale(zero=False),
                     axis=alt.Axis(tickCount=6, **axis)),
        x2="Bis:Q", color=color, tooltip=tip)
    pt = alt.Chart(c).mark_point(filled=True, size=80, opacity=1).encode(y=y, x="Vola:Q", color=color, tooltip=tip)
    return (rng + pt).properties(height=34 * len(sort)).configure_view(strokeWidth=0)


def model_table(fc, order=MAIN_ORDER, weight_col="weight"):
    rows = []
    for m in order:
        r = fc[fc["model"] == m]
        if r.empty:
            continue
        row = {"Modell": LABELS[m]}
        for h, name in [(1, "1 Tag"), (5, "5 Tage")]:
            x = r[r["h"] == h].iloc[0]
            w = x.get(weight_col, np.nan)
            row[f"Erwartet {name}"] = f"{vol(x['mean']):.0f} %"
            row[f"90 %-Bereich {name}"] = f"{vol(x['q0.05']):.0f} – {vol(x['q0.95']):.0f} %"
            row[f"Gewicht {name}"] = "–" if pd.isna(w) else pct(w)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Modell")


def new_models_section(sym, fc, h):
    st.subheader("Neu, nur live: t0 und Timestral", divider=False)
    if not fc["model"].isin(NEW_MODELS).any():
        st.info("t0 und Timestral sind in dieser Prognose nicht dabei. Installation und Prüfung: "
                "`src/03_t0_timestral.ipynb`. Danach neu berechnen.")
        return
    order = [COMBO] + NEW_ORDER
    st.altair_chart(model_chart(fc, h, order=order, weight_col="weight_new"), width="stretch")
    st.caption("Die beiden Modelle sind erst nach unserer Testperiode erschienen und könnten die Daten im "
               "Training gesehen haben. Sie laufen deshalb nur mit und fliessen weder in die Hauptprognose "
               "(blau, zum Vergleich) noch in die Kursspanne ein. Orange: zweite Kombination mit allen sechs "
               "Experten, mit gesetzten Startgewichten (siehe «Methode»). Ob sie besser ist, zeigt das Logbuch.")
    with st.expander("Als Tabelle anzeigen (mit allen Gewichten der zweiten Kombination)"):
        st.dataframe(model_table(fc, order=[COMBO_NEW] + POOL + NEW_MODELS, weight_col="weight_new"), width="stretch")


def fmt_price(x):
    return (f"{x:,.0f}" if x >= 1000 else f"{x:,.2f}").replace(",", "'")


def price_headline(sub):
    p0, origin = sub["p0"].iloc[0], sub["origin"].iloc[0]
    cells = [f"""<div class="fc-cell"><div class="fc-when">Schlusskurs, {origin:%d.%m.}</div>
                 <div class="fc-value is-range">{fmt_price(p0)}&nbsp;$</div>
                 <div class="fc-range">in US-Dollar (USDT)</div></div>"""]
    for h in HORIZONS:
        r = sub[sub["h"] == h].iloc[0]
        down, up = 100 * (1 - r["p0.05"] / p0), 100 * (r["p0.95"] / p0 - 1)
        cells.append(f"""<div class="fc-cell"><div class="fc-when">Spanne für den Schlusskurs am {r['target_date']:%d.%m.}</div>
                 <div class="fc-value is-forecast is-range">{fmt_price(r['p0.05'])} – {fmt_price(r['p0.95'])}</div>
                 <div class="fc-range">90&nbsp;% wahrscheinlich, also −{down:.1f}&nbsp;% bis +{up:.1f}&nbsp;%</div></div>""")
    st.markdown(f'<div class="fc-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def price_chart(price, sub, days=90):
    hist = price.iloc[-days:].rename("Kurs").rename_axis("Datum").reset_index()
    hist["Schlusskurs"] = hist["Kurs"].map(fmt_price)
    c = sub.assign(Datum=sub["target_date"], Mitte=sub["p0.50"], Von90=sub["p0.05"], Bis90=sub["p0.95"],
                   Von50=sub["p0.25"], Bis50=sub["p0.75"],
                   Horizont=sub["h"].map({1: "1 Tag voraus", 5: "5 Tage voraus"}))
    for col in ["Von90", "Bis90", "Von50", "Bis50"]:
        c[f"{col}_txt"] = c[col].map(fmt_price)
    axis = dict(labelColor=MUTED, titleColor=INK_2, gridColor=GRID, domainColor="#c3c2b7", tickColor="#c3c2b7")
    x = alt.X("Datum:T", title=None, axis=alt.Axis(format="%d.%m.", **axis))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Datum"], empty=False)
    line = alt.Chart(hist).mark_line(color=INK_2, strokeWidth=2).encode(
        x=x, y=alt.Y("Kurs:Q", title="Kurs (US-Dollar)", scale=alt.Scale(zero=False),
                     axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), /,/g, \"'\")", **axis)))
    hover = alt.Chart(hist).mark_point(size=60, filled=True, color=INK_2).encode(
        x=x, y="Kurs:Q", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Datum:T", format="%d.%m.%Y"), alt.Tooltip("Schlusskurs:N")],
    ).add_params(nearest)
    rule = alt.Chart(hist).mark_rule(color=GRID).encode(x=x).transform_filter(nearest)
    tip = [alt.Tooltip("Horizont:N", title="Spanne"), alt.Tooltip("Datum:T", format="%d.%m.%Y"),
           alt.Tooltip("Von90_txt:N", title="90 % ab"), alt.Tooltip("Bis90_txt:N", title="90 % bis"),
           alt.Tooltip("Von50_txt:N", title="50 % ab"), alt.Tooltip("Bis50_txt:N", title="50 % bis")]
    r90 = alt.Chart(c).mark_rule(color=BLUE, strokeWidth=2, opacity=0.55).encode(x=x, y="Von90:Q", y2="Bis90:Q", tooltip=tip)
    r50 = alt.Chart(c).mark_rule(color=BLUE, strokeWidth=7, opacity=0.9).encode(x=x, y="Von50:Q", y2="Bis50:Q", tooltip=tip)
    label = alt.Chart(c[c["h"] == 5]).mark_text(align="left", dx=10, color=BLUE, fontSize=12).encode(
        x=x, y="Mitte:Q", text=alt.value("Spanne"))
    return (rule + line + hover + r90 + r50 + label).properties(height=300).configure_view(strokeWidth=0)


def price_section(sym, plog, origin):
    st.subheader("Was das für den Kurs heisst", divider=False)
    price = load_price(sym)
    sub = plog[(plog["sid"] == sym) & (plog["origin"] == origin)] if not plog.empty else plog
    if price is None or sub.empty:
        st.info("Kursdaten fehlen noch. Sie werden beim nächsten Klick auf «Daten laden und Prognose berechnen» geladen.")
        return
    price_headline(sub)
    st.altair_chart(price_chart(price, sub), width="stretch")
    st.caption("Grau: Schlusskurs der letzten 90 Tage. Blau: Spanne aus der Vola-Prognose, dünn 90 %, dick 50 %. "
               "Die Spanne zeigt, wie weit sich der Kurs bei der erwarteten Volatilität bewegen kann. "
               "Sie sagt nichts über die Richtung voraus und ist keine Anlageempfehlung.")


def coin_view(sym, log, plog):
    d = load_daily(sym)
    sub = log[log["sid"] == sym] if not log.empty else log
    if sub.empty:
        st.info("Für diesen Coin gibt es noch keine Prognose. Klicke oben auf «Daten laden und Prognose berechnen».")
        return
    fc = sub[sub["origin"] == sub["origin"].max()]
    if fc["origin"].iloc[0] < d.index[-1]:
        st.warning(f"Die angezeigte Prognose stammt vom Datenstand {fc['origin'].iloc[0]:%d.%m.%Y}. "
                   "Es gibt neuere Daten. Berechne die Prognose neu.")
    n_exp = int(fc.loc[fc["model"] == COMBO, "n_experts"].iloc[0])
    if n_exp < len(POOL):
        st.warning(f"Diese Prognose beruht auf {n_exp} statt {len(POOL)} Experten.")

    headline(d, fc)
    st.altair_chart(history_chart(d, fc), width="stretch")
    st.caption("Grau: gemessene Volatilität der letzten 90 Tage. Blau: Prognose der Kombination mit 90-%-Bereich.")

    price_section(sym, plog, fc["origin"].iloc[0])

    st.subheader("Was die einzelnen Modelle sagen", divider=False)
    h_choice = st.segmented_control("Horizont", ["1 Tag voraus", "5 Tage voraus"], default="1 Tag voraus",
                                    key=f"h_{sym}", label_visibility="collapsed")
    h = 5 if h_choice == "5 Tage voraus" else 1
    st.altair_chart(model_chart(fc, h), width="stretch")
    with st.expander("Als Tabelle anzeigen"):
        st.dataframe(model_table(fc), width="stretch")

    new_models_section(sym, fc, h)


def log_summary(done, order):
    summ = (done.groupby("model").agg(Prognosen=("qlike", "size"), QLIKE=("qlike", "mean"),
                                      Trefferquote=("inside", "mean"))
            .reindex(order).dropna(how="all"))
    summ.index = summ.index.map(LABELS)
    summ["QLIKE"] = summ["QLIKE"].map(lambda x: f"{x:.3f}")
    summ["Trefferquote"] = summ["Trefferquote"].map(pct)
    summ["Prognosen"] = summ["Prognosen"].astype(int)
    return summ.rename(columns={"Trefferquote": "Im 90-%-Bereich"})


def logbook_view(log):
    st.write("Jede berechnete Prognose wird gespeichert. Sobald der Zieltag vorbei ist, "
             "steht hier, was tatsächlich eingetreten ist. Ausgewertet werden nur Prognosen "
             "für Tage nach der Testperiode (ab September 2026).")
    if log.empty:
        st.info("Das Logbuch ist noch leer.")
        return
    st.subheader("Volatilität", divider=False)
    real = pd.concat([load_daily(s)["rv"].rename("y").to_frame().assign(sid=s) for s in SYMBOLS])
    real = real.rename_axis("target_date").reset_index()
    ev = log.merge(real, on=["sid", "target_date"], how="left")
    ev = ev[ev["target_date"] > pd.Timestamp(TEST_END)]
    done = ev.dropna(subset=["y"])

    if done.empty:
        st.info("Noch keine Prognose kann überprüft werden. Der erste Zieltag ist noch nicht vorbei.")
    else:
        done = done.assign(qlike=qlike(done["y"].to_numpy(), done["mean"].to_numpy()),
                           inside=(done["y"] >= done["q0.05"]) & (done["y"] <= done["q0.95"]))
        n = done[done["model"] == COMBO].shape[0]
        st.markdown(f"**{n} überprüfte Prognosen der Kombination.** "
                    "Kleinerer QLIKE heisst genauer. Im 90-%-Bereich sollten etwa 90 % liegen.")
        st.dataframe(log_summary(done, MAIN_ORDER), width="stretch")
        if n < 30:
            st.caption("Mit weniger als etwa 30 Prognosen pro Modell sind diese Werte noch sehr zufällig.")

        new_done = done[done["model"] == COMBO_NEW]
        if not new_done.empty:
            st.markdown("**Neu, nur live: t0 und Timestral.** Verglichen auf denselben Tagen, also nur auf "
                        "Prognosen, bei denen die neuen Modelle schon mitliefen.")
            keys = new_done[["sid", "origin", "h"]].drop_duplicates()
            same = done.merge(keys, on=["sid", "origin", "h"])
            st.dataframe(log_summary(same, [COMBO] + NEW_ORDER + ["Chronos-2-full", "TimesFM-2.5"]), width="stretch")

    combo = ev[ev["model"] == COMBO].sort_values(["target_date", "sid", "h"], ascending=[False, True, True])
    table = pd.DataFrame({
        "Coin": combo["sid"].map(SYMBOLS), "Zieltag": combo["target_date"].dt.strftime("%d.%m.%Y"),
        "Horizont": combo["h"].map({1: "1 Tag", 5: "5 Tage"}),
        "Prognose": [f"{v:.0f} %" for v in vol(combo["mean"])],
        "90 %-Bereich": [f"{vol(a):.0f} – {vol(b):.0f} %" for a, b in zip(combo["q0.05"], combo["q0.95"])],
        "Gemessen": [("–" if pd.isna(y) else f"{vol(y):.0f} %") for y in combo["y"]],
        "Im Bereich": [("offen" if pd.isna(y) else ("ja" if lo <= y <= hi else "nein"))
                       for y, lo, hi in zip(combo["y"], combo["q0.05"], combo["q0.95"])],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    price_logbook_view()


def price_logbook_view():
    plog = read_price_log()
    st.subheader("Kursspanne", divider=False)
    if plog.empty:
        st.info("Noch keine Kursspanne gespeichert.")
        return
    real = []
    for s in SYMBOLS:
        p = load_price(s)
        if p is not None:
            real.append(p.rename("close_real").rename_axis("target_date").reset_index().assign(sid=s))
    ev = plog.merge(pd.concat(real), on=["sid", "target_date"], how="left") if real else plog.assign(close_real=np.nan)
    ev = ev[ev["target_date"] > pd.Timestamp(TEST_END)].sort_values(["target_date", "sid", "h"], ascending=[False, True, True])
    done = ev.dropna(subset=["close_real"])
    if not done.empty:
        in90 = ((done["close_real"] >= done["p0.05"]) & (done["close_real"] <= done["p0.95"])).mean()
        in50 = ((done["close_real"] >= done["p0.25"]) & (done["close_real"] <= done["p0.75"])).mean()
        st.markdown(f"**{len(done)} überprüfte Spannen.** Im 90-%-Bereich: {pct(in90)} (Soll 90 %). "
                    f"Im 50-%-Bereich: {pct(in50)} (Soll 50 %).")
    table = pd.DataFrame({
        "Coin": ev["sid"].map(SYMBOLS), "Schlusskurs am": ev["target_date"].dt.strftime("%d.%m.%Y"),
        "Horizont": ev["h"].map({1: "1 Tag", 5: "5 Tage"}),
        "Spanne 90 %": [f"{fmt_price(a)} – {fmt_price(b)}" for a, b in zip(ev["p0.05"], ev["p0.95"])],
        "Tatsächlich": [("–" if pd.isna(y) else fmt_price(y)) for y in ev["close_real"]],
        "Im Bereich": [("offen" if pd.isna(y) else ("ja" if lo <= y <= hi else "nein"))
                       for y, lo, hi in zip(ev["close_real"], ev["p0.05"], ev["p0.95"])],
    })
    st.dataframe(table, hide_index=True, width="stretch")


def method_view():
    w = static_weights()
    wt = pd.DataFrame({"1 Tag voraus": w[1], "5 Tage voraus": w[5]}).rename(index=LABELS).map(pct)
    st.markdown("""
**Was wird prognostiziert?** Die realisierte Varianz (RV) eines UTC-Tages: die Summe der quadrierten
5-Minuten-Log-Renditen. Angezeigt wird sie als annualisierte Volatilität √(365 · RV).

**Woher kommt die Prognose?** Vier Experten prognostizieren die RV aus ihrem bisherigen Verlauf:
Chronos-2 und TimesFM-2.5 (vortrainierte Foundation Models, zero-shot), HAR und GARCH (Ökonometrie).
Die Kombination mischt ihre Prognosen mit festen Gewichten. Gelernt wurden diese Gewichte auf der
Trainingsperiode bis 31.10.2025, genau wie die Methode «Statisch» in `01_kombination.ipynb`.
""")
    st.dataframe(wt, width="stretch")
    st.markdown("""
**Wie lese ich den 90-%-Bereich?** Mit 90 % Wahrscheinlichkeit liegt die Volatilität des Tages in
diesem Bereich. Je breiter er ist, desto unsicherer sind die Modelle.

**Wie entsteht die Kursspanne?** Die RV eines Tages misst die Varianz der Tagesrendite. Bei bekannter
Varianz ist die Rendite ungefähr normalverteilt mit Mittelwert 0. Weil die Vola-Prognose selbst unsicher ist,
wird über ihre 21 Quantile gemischt. Für 5 Tage werden die Varianzen der Tage 1 bis 5 addiert, die Tage 2 bis 4
werden zwischen der 1- und der 5-Tages-Prognose interpoliert. Die Mitte der Spanne ist der letzte Schlusskurs:
Die Spanne sagt, wie weit sich der Kurs bewegen kann, nicht in welche Richtung. Im Rückblick (train und val)
lag der Kurs in 88 bis 90 % der Fälle in der 90-%-Spanne.

**Was passiert mit den Forschungsdaten?** Nichts. Neue Tage und das Logbuch werden in `src/data/live/`
gespeichert. Die Testperiode (März bis August 2026) wird nicht ausgewertet.

**Was ist mit t0 und Timestral?** Die beiden Foundation Models (t0-gamma von The Forecasting Company,
Timestral-209M von Mistral AI) sind im September bzw. Oktober 2026 erschienen, also nach unserer
Testperiode. Sie könnten die Daten im Training gesehen haben. Deshalb lernen wir aus der Vergangenheit
keine Gewichte für sie. Sie laufen nur mit, separat von der Hauptprognose und der Kursspanne, und werden
über das Logbuch sauber ausgewertet, weil jede Prognose vor dem Zieltag gespeichert wird.

Die **zweite Kombination** nimmt alle sechs Experten. Ihre Gewichte sind gesetzt, nicht gelernt: Die
Foundation Models machen fast dieselben Fehler. Deshalb wird ihr bisheriger Gesamtanteil gleichmässig auf
alle vier verteilt, HAR und GARCH behalten ihr Gewicht.
""")
    wn = pd.DataFrame({"1 Tag voraus": new_weights(w[1], POOL_NEW), "5 Tage voraus": new_weights(w[5], POOL_NEW)})
    st.dataframe(wn.rename(index=LABELS).map(pct), width="stretch")


def main():
    st.set_page_config(page_title="Volatilitätsprognose", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_l:
        st.title("Volatilitätsprognose")
    with top_r:
        clicked = st.button("Daten laden und Prognose berechnen", type="primary", width="stretch")

    log, plog = read_log(), read_price_log()
    stand = max(load_daily(s).index[-1] for s in SYMBOLS)
    made = f", zuletzt berechnet am {log['made_at'].max():%d.%m.%Y um %H:%M} UTC" if not log.empty else ""
    st.markdown(f'<p class="lead">Daten bis {stand:%d.%m.%Y}{made}.</p>', unsafe_allow_html=True)

    if clicked:
        with st.status("Prognose wird berechnet …", expanded=True) as status:
            t0 = time.time()
            try:
                report = run_update(status)
                report.append(("success", f"Prognose berechnet (Dauer {time.time() - t0:.0f} s)."))
            except Exception as e:
                report = [("error", f"Die Prognose konnte nicht berechnet werden: {e}")]
        st.session_state["report"] = report
        st.rerun()

    for kind, msg in st.session_state.pop("report", []):
        if kind != "info":
            getattr(st, kind)(msg)

    tabs = st.tabs(list(SYMBOLS.values()) + ["Logbuch", "Methode"])
    for tab, sym in zip(tabs, SYMBOLS):
        with tab:
            coin_view(sym, log, plog)
    with tabs[len(SYMBOLS)]:
        logbook_view(log)
    with tabs[len(SYMBOLS) + 1]:
        method_view()


if __name__ == "__main__":
    main()
