"""Live-Betrieb der App: derselbe Walk-Forward wie in 04_training, nur in Echtzeit.

1. Prognosen nachführen: Für jeden Ursprung nach dem Ende der Forschungsdaten rechnen die Experten
   (Code aus experts.py) und speichern ihre Prognosen in data/live/forecasts/. Bekannte Ziele werden
   bei jedem Lauf nachgetragen.
2. Panels bauen: Forschungsprognosen (results/) + Live-Prognosen, Features wie in 03 (features.py).
3. Monatlich trainieren: Für jeden Zielmonat werden «Statisch» und «Gating linear» einmal trainiert,
   genau wie ein Fold in 04 (combine.make_fold): Training mit allen Zielen bis Monatsbeginn - h,
   lambda auf den letzten TUNE_DAYS gewählt. Zusätzlich bleibt die gesperrte Testperiode
   (test_start bis Ende der Forschungsdaten) aus dem Training. Die Modelle werden gespeichert, alle
   Prognosen eines Monats verwenden dasselbe Modell.
4. Heutige Prognose: Gewichte beider Methoden für den letzten Ursprung.
"""
import json
import pickle
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import combine as cb
from . import common as C
from . import experts as ex
from . import features as F

STATIC, GATE = "Statisch", "Gating linear"


@dataclass
class Settings:
    sids: list
    pool: list
    feature_cfg: F.FeatureConfig
    lambdas: list
    tune_days: int
    gap_days: int
    train_start: object
    window_days: object
    alpha: float
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    research_dir: Path              # Prognosen der Forschung (results/)
    research_daily_dir: Path        # Tagesdaten der Forschung (Ende = Ende der Forschungsdaten)
    live_dir: Path
    run: str
    notes: list = field(default_factory=list)

    @property
    def forecast_dir(self):
        return self.live_dir / "forecasts"

    @property
    def model_dir(self):
        return self.live_dir / "models"


def load_settings(src_dir, live_dir, dataset="binance", run=None):
    """Konfiguration aus data/<dataset>.json und dem Lauf in results/runs/<run>/config.json (04_training).
    So verwendet die App dieselben Features, lambdas und Tuning-Tage wie die Forschung."""
    src_dir, live_dir = Path(src_dir), Path(live_dir)
    run = run or f"{dataset}_walkforward"
    info = json.loads((src_dir / "data" / f"{dataset}.json").read_text())
    sids = info["sids"]
    cfg_path = src_dir / "results" / "runs" / run / "config.json"
    rc = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    notes = [] if rc else [f"{cfg_path} fehlt, es gelten die Standardwerte aus 03/04."]
    pool = rc.get("POOL") or ["Chronos-2-full", "TimesFM-2.5", "HAR", "GARCH"]
    panel = rc.get("panel", {})
    if panel.get("legacy"):
        fcfg = F.FeatureConfig.legacy(pool, sids)
    else:
        names = {f.name for f in fields(F.FeatureConfig)} - {"pool", "sids"}
        fcfg = F.FeatureConfig(pool, sids, **{k: v for k, v in panel.items() if k in names})
    daily_dir = src_dir / info["daily_dir"]
    test_end = max(pd.read_parquet(daily_dir / f"{sid}_daily.parquet").index.max() for sid in sids)
    return Settings(
        sids=sids, pool=pool, feature_cfg=fcfg,
        lambdas=rc.get("LAMBDAS", [1e-4, 1e-3, 1e-2, 1e-1, 1.0]), tune_days=rc.get("TUNE_DAYS", 90),
        gap_days=rc.get("GAP_DAYS", 0), train_start=rc.get("TRAIN_START"), window_days=rc.get("WINDOW_DAYS"),
        alpha=rc.get("ALPHA", 1.0), test_start=pd.Timestamp(info["test_start"]), test_end=pd.Timestamp(test_end),
        research_dir=src_dir / info["forecast_dir"], research_daily_dir=daily_dir, live_dir=live_dir,
        run=run, notes=notes)


# =====================================================================================
# 1. Prognosen nachführen
# =====================================================================================
def _store_path(st, sid, model):
    return st.forecast_dir / f"{sid}__{model}.parquet"


def read_store(st, sid, model):
    p = _store_path(st, sid, model)
    return pd.read_parquet(p) if p.exists() else None


def _live_from(st, sid):
    # erster Ursprung, den die Forschung nicht mehr prognostiziert (ihr letzter Ursprung + 1 Zeile)
    dates = pd.read_parquet(st.research_daily_dir / f"{sid}_daily.parquet").index
    return dates[len(dates) - max(C.HORIZONS)]


def _refresh_targets(df, s):
    # Ziele, die inzwischen bekannt sind, nachtragen (Ziel = h Zeilen nach dem Ursprung, wie in 02)
    pos = pd.Series(np.arange(len(s.dates)), index=s.dates)
    i = pos.reindex(df["origin"]).to_numpy()
    ok = np.isfinite(i)
    tgt = np.where(ok, i, -10 ** 9).astype(int) + df["h"].to_numpy()
    known = ok & (tgt < len(s.dates)) & (tgt >= 0)
    df = df.copy()
    df.loc[known, "y"] = s.rv[tgt[known]]
    df.loc[known, "target_date"] = s.dates[tgt[known]]
    return df


def update_forecasts(st, daily, models, get_expert, progress=print):
    """Rechnet für jedes Modell die fehlenden Ursprünge seit Ende der Forschungsdaten. get_expert(name)
    liefert ein Expertenobjekt oder wirft eine Exception (z.B. Paket fehlt). Rückgabe: {Modell: Fehler}."""
    st.forecast_dir.mkdir(parents=True, exist_ok=True)
    errors, experts = {}, {}
    for sid in st.sids:
        s = ex.RVSeries(sid, daily[sid], all_origins=True)
        start = _live_from(st, sid)
        for m in models:
            if m in errors:
                continue
            store = read_store(st, sid, m)
            have = set(pd.DatetimeIndex(store["origin"])) if store is not None else set()
            todo = [t for t in s.origins if s.dates[t] >= start and s.dates[t] not in have]
            if todo:
                try:
                    if m not in experts:
                        experts[m] = get_expert(m)
                    progress(f"{m}: {len(todo)} Ursprünge für {sid} …")
                    mean, q = experts[m].forecast(s, np.asarray(todo))
                    new = ex.forecast_frame(m, s, todo, mean, q)
                    store = new if store is None else pd.concat([store, new], ignore_index=True)
                except Exception as e:
                    errors[m] = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                    continue
            if store is not None:
                store = _refresh_targets(store.sort_values(["h", "origin"]).reset_index(drop=True), s)
                store.to_parquet(_store_path(st, sid, m))
    return errors


def latest(st, sid, model, origin):
    """Live-Prognose eines Modells für einen Ursprung: {h: (mean, q)} oder None."""
    df = read_store(st, sid, model)
    if df is None:
        return None
    df = df[df["origin"] == origin]
    if df.empty or not set(C.HORIZONS) <= set(df["h"]):
        return None
    return {int(r["h"]): (float(r["mean"]), r[C.QCOLS].to_numpy(float)) for _, r in df.iterrows()}


# =====================================================================================
# 2. Panels
# =====================================================================================
def assemble(st):
    """Prognosen des Pools: Forschung (results/) + Live (data/live/forecasts/), ohne Doppel."""
    parts = []
    for sid in st.sids:
        for m in st.pool:
            p = st.research_dir / f"{sid}__{m}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p).drop(columns="split", errors="ignore").assign(_src=0))
            live = read_store(st, sid, m)
            if live is not None:
                parts.append(live.assign(_src=1))
    fc = pd.concat(parts, ignore_index=True)
    fc["origin"] = pd.to_datetime(fc["origin"])
    fc["target_date"] = pd.to_datetime(fc["target_date"])
    fc = fc.sort_values("_src").drop_duplicates(["sid", "model", "h", "origin"], keep="first")
    return fc.drop(columns="_src").reset_index(drop=True)


def build_panels(st, daily):
    frames, groups = F.build_panels(assemble(st), daily, st.feature_cfg)
    return {h: cb.Panel(df, h, st.pool, groups) for h, df in frames.items()}, groups


# =====================================================================================
# 3. Monatlich trainieren
# =====================================================================================
def methods(st):
    return [cb.Method(STATIC, partial(cb.Gate, groups=())),
            cb.Method(GATE, cb.Gate, {"lam": list(st.lambdas)})]


def test_mask(st, P):
    td = P.td
    return np.asarray((td >= st.test_start) & (td <= st.test_end))


def train_month(st, P, month):
    """Ein Fold wie in 04 für den Zielmonat `month`, ohne die Testperiode. Rückgabe wie gespeichert."""
    cb.ALPHA = st.alpha
    month = pd.Timestamp(month).to_period("M").start_time
    fold = cb.make_fold(P, month, np.zeros(len(P), bool), st.tune_days, st.gap_days,
                        st.train_start, st.window_days, exclude=test_mask(st, P))
    if fold.train.sum() == 0:
        raise ValueError(f"Keine Trainingsdaten für {month:%m.%Y}, h={P.h}")
    out = {"month": month, "h": P.h, "pool": list(st.pool), "features": st.feature_cfg.to_dict(),
           "trained_at": pd.Timestamp.now(tz="UTC").tz_localize(None), "combiners": {}, "info": {}}
    td = P.td[fold.train]
    for m in methods(st):
        c, best, tuning = cb.fit_predict(P, m, fold)
        out["combiners"][m.name] = c
        out["info"][m.name] = {"lam": best.get("lam"), "n_train": int(fold.train.sum()),
                               "train_from": td.min(), "train_to": td.max(),
                               "n_tune": int(fold.tune_eval.sum()),
                               "tuning": tuning}
    return out


def get_month_model(st, P, month):
    """Gespeichertes Modell des Monats laden oder einmal trainieren (gleiche Konfiguration vorausgesetzt)."""
    month = pd.Timestamp(month).to_period("M").start_time
    path = st.model_dir / f"{month:%Y-%m}_h{P.h}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            mod = pickle.load(f)
        if mod["pool"] == list(st.pool) and mod["features"] == st.feature_cfg.to_dict():
            return mod, False
    mod = train_month(st, P, month)
    st.model_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mod, f)
    return mod, True


# =====================================================================================
# 4. Heutige Prognose
# =====================================================================================
def today(st, panels, sid, origin, progress=print):
    """Gewichte von «Statisch» und «Gating linear» am Ursprung, je Horizont.
    Rückgabe {h: {"Statisch": w (M,), "Gating linear": w (M,), "model": Modell-Dict}} bzw. None je h,
    falls am Ursprung nicht alle Experten des Pools prognostiziert haben."""
    out = {}
    for h, P in panels.items():
        row = np.asarray((P.meta["sid"] == sid) & (P.meta["origin"] == origin))
        if not row.any():
            out[h] = None
            continue
        Pr = P.take(row)
        month = Pr.td[0].to_period("M").start_time
        mod, trained = get_month_model(st, P, month)
        if trained:
            progress(f"Neues Modell für {month:%m.%Y}, h={h} trainiert.")
        out[h] = {name: c.weights(Pr)[0] for name, c in mod["combiners"].items()}
        out[h]["model"] = mod
    return out
