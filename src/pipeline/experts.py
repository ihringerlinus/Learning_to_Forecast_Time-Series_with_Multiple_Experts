"""Die Experten: Rolling-Origin-Prognosen von Mittelwert und 21 Quantilen der RV.

Jeder Experte sieht für den Ursprung t nur Daten bis einschliesslich t. Ökonometrische Modelle liefern
Quantile aus den empirischen Log-Residuen und den Mittelwert per Smearing (Duan 1983), Foundation Models
liefern Quantile direkt, der Mittelwert entsteht durch Integration der Quantile.

Neuen Experten hinzufügen: Klasse mit `name` und `forecast(s, origins)` schreiben und in EXPERTS eintragen.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import common as C
from .common import dist_from_log_resid, mean_from_quantiles

try:
    from tqdm.auto import tqdm
except ImportError:                                 # tqdm ist optional
    def tqdm(x, **kwargs):
        return x


class RVSeries:
    """Eine Serie: Tages-RV und Renditen. `origins` sind die Indizes der Ursprünge.

    Standard (Notebooks): nur Ursprünge, deren Ziel für alle Horizonte bekannt ist.
    `all_origins=True` (App): bis zum letzten Tag, die Ziele der letzten Ursprünge sind noch unbekannt.
    """

    def __init__(self, sid, df, all_origins=False):
        self.sid = sid
        self.dates = df.index
        self.rv = df["rv"].to_numpy(float)
        self.ret = df["ret"].to_numpy(float)
        self.logrv = np.log(self.rv)
        last = len(df) if all_origins else len(df) - max(C.HORIZONS)
        self.origins = np.arange(C.MIN_HISTORY - 1, last)


class Expert:
    name = "base"

    def forecast(self, s, origins):
        raise NotImplementedError

    @staticmethod
    def _alloc(n):
        return (np.full((n, len(C.HORIZONS)), np.nan),
                np.full((n, len(C.HORIZONS), len(C.QUANTILES)), np.nan))


# =====================================================================================
# Ökonometrische Experten
# =====================================================================================
class RandomWalk(Expert):
    name = "RW"

    def forecast(self, s, origins):
        L = s.logrv
        mean, q = self._alloc(len(origins))
        for i, t in enumerate(origins):
            lo = max(0, t - C.EST_WINDOW + 1)
            for j, h in enumerate(C.HORIZONS):
                resid = L[lo + h: t + 1] - L[lo: t + 1 - h]
                mean[i, j], q[i, j] = dist_from_log_resid(L[t], resid)
        return mean, q


class EWMA(Expert):
    name = "EWMA"

    def __init__(self, halflife=5):
        self.halflife = halflife

    def forecast(self, s, origins):
        L = s.logrv
        sm = pd.Series(L).ewm(halflife=self.halflife).mean().to_numpy()   # kausal: nur Daten bis t
        mean, q = self._alloc(len(origins))
        for i, t in enumerate(origins):
            lo = max(0, t - C.EST_WINDOW + 1)
            for j, h in enumerate(C.HORIZONS):
                resid = L[lo + h: t + 1] - sm[lo: t + 1 - h]
                mean[i, j], q[i, j] = dist_from_log_resid(sm[t], resid)
        return mean, q


class HAR(Expert):
    # log-HAR (Corsi 2009): direkte Prognose je Horizont, OLS im rollierenden Fenster
    name = "HAR"

    def forecast(self, s, origins):
        rv = pd.Series(s.rv)
        X = np.column_stack([np.ones(len(rv)), np.log(rv),
                             np.log(rv.rolling(5).mean()), np.log(rv.rolling(22).mean())])
        L = s.logrv
        mean, q = self._alloc(len(origins))
        for i, t in enumerate(origins):
            lo = max(21, t - C.EST_WINDOW + 1)
            for j, h in enumerate(C.HORIZONS):
                rows = np.arange(lo, t - h + 1)        # nur Zeilen, deren Ziel u+h <= t schon bekannt ist
                beta, *_ = np.linalg.lstsq(X[rows], L[rows + h], rcond=None)
                resid = L[rows + h] - X[rows] @ beta
                mean[i, j], q[i, j] = dist_from_log_resid(X[t] @ beta, resid)
        return mean, q


class GARCH(Expert):
    # GARCH(1,1)-t auf Tagesrenditen. Niveau und Streuung über log(RV_u / sigma²_u) auf die RV kalibriert.
    name = "GARCH"

    def __init__(self, dist="t", refit_every=None):
        self.dist = dist
        self.refit_every = C.GARCH_REFIT_EVERY if refit_every is None else refit_every

    def forecast(self, s, origins):
        import warnings
        with warnings.catch_warnings():              # arch warnt bei jedem Refit (Startwerte, Konvergenz)
            warnings.simplefilter("ignore")
            return self._forecast(s, origins)

    def _forecast(self, s, origins):
        from arch import arch_model
        y_all = 100 * s.ret                      # Prozent-Renditen für eine stabile Optimierung
        mean, q = self._alloc(len(origins))
        params = None
        for i, t in enumerate(origins):
            lo = max(0, t - C.EST_WINDOW + 1)
            am = arch_model(y_all[lo: t + 1], mean="Constant", vol="GARCH", p=1, q=1,
                            dist=self.dist, rescale=False)
            res = None
            if params is None or i % self.refit_every == 0:
                try:
                    res = am.fit(disp="off", show_warning=False,
                                 starting_values=None if params is None else params.to_numpy())
                    params = res.params
                except Exception:
                    res = None
            if res is None:                        # zwischen Refits: alte Parameter auf neue Daten
                res = am.fix(params)
            sig2_in = np.asarray(res.conditional_volatility) ** 2 / 1e4
            sig2_fc = res.forecast(horizon=max(C.HORIZONS), reindex=False).variance.to_numpy()[-1] / 1e4
            resid = s.logrv[lo: t + 1] - np.log(sig2_in)
            for j, h in enumerate(C.HORIZONS):
                mean[i, j], q[i, j] = dist_from_log_resid(np.log(sig2_fc[h - 1]), resid)
        return mean, q


# =====================================================================================
# Foundation Models (zero-shot auf der log-RV, Kontext CONTEXT Tage)
# =====================================================================================
DEVICE = None    # None = cuda falls verfügbar, sonst cpu


def _device():
    import torch
    return DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")


def contexts_for(s, origins):
    L = s.logrv.astype(np.float32)
    return [L[max(0, t - C.CONTEXT + 1): t + 1] for t in origins]


def _log_to_dist(log_q):
    # (n, H, 21) Log-Quantile -> mean (n, nH), q (n, nH, 21) für HORIZONS
    log_q = np.asarray(log_q, float)[:, [h - 1 for h in C.HORIZONS], :]
    q = np.sort(np.exp(log_q), axis=-1)
    return mean_from_quantiles(q), q


def _numpy(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, float)


class Chronos2(Expert):
    # cross_learning=False: sonst teilen Kontexte im Batch Information (Leakage)
    def __init__(self, model_id="amazon/chronos-2", name="Chronos-2-full", chunk=1024):
        from chronos import Chronos2Pipeline
        self.name, self.chunk = name, chunk
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=_device())

    def forecast(self, s, origins):
        ctx = contexts_for(s, origins)
        preds = []
        for k in tqdm(range(0, len(ctx), self.chunk), desc=f"{self.name} {s.sid}", leave=False):
            qs, _ = self.pipe.predict_quantiles(
                ctx[k: k + self.chunk], prediction_length=max(C.HORIZONS), quantile_levels=C.QUANTILES,
                batch_size=C.FM_BATCH, context_length=C.CONTEXT,
                cross_learning=False)       # sonst teilen Kontexte im Batch Information -> Leakage
            preds += [p[0].detach().float().cpu().numpy() for p in qs]
        return _log_to_dist(np.stack(preds))


_ND = NormalDist()
NATIVE9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_Z9 = np.array([_ND.inv_cdf(p) for p in NATIVE9])
_Z21 = np.array([_ND.inv_cdf(p) for p in C.QUANTILES])


def expand_quantiles(logq9):
    # (..., 9) Log-Quantile auf 0.1..0.9 -> (..., 21) Log-Quantile auf QUANTILES
    # innen linear auf der Normal-Skala, an den Rändern mit der lokalen Steigung
    q = np.asarray(logq9, float)
    out = np.empty(q.shape[:-1] + (len(C.QUANTILES),))
    lo_slope = np.maximum((q[..., 2] - q[..., 0]) / (_Z9[2] - _Z9[0]), 1e-6)
    hi_slope = np.maximum((q[..., 8] - q[..., 6]) / (_Z9[8] - _Z9[6]), 1e-6)
    for k, z in enumerate(_Z21):
        if z < _Z9[0] - 1e-12:
            out[..., k] = q[..., 0] + lo_slope * (z - _Z9[0])
        elif z > _Z9[-1] + 1e-12:
            out[..., k] = q[..., 8] + hi_slope * (z - _Z9[-1])
        else:
            j = int(np.clip(np.searchsorted(_Z9, z), 1, 8))
            w = (z - _Z9[j - 1]) / (_Z9[j] - _Z9[j - 1])
            out[..., k] = (1 - w) * q[..., j - 1] + w * q[..., j]
    return out


class TimesFM25(Expert):
    # liefert nur die Quantile 0.1 … 0.9, die übrigen werden in Logs ergänzt (expand_quantiles)
    name = "TimesFM-2.5"

    def __init__(self, repo="google/timesfm-2.5-200m-pytorch", chunk=1024):
        import timesfm
        self.chunk = chunk
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
        self.model.compile(timesfm.ForecastConfig(
            max_context=C.CONTEXT, max_horizon=max(C.HORIZONS), normalize_inputs=True,
            per_core_batch_size=C.FM_BATCH,      # Default 1 wäre ca. 8x langsamer
            use_continuous_quantile_head=True, fix_quantile_crossing=True,
            infer_is_positive=False))            # log-RV ist negativ

    def forecast(self, s, origins):
        ctx = contexts_for(s, origins)
        parts = []
        for k in tqdm(range(0, len(ctx), self.chunk), desc=f"TimesFM {s.sid}", leave=False):
            # list(...) nötig: forecast() hängt Padding an die übergebene Liste an
            _, q = self.model.forecast(horizon=max(C.HORIZONS), inputs=list(ctx[k: k + self.chunk]))
            parts.append(np.asarray(q)[:, :max(C.HORIZONS), 1:])    # Index 0 = Punktwert, 1..9 = q0.1..q0.9
        logq9 = np.concatenate(parts)[:, [h - 1 for h in C.HORIZONS], :]
        q = np.sort(np.exp(expand_quantiles(logq9)), axis=-1)
        return mean_from_quantiles(q), q


def _pad_left(ctx):
    # Kontexte unterschiedlicher Länge -> (B, T) mit NaN links (= fehlend für t0 und Timestral)
    T = max(len(c) for c in ctx)
    out = np.full((len(ctx), T), np.nan, dtype=np.float32)
    for i, c in enumerate(ctx):
        out[i, T - len(c):] = c
    return out


class T0Gamma(Expert):
    """t0-gamma (The Forecasting Company, Sept. 2026). Erschienen nach unserer Testperiode: möglicherweise
    kontaminiert, deshalb nicht im Standard-Pool. Liefert direkt unsere 21 Quantile.
    symmetric_averaging und trim_leading_missing wie vom Hersteller empfohlen, clip_nonnegative aus (log-RV < 0)."""
    name = "t0-gamma"

    def __init__(self, repo="theforecastingcompany/t0-gamma", batch=64):
        from t0 import T0Forecaster
        self.batch = batch
        self.model = T0Forecaster.from_pretrained(repo).eval()

    def predict_log(self, ctx):
        out = self.model.predict(_pad_left(ctx), horizon=max(C.HORIZONS), quantile_levels=C.QUANTILES,
                                 trim_leading_missing=True, symmetric_averaging=True,
                                 clip_nonnegative=False, snap_integers=False)
        return _numpy(out.quantiles).reshape(len(ctx), max(C.HORIZONS), len(C.QUANTILES))

    def forecast(self, s, origins):
        ctx = contexts_for(s, origins)
        parts = [self.predict_log(ctx[k: k + self.batch])
                 for k in tqdm(range(0, len(ctx), self.batch), desc=f"t0 {s.sid}", leave=False)]
        return _log_to_dist(np.concatenate(parts))


class Timestral(Expert):
    """Timestral-209M (Mistral AI, Okt. 2026). Erschienen nach unserer Testperiode: möglicherweise
    kontaminiert, deshalb nicht im Standard-Pool. Liefert seine 21 trainierten Quantile (= unsere)."""
    name = "Timestral-209M"

    def __init__(self, repo="mistralai/Timestral-209M", batch=64):
        from timestral import TimestralForecaster
        self.batch = batch
        self.model = TimestralForecaster.from_pretrained(repo, device=_device())
        levels = np.asarray(self.model.quantile_levels, float)
        if len(levels) != len(C.LV) or not np.allclose(levels, C.LV):
            raise ValueError(f"Timestral liefert andere Quantilstufen als erwartet: {tuple(levels)}")

    def predict_log(self, ctx):
        out = self.model.forecast(_pad_left(ctx)[:, None, :], prediction_length=max(C.HORIZONS),
                                  symmetric_averaging=True, clip_nonnegative=False)
        return _numpy(out.quantiles)[:, 0]                           # (B, H, 21)

    def forecast(self, s, origins):
        ctx = contexts_for(s, origins)
        parts = [self.predict_log(ctx[k: k + self.batch])
                 for k in tqdm(range(0, len(ctx), self.batch), desc=f"Timestral {s.sid}", leave=False)]
        return _log_to_dist(np.concatenate(parts))


EXPERTS = {
    "RW": RandomWalk,
    "EWMA": EWMA,
    "HAR": HAR,
    "GARCH": GARCH,
    "Chronos-2-full": lambda: Chronos2("amazon/chronos-2", "Chronos-2-full"),
    "Chronos-2": lambda: Chronos2("autogluon/chronos-2-small", "Chronos-2"),
    "TimesFM-2.5": TimesFM25,
    "t0-gamma": T0Gamma,
    "Timestral-209M": Timestral,
}
FOUNDATION = ["Chronos-2-full", "Chronos-2", "TimesFM-2.5", "t0-gamma", "Timestral-209M"]
NEW_MODELS = ["t0-gamma", "Timestral-209M"]     # nach der Testperiode erschienen, möglicherweise kontaminiert


def forecast_frame(name, s, origins, mean, q):
    """Prognosen im Format der Ergebnisdateien: eine Zeile pro Ursprung und Horizont.
    Ist das Ziel noch nicht bekannt (letzte Ursprünge in der App), sind `y` = NaN und
    `target_date` = Ursprung + h Kalendertage."""
    origins = np.asarray(origins)
    parts = []
    for j, h in enumerate(C.HORIZONS):
        tgt = origins + h
        known = tgt < len(s.dates)
        td = pd.DatetimeIndex(s.dates[origins]) + pd.Timedelta(days=h)
        td = td.where(~known, pd.DatetimeIndex(s.dates[np.minimum(tgt, len(s.dates) - 1)]))
        df = pd.DataFrame({"sid": s.sid, "model": name, "h": h,
                           "origin": s.dates[origins], "target_date": td,
                           "y": np.where(known, s.rv[np.minimum(tgt, len(s.rv) - 1)], np.nan),
                           "mean": mean[:, j]})
        parts.append(pd.concat([df, pd.DataFrame(q[:, j, :], columns=C.QCOLS)], axis=1))
    return pd.concat(parts, ignore_index=True)
