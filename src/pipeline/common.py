"""Einstellungen und Hilfsfunktionen, die Notebooks und App teilen.

Eine Änderung hier wirkt auf die Notebooks und auf die App. Gespeicherte Prognosen in results/ werden
dadurch nicht neu gerechnet: Wer z.B. EST_WINDOW ändert, muss die betroffenen Parquet-Dateien löschen.
"""
import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]   # = native Chronos-2-Quantile
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
LV = np.asarray(QUANTILES)

QMASS = (np.r_[LV[1:], 1.0] - np.r_[0.0, LV[:-1]]) / 2     # Wahrscheinlichkeitsmasse je Quantil (Trapez)
QMASS = QMASS / QMASS.sum()

# Alle Werte in Zeitschritten der Daten (binance_1h: Stunden). Fenster, die in Tagen gemeint sind
# (HAR, EWMA, Features), rechnen experts.py und features.py mit steps_per_day() selbst um.
HORIZONS = [1, 24]              # Schritte voraus (1 h, 24 h)
MIN_HISTORY = 24 * 60           # Schritte Historie vor dem ersten Ursprung (60 Tage)
EST_WINDOW = 24 * 90            # rollierendes Schätzfenster für RW / EWMA / HAR / GARCH (90 Tage)
GARCH_REFIT_EVERY = 24          # GARCH alle k Ursprünge neu schätzen (täglich)
CONTEXT = 1024                  # Kontext der Foundation Models (Schritte, 1024 h ≈ 43 Tage)
FM_BATCH = 128
ORIGIN_START = "2025-01-01"


def steps_per_day(index):
    """Zeitschritte pro Tag aus einem Datums-Index (Stundendaten: 24, Tagesdaten: 1)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return 1
    step = pd.Series(idx).diff().median()
    return max(1, int(round(pd.Timedelta(days=1) / step)))

def qlike(y, f):
    r = y / f
    return r - np.log(r) - 1


def mean_from_quantiles(q):
    return q @ QMASS                                             # E[Y] als Integral der Quantilfunktion


def cap_quantiles(fc, k=8.0):
    """Quantile auf [Median·e^-k, Median·e^k] begrenzen, Mittelwert in gekappten Zeilen neu aus den Quantilen.
    Chronos-2 und TiRex liefern auf Volatilität vereinzelt q0.99 bis e^40 über dem Median. Mit k=8 trifft
    das nur diese Ausreisser; breite, aber echte Verteilungen (RW, HAR, GARCH: max. ~e^7) bleiben unverändert."""
    Q = fc[QCOLS].to_numpy()
    med = fc["q0.50"].to_numpy()[:, None]
    Qc = np.clip(Q, med * np.exp(-k), med * np.exp(k))
    hit = (np.abs(Qc - Q) > 0).any(axis=1)
    fc = fc.copy()
    fc[QCOLS] = Qc
    fc.loc[hit, "mean"] = mean_from_quantiles(Qc[hit])
    return fc


def dist_from_log_resid(log_point, resid):
    # Log-Punktprognose + empirische Log-Residuen -> (Mittelwert, Quantile) auf RV-Skala
    resid = resid[np.isfinite(resid)]
    q = np.exp(log_point + np.quantile(resid, LV))
    mean = np.exp(log_point) * np.mean(np.exp(resid))            # Smearing-Schätzer
    return mean, q
