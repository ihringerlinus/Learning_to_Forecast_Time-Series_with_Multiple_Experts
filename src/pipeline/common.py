"""Einstellungen und Hilfsfunktionen, die Notebooks und App teilen.

Eine Änderung hier wirkt auf die Notebooks und auf die App. Gespeicherte Prognosen in results/ werden
dadurch nicht neu gerechnet: Wer z.B. EST_WINDOW ändert, muss die betroffenen Parquet-Dateien löschen.
"""
import numpy as np

QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]   # = native Chronos-2-Quantile
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
LV = np.asarray(QUANTILES)

QMASS = (np.r_[LV[1:], 1.0] - np.r_[0.0, LV[:-1]]) / 2     # Wahrscheinlichkeitsmasse je Quantil (Trapez)
QMASS = QMASS / QMASS.sum()

HORIZONS = [1, 5]               # Tage voraus
MIN_HISTORY = 500               # Tage Historie vor dem ersten Ursprung
EST_WINDOW = 1000               # rollierendes Schätzfenster für RW / EWMA / HAR / GARCH
GARCH_REFIT_EVERY = 5           # GARCH alle k Ursprünge neu schätzen
CONTEXT = 1024                  # Kontext der Foundation Models (Tage)
FM_BATCH = 128


def qlike(y, f):
    r = y / f
    return r - np.log(r) - 1


def mean_from_quantiles(q):
    return q @ QMASS                                             # E[Y] als Integral der Quantilfunktion


def dist_from_log_resid(log_point, resid):
    # Log-Punktprognose + empirische Log-Residuen -> (Mittelwert, Quantile) auf RV-Skala
    resid = resid[np.isfinite(resid)]
    q = np.exp(log_point + np.quantile(resid, LV))
    mean = np.exp(log_point) * np.mean(np.exp(resid))            # Smearing-Schätzer
    return mean, q
