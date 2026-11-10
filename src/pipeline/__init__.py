"""Gemeinsamer Code der Pipeline (Notebooks 02–04) und der App.

- common:   Quantile, Horizonte, Schätzfenster, Hilfsfunktionen
- experts:  die Experten (RW, EWMA, HAR, GARCH, Chronos-2, TimesFM-2.5, t0, Timestral)
- features: Features und Panels für das Gate
- combine:  Panel, Loss, Baselines, Gate, Walk-Forward-Folds

Die Notebooks importieren von hier, die App ebenso. So rechnen beide garantiert gleich.
Einstellungen, die für Notebooks und App gleich sein müssen, stehen in common.py.
"""
