"""Features für das Gate und Panels (eine Zeile pro Serie und Ursprung, ein Panel pro Horizont).

Alle Features sind am Ursprung t bekannt. Sie werden hier nicht standardisiert: Mittelwert und Streuung
schätzt das Gate pro Fold nur auf den Trainingszeilen.

| Gruppe     | Features                                                      |
|------------|---------------------------------------------------------------|
| market     | logrv_dev, trend, level_rel (bzw. level), volvol, ret1, ret5  |
| disagree   | ln mu_m - Mittel der ln mu_j                                  |
| recent     | ln L_m - Mittel der ln L_j, L = EWMA-QLIKE, um h verschoben    |
| calendar   | weekend; optional hour_sin/cos, us_open, fomc, cpi, event_day  |
|            | (alle am Zielzeitpunkt, siehe cfg.calendar und exo.py)         |
| exo        | alle Spalten exo_* der Tagesdaten                             |
| volume     | exo_vol_*  (Teilmenge von exo, Quelle exo.py)                 |
| dvol       | exo_dvol_* (Teilmenge von exo)                                |
| markets    | exo_mkt_*  (Teilmenge von exo)                                |

`exo` und seine Teilgruppen nie zusammen in cfg.groups verwenden (doppelte Features).
| series     | One-Hot der Serie (standardmässig aus)                        |
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import common as C
from .common import qlike


EXO_GROUPS = {"volume": "exo_vol_", "dvol": "exo_dvol_", "markets": "exo_mkt_"}   # Teilgruppen von exo


@dataclass
class FeatureConfig:
    pool: list
    sids: list
    groups: list = field(default_factory=lambda: ["market", "disagree", "recent", "calendar", "exo"])
    # Alle Fenster in Tagen. Bei Stundendaten werden sie mit 24 multipliziert (C.steps_per_day).
    halflives: list = field(default_factory=lambda: [10, 60])     # Tage
    min_periods: int = 5                                           # Tage
    level: str = "relative"         # "relative" oder "absolute" (alt)
    level_window: int = 250                                        # Tage
    drop_redundant: bool = True
    calendar: list = field(default_factory=lambda: ["weekend"])   # + "hour", "us_open", "events" (exo.py)

    @classmethod
    def legacy(cls, pool, sids):
        # exakt die 14 Features aus archiv/01_kombination.ipynb
        return cls(pool, sids, groups=["market", "disagree", "recent"], halflives=[10],
                   level="absolute", drop_redundant=False)

    def to_dict(self):
        return asdict(self)


def add_recent(fc, cfg):
    """QLIKE und jüngste Fehler je Halbwertszeit. Für Horizont h um h Ursprünge verschoben:
    am Ursprung t ist das Ziel der Prognose von t-h gerade bekannt geworden."""
    fc = fc.sort_values(["sid", "h", "model", "origin"]).reset_index(drop=True)
    fc["qlike"] = qlike(fc["y"].to_numpy(), fc["mean"].to_numpy())
    k = C.steps_per_day(np.unique(fc["origin"]))          # Ursprünge pro Tag (Stundendaten: 24)
    for hl in cfg.halflives:                              # Spaltenname bleibt in Tagen (recent10, recent60)
        fc[f"recent{hl}"] = np.nan
        for h, g in fc.groupby("h"):
            fc.loc[g.index, f"recent{hl}"] = g.groupby(["sid", "model"])["qlike"].transform(
                lambda x: x.shift(h).ewm(halflife=hl * k, min_periods=cfg.min_periods * k).mean())
    return fc


def market_features(d, cfg):
    # Fenster in Tagen (5 = Woche, 22 = Monat), umgerechnet in Schritte. ret1 bleibt der letzte Schritt.
    k = C.steps_per_day(d.index)
    rv, ret = d["rv"], d["ret"]
    lrv, rv22 = np.log(rv), rv.rolling(22 * k).mean()
    f = pd.DataFrame({"logrv_dev": lrv - np.log(rv22),
                      "trend": np.log(rv.rolling(5 * k).mean()) - np.log(rv22)})
    if cfg.level == "absolute":
        f["level"] = np.log(rv22)
    else:
        f["level_rel"] = np.log(rv22) - lrv.rolling(cfg.level_window * k, min_periods=60 * k).mean()
    f["volvol"] = lrv.rolling(22 * k).std()
    f["ret1"] = ret / np.sqrt(rv22)
    f["ret5"] = ret.rolling(5 * k).sum() / np.sqrt(5 * k * rv22)
    return f


def daily_features(daily, cfg):
    """daily: {sid: DataFrame mit rv, ret (und exo_*)} -> Marktzustand, exo_* und scale je (sid, origin)."""
    parts = []
    for sid in cfg.sids:
        d = daily[sid]
        f = market_features(d, cfg)
        f["scale"] = d["rv"].rolling(22 * C.steps_per_day(d.index)).mean()
        for c in d.columns:
            if c.startswith("exo_"):
                f[c] = d[c]
        f.index = pd.MultiIndex.from_arrays([np.repeat(sid, len(f)), f.index], names=["sid", "origin"])
        parts.append(f)
    return pd.concat(parts)


def build_panel(fc, daily_feat, h, cfg, true_var=None):
    """Panel für Horizont h. fc braucht die Spalten aus add_recent. Zeilen mit unbekanntem Ziel (y = NaN)
    bleiben erhalten (App: heutige Prognose)."""
    pool = cfg.pool
    market = [c for c in daily_feat.columns if c != "scale" and not c.startswith("exo_")]
    exo = [c for c in daily_feat.columns if c.startswith("exo_")]
    sub = fc[fc["h"] == h]
    rec = [f"recent{hl}" for hl in cfg.halflives]
    wide = sub.set_index(["sid", "origin", "model"])[["y", "target_date", "mean"] + rec + C.QCOLS].unstack("model")
    mean = wide["mean"][pool].to_numpy()
    Q = np.stack([wide[c][pool].to_numpy() for c in C.QCOLS], axis=-1)
    ok = np.isfinite(mean).all(1) & np.isfinite(Q).all((1, 2))     # nur Zeilen, für die alle Experten prognostizieren
    wide, mean, Q = wide[ok], mean[ok], Q[ok]

    meta = pd.DataFrame({"y": wide["y"][pool[0]].to_numpy(),
                         "target_date": pd.to_datetime(wide["target_date"][pool[0]].to_numpy())}, index=wide.index)
    meta = meta.join(daily_feat).reset_index()
    if true_var is not None:
        meta = meta.merge(true_var, on=["sid", "target_date"], how="left")

    keep = pool[:-1] if cfg.drop_redundant else pool
    groups, X = {"market": market}, {c: meta[c].to_numpy() for c in market}
    logm = np.log(mean)
    groups["disagree"] = [f"disagree_{m}" for m in keep]
    for j, m in enumerate(keep):
        X[f"disagree_{m}"] = logm[:, j] - logm.mean(1)
    groups["recent"] = []
    for hl in cfg.halflives:
        lrec = np.log(wide[f"recent{hl}"][pool].to_numpy())
        for j, m in enumerate(keep):
            name = f"recent_{m}" if len(cfg.halflives) == 1 else f"recent{hl}_{m}"
            groups["recent"].append(name)
            X[name] = lrec[:, j] - lrec.mean(1)
    groups["calendar"] = []
    if "weekend" in cfg.calendar:
        groups["calendar"].append("weekend")
        X["weekend"] = (meta["target_date"].dt.day_of_week >= 5).astype(float).to_numpy()
    extra = [c for c in cfg.calendar if c != "weekend"]
    if extra:                       # am Zielzeitpunkt: Uhrzeit, Börse offen, FOMC/CPI (stehen im Voraus fest)
        from .exo import calendar_features
        cal = calendar_features(meta["target_date"], extra)
        groups["calendar"] += list(cal.columns)
        X.update({c: cal[c].to_numpy() for c in cal.columns})
    groups["exo"] = exo
    for g, prefix in EXO_GROUPS.items():
        groups[g] = [c for c in exo if c.startswith(prefix)]
    X.update({c: meta[c].to_numpy() for c in exo})
    groups["series"] = [f"series_{s}" for s in cfg.sids]
    X.update({f"series_{s}": (meta["sid"] == s).astype(float).to_numpy() for s in cfg.sids})
    groups = {g: groups[g] for g in cfg.groups if groups.get(g)}
    feats = [c for g in groups.values() for c in g]
    if len(feats) != len(set(feats)):
        raise ValueError(f"Features doppelt in den Gruppen {list(groups)}: 'exo' nicht zusammen mit "
                         f"{list(EXO_GROUPS)} verwenden")

    out = meta[["sid", "origin", "target_date", "y", "scale"] + [c for c in ["true_var", "regime"] if c in meta]].copy()
    out.insert(1, "h", h)
    cols = {f"mean|{m}": mean[:, j] for j, m in enumerate(pool)}
    cols.update({f"{qc}|{m}": Q[:, j, k] for j, m in enumerate(pool) for k, qc in enumerate(C.QCOLS)})
    cols.update({f"recent|{m}": wide[rec[0]][m].to_numpy() for m in pool})
    cols.update({f"x|{c}": X[c] for g in groups.values() for c in g})
    return pd.concat([out, pd.DataFrame(cols)], axis=1), groups


def build_panels(fc, daily, cfg, true_var=None):
    """Alles in einem Schritt: jüngste Fehler, Tagesfeatures, ein Panel pro Horizont."""
    fc = add_recent(fc[fc["model"].isin(cfg.pool)], cfg)
    dfeat = daily_features(daily, cfg)
    panels, groups = {}, {}
    for h in sorted(fc["h"].unique()):
        panels[int(h)], groups = build_panel(fc, dfeat, h, cfg, true_var)
    return panels, groups
