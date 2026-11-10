"""Features für das Gate und Panels (eine Zeile pro Serie und Ursprung, ein Panel pro Horizont).

Alle Features sind am Ursprung t bekannt. Sie werden hier nicht standardisiert: Mittelwert und Streuung
schätzt das Gate pro Fold nur auf den Trainingszeilen.

| Gruppe     | Features                                                      |
|------------|---------------------------------------------------------------|
| market     | logrv_dev, trend, level_rel (bzw. level), volvol, ret1, ret5  |
| disagree   | ln mu_m - Mittel der ln mu_j                                  |
| recent     | ln L_m - Mittel der ln L_j, L = EWMA-QLIKE, um h verschoben    |
| calendar   | weekend (Zieldatum Samstag/Sonntag)                           |
| exo        | alle Spalten exo_* der Tagesdaten                             |
| series     | One-Hot der Serie (standardmässig aus)                        |
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import common as C
from .common import qlike


@dataclass
class FeatureConfig:
    pool: list
    sids: list
    groups: list = field(default_factory=lambda: ["market", "disagree", "recent", "calendar", "exo"])
    halflives: list = field(default_factory=lambda: [10, 60])
    min_periods: int = 5
    level: str = "relative"         # "relative" oder "absolute" (alt)
    level_window: int = 250
    drop_redundant: bool = True

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
    for hl in cfg.halflives:
        fc[f"recent{hl}"] = np.nan
        for h, g in fc.groupby("h"):
            fc.loc[g.index, f"recent{hl}"] = g.groupby(["sid", "model"])["qlike"].transform(
                lambda x: x.shift(h).ewm(halflife=hl, min_periods=cfg.min_periods).mean())
    return fc


def market_features(d, cfg):
    rv, ret = d["rv"], d["ret"]
    lrv, rv22 = np.log(rv), rv.rolling(22).mean()
    f = pd.DataFrame({"logrv_dev": lrv - np.log(rv22), "trend": np.log(rv.rolling(5).mean()) - np.log(rv22)})
    if cfg.level == "absolute":
        f["level"] = np.log(rv22)
    else:
        f["level_rel"] = np.log(rv22) - lrv.rolling(cfg.level_window, min_periods=60).mean()
    f["volvol"] = lrv.rolling(22).std()
    f["ret1"] = ret / np.sqrt(rv22)
    f["ret5"] = ret.rolling(5).sum() / np.sqrt(5 * rv22)
    return f


def daily_features(daily, cfg):
    """daily: {sid: DataFrame mit rv, ret (und exo_*)} -> Marktzustand, exo_* und scale je (sid, origin)."""
    parts = []
    for sid in cfg.sids:
        d = daily[sid]
        f = market_features(d, cfg)
        f["scale"] = d["rv"].rolling(22).mean()
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
    groups["calendar"] = ["weekend"]
    X["weekend"] = (meta["target_date"].dt.day_of_week >= 5).astype(float).to_numpy()
    groups["exo"] = exo
    X.update({c: meta[c].to_numpy() for c in exo})
    groups["series"] = [f"series_{s}" for s in cfg.sids]
    X.update({f"series_{s}": (meta["sid"] == s).astype(float).to_numpy() for s in cfg.sids})
    groups = {g: groups[g] for g in cfg.groups if groups.get(g)}

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
