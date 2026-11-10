"""Kombination der Experten: Panel, Loss, Baselines, Gate und Walk-Forward-Folds.

Alle Methoden mischen Mittelwert und Quantile mit denselben Gewichten w >= 0, sum w = 1.
Schnittstelle wie scikit-learn: `fit(panel_train)` lernt, `weights(panel)` gibt Gewichte (n, M).

Loss: QLIKE(y, mu) + ALPHA * Pinball(y, q) / s_t (+ lambda * sum W² beim Gate),
s_t = RV-Niveau der letzten 22 Tage.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import common as C
from .common import qlike

ALPHA = 1.0                 # Gewicht des Pinball-Terms im Loss (04_training setzt es aus seiner Konfiguration)
DTYPE = torch.float64       # explizit statt torch.set_default_dtype, damit Foundation Models in der App unberührt bleiben


class Panel:
    """Ein Horizont: meta (n Zeilen), mean (n, M), Q (n, M, 21), X (n, k) roh, recent (n, M)."""

    def __init__(self, df, h, pool, groups):
        self.h, self.models, self.groups = h, list(pool), groups
        self.meta = df[[c for c in df.columns if "|" not in c]].reset_index(drop=True)
        self.mean = df[[f"mean|{m}" for m in pool]].to_numpy()
        self.Q = np.stack([df[[f"{qc}|{m}" for qc in C.QCOLS]].to_numpy() for m in pool], axis=1)
        self.recent = df[[f"recent|{m}" for m in pool]].to_numpy()
        feats = [c for g in groups.values() for c in g]
        self.X = df[[f"x|{c}" for c in feats]].set_axis(feats, axis=1).reset_index(drop=True)

    def __len__(self):
        return len(self.meta)

    def take(self, mask):
        P = object.__new__(Panel)
        P.h, P.models, P.groups = self.h, self.models, self.groups
        P.meta = self.meta[mask].reset_index(drop=True)
        P.mean, P.Q, P.recent = self.mean[mask], self.Q[mask], self.recent[mask]
        P.X = self.X[mask].reset_index(drop=True)
        return P

    def features(self, groups=None):
        groups = self.groups if groups is None else groups
        return [f for g in groups for f in self.groups.get(g, [])]

    @property
    def y(self):
        return self.meta["y"].to_numpy()

    @property
    def td(self):
        return pd.DatetimeIndex(self.meta["target_date"])


# =====================================================================================
# Loss
# =====================================================================================
def T(a):
    return torch.as_tensor(np.asarray(a, dtype=float), dtype=DTYPE)


def comb_loss_torch(w, mean, Q, y, s, alpha=None):
    alpha = ALPHA if alpha is None else alpha
    f = (w * mean).sum(1)
    Qc = torch.einsum("nm,nmq->nq", w, Q)
    r = y / f
    d = y[:, None] - Qc
    lv = torch.as_tensor(C.LV, dtype=DTYPE)
    pin = torch.maximum(lv * d, (lv - 1) * d).mean(1)
    return (r - torch.log(r) - 1).mean() + alpha * (pin / s).mean()


def comb_loss(w, P):
    with torch.no_grad():
        return float(comb_loss_torch(T(w), T(P.mean), T(P.Q), T(P.y), T(P.meta["scale"])))


def last_days(P, window_days):
    # Maske: nur die letzten window_days (nach Zieldatum), None = alle
    if window_days is None or len(P) == 0:
        return np.ones(len(P), bool)
    return np.asarray(P.td > P.td.max() - pd.Timedelta(days=window_days))


# =====================================================================================
# Baselines
# =====================================================================================
class Combiner:
    def fit(self, P):
        return self

    def weights(self, P):
        raise NotImplementedError


class Equal(Combiner):
    def weights(self, P):
        M = len(P.models)
        return np.full((len(P), M), 1 / M)


class BestSingle(Combiner):
    def fit(self, P):
        self.best_ = int(np.argmin(qlike(P.y[:, None], P.mean).mean(0)))
        return self

    def weights(self, P):
        return np.eye(len(P.models))[np.full(len(P), self.best_)]


class InverseLoss(Combiner):
    # Bates/Granger: w_m proportional zu 1 / mittlerer QLIKE im Training
    def __init__(self, window_days=None):
        self.window_days = window_days

    def fit(self, P):
        P = P.take(last_days(P, self.window_days))
        inv = 1 / qlike(P.y[:, None], P.mean).mean(0)
        self.w_ = inv / inv.sum()
        return self

    def weights(self, P):
        return np.tile(self.w_, (len(P), 1))


class RecentInverseLoss(Combiner):
    # w_m,t proportional zu 1 / jüngster EWMA-QLIKE (am Ursprung bekannt), kein Training
    def weights(self, P):
        w = 1 / P.recent
        w = w / w.sum(1, keepdims=True)
        w[~np.isfinite(w).all(1)] = 1 / len(P.models)       # Aufwärmphase: Gleichgewicht
        return w


# =====================================================================================
# Gate: w = softmax(b + f(x_t)), f linear (hidden=0) oder MLP mit tanh
# =====================================================================================
class GateNet(nn.Module):
    def __init__(self, n_feat, n_exp, hidden=0):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_exp, dtype=DTYPE))
        if n_feat == 0:
            self.net = None
        elif hidden == 0:
            self.net = nn.Linear(n_feat, n_exp, bias=False, dtype=DTYPE)
        else:
            self.net = nn.Sequential(nn.Linear(n_feat, hidden, dtype=DTYPE), nn.Tanh(),
                                     nn.Linear(hidden, n_exp, bias=False, dtype=DTYPE))
        if self.net is not None:
            nn.init.zeros_((self.net if hidden == 0 else self.net[-1]).weight)

    def forward(self, X):
        z = self.logits.expand(len(X), -1)
        if self.net is not None:
            z = z + self.net(X)
        return torch.softmax(z, dim=-1)

    def penalty(self, include_bias=False):
        if self.net is None:
            return torch.tensor(0.0, dtype=DTYPE)
        return sum((p ** 2).sum() for n, p in self.net.named_parameters() if include_bias or n.endswith("weight"))


class Gate(Combiner):
    """Start bei Gleichgewicht (b = 0, letzte Schicht = 0). Features mit Mittelwert und Streuung der
    Trainingszeilen standardisiert, fehlende Werte -> 0. L2-Strafe lambda nur auf die Gewichtsmatrizen.
    groups=() heisst keine Features, also statische Gewichte softmax(b). Full-batch L-BFGS."""

    def __init__(self, hidden=0, lam=0.0, groups=None, seeds=1, window_days=None, penalize_bias=False):
        self.hidden, self.lam, self.groups, self.seeds = hidden, lam, groups, seeds
        self.window_days, self.penalize_bias = window_days, penalize_bias

    def _X(self, P):
        X = ((P.X[self.feats_] - self.mu_) / self.sd_).fillna(0.0)
        return T(X.to_numpy()).reshape(len(P), len(self.feats_))

    def fit(self, P):
        P = P.take(last_days(P, self.window_days))
        self.models_ = list(P.models)
        self.feats_ = P.features(self.groups)
        X = P.X[self.feats_]
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1).fillna(1)     # nur Trainingszeilen
        args = (self._X(P), T(P.mean), T(P.Q), T(P.y), T(P.meta["scale"]))
        self.nets_ = [self._fit_one(seed, *args) for seed in range(self.seeds if self.hidden else 1)]
        return self

    def _fit_one(self, seed, X, mean, Q, y, s):
        torch.manual_seed(seed)
        net = GateNet(X.shape[1], mean.shape[1], self.hidden)
        opt = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = comb_loss_torch(net(X), mean, Q, y, s) + self.lam * net.penalty(self.penalize_bias)
            loss.backward()
            return loss

        opt.step(closure)
        return net

    def weights(self, P):
        X = self._X(P)
        with torch.no_grad():
            return np.mean([net(X).numpy() for net in self.nets_], axis=0)

    def coef(self):
        # W des linearen Gates auf standardisierten Features (Features x Experten)
        return pd.DataFrame(self.nets_[0].net.weight.detach().numpy().T, index=self.feats_, columns=self.models_)


# =====================================================================================
# Methoden, Folds, Fitten
# =====================================================================================
@dataclass
class Method:
    name: str
    make: object                               # make(**params) -> Combiner
    grid: dict = field(default_factory=dict)   # Hyperparameter -> Kandidaten

    def candidates(self):
        return [dict(zip(self.grid, v)) for v in itertools.product(*self.grid.values())] or [{}]


@dataclass
class Fold:
    label: str
    tune_fit: np.ndarray
    tune_eval: np.ndarray
    train: np.ndarray
    test: np.ndarray


def make_fold(P, start, test, tune_days, gap_days=0, train_start=None, window_days=None, exclude=None):
    """Ein Walk-Forward-Fold, dessen erstes ausgewertetes Ziel `start` ist.

    Training: alle Ziele bis start - h - gap (beim ersten Test-Ursprung bekannt). `exclude` (Maske) nimmt
    zusätzlich Zeilen aus dem Training, z.B. eine gesperrte Testperiode. Zeilen ohne bekanntes Ziel werden
    nie trainiert. lambda wird auf den letzten tune_days des Trainings gewählt (tune_eval), gefittet wird
    dafür auf tune_fit, das h Tage vor tune_eval endet."""
    td = P.td
    lag = pd.Timedelta(days=P.h + gap_days)
    cutoff = pd.Timestamp(start) - lag
    train = np.asarray(td <= cutoff) & np.isfinite(P.y)
    if train_start:
        train &= np.asarray(td >= pd.Timestamp(train_start))
    if window_days:
        train &= np.asarray(td > cutoff - pd.Timedelta(days=window_days))
    if exclude is not None:
        train &= ~np.asarray(exclude)
    tune_start = cutoff - pd.Timedelta(days=tune_days)              # tune_eval: (tune_start, cutoff]
    tune_eval = train & np.asarray(td > tune_start)
    tune_fit = train & np.asarray(td <= tune_start + pd.Timedelta(days=1) - lag)
    return Fold(pd.Timestamp(start).strftime("%Y-%m-%d"), tune_fit, tune_eval, train, np.asarray(test))


def walkforward_folds(P, eval_start, eval_end, freq, tune_days, gap_days=0, train_start=None, window_days=None):
    td = P.td
    start = pd.Timestamp(eval_start) if eval_start else td.min() + 0.6 * (td.max() - td.min())
    start = start.normalize()
    end = pd.Timestamp(eval_end) if eval_end else td.max()
    bounds = [start] + [d for d in pd.date_range(start, end, freq=freq) if d > start] + [end + pd.Timedelta(days=1)]
    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        test = np.asarray((td >= a) & (td < b))
        f = make_fold(P, a, test, tune_days, gap_days, train_start, window_days)
        if test.any() and f.train.any():
            folds.append(f)
    return folds


def fixed_folds(P, train_end, val_end):
    td = P.td
    tr = np.asarray(td <= pd.Timestamp(train_end))
    va = np.asarray((td > pd.Timestamp(train_end)) & (td <= pd.Timestamp(val_end)))
    return [Fold("val", tr, va, tr, va)]


def fit_predict(P, method, fold):
    """lambda (bzw. allgemein die Hyperparameter) auf tune_fit/tune_eval wählen, dann auf train fitten."""
    cands, tuning, cache = method.candidates(), [], {}
    best = cands[0]
    if len(cands) > 1:
        Pf, Pe = P.take(fold.tune_fit), P.take(fold.tune_eval)
        if len(Pf) == 0 or len(Pe) == 0:
            raise ValueError(f"Fold {fold.label}: zu wenig Daten zum Tunen (TUNE_DAYS / Trainingsfenster prüfen)")
        same = np.array_equal(fold.tune_fit, fold.train)     # fixed: tune_fit = train -> nicht doppelt fitten
        best_loss = np.inf
        for params in cands:
            c = method.make(**params).fit(Pf)
            loss = comb_loss(c.weights(Pe), Pe)
            tuning.append({"h": P.h, "fold": fold.label, "Methode": method.name, **params, "Loss tune": loss})
            if same:
                cache[tuple(params.items())] = c
            if loss < best_loss:
                best, best_loss = params, loss
    c = cache.get(tuple(best.items())) or method.make(**best).fit(P.take(fold.train))
    return c, best, tuning
