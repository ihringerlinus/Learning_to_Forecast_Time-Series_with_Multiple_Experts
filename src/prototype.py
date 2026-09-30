"""
Erster Prototyp: Chronos-2 und Toto 2.0 auf absoluten Log-Returns einer Aktie.

Ablauf:
  1. Kursdaten mit yfinance laden
  2. Absolute Log-Returns berechnen (Proxy fuer Volatilitaet)
  3. Die letzten HORIZON Tage als Testfenster abschneiden
  4. Beide Modelle prognostizieren das Testfenster (Zero-Shot)
  5. MAE ausgeben und Prognosen plotten
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yfinance as yf
from chronos import Chronos2Pipeline
from toto2 import Toto2Model

# ---------------- Einstellungen ----------------
TICKER = "^GSPC"          # S&P 500; z.B. "^SSMI" fuer den SMI
START = "2015-01-01"
CONTEXT_LENGTH = 512      # wie viele vergangene Tage die Modelle sehen
HORIZON = 20              # wie viele Tage vorhergesagt werden
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 1. + 2. Daten ----------------
def load_abs_returns(ticker: str, start: str) -> pd.Series:
    prices = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    prices = prices.squeeze()  # DataFrame mit einer Spalte -> Series
    log_ret = np.log(prices).diff().dropna()
    return log_ret.abs().rename("target")


series = load_abs_returns(TICKER, START)
context = series.iloc[-(CONTEXT_LENGTH + HORIZON):-HORIZON]
truth = series.iloc[-HORIZON:]
print(f"{TICKER}: {len(series)} Tage geladen, Kontext {len(context)}, Test {len(truth)}")


# ---------------- 4a. Chronos-2 ----------------
# Chronos-2 is a quantile model (not sample-based); default levels are 0.1 … 0.9
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def forecast_chronos(context: pd.Series, horizon: int) -> dict:
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    context_tensor = torch.tensor(context.values, dtype=torch.float32)
    # predict_quantiles returns (quantiles_list, mean_list)
    # each element of quantiles_list has shape (n_variates, horizon, n_quantiles)
    quantiles, mean = pipeline.predict_quantiles(
        context_tensor.view(1, 1, -1),   # (n_series=1, n_variates=1, history)
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
    )
    # shape: (n_quantiles, horizon) — treat each quantile line as one "sample path"
    q = quantiles[0][0].float().cpu().numpy().T   # (n_quantiles, horizon)
    q = np.clip(q, 0, None)
    return {
        "samples": q,
        "median": q[QUANTILE_LEVELS.index(0.5)],
    }


# ---------------- 4b. Toto 2.0 ----------------
def forecast_toto(context: pd.Series, horizon: int) -> dict:
    model = Toto2Model.from_pretrained("Datadog/Toto-2.0-313m").to(DEVICE).eval()
    # Form: (batch, n_variates, time_steps)
    target = torch.tensor(context.values, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
    with torch.no_grad():
        q = model.forecast(
            {
                "target": target,
                "target_mask": torch.ones_like(target, dtype=torch.bool),
                "series_ids": torch.zeros(1, 1, dtype=torch.long, device=DEVICE),
            },
            horizon=horizon,
            decode_block_size=768,
            has_missing_values=False,
        )
    # q hat Form (9, batch, n_variates, horizon), Quantile 0.1 ... 0.9
    q = q[:, 0, 0, :].float().cpu().numpy()   # (9, horizon)
    q = np.clip(q, 0, None)
    # Toto exposes 9 quantile lines; treat them as the fan of samples
    return {
        "samples": q,          # (9, horizon)
        "median": q[4],        # q50
    }


forecasts = {
    "Chronos-2": forecast_chronos(context, HORIZON),
    "Toto 2.0 (313m)": forecast_toto(context, HORIZON),
}

# Clipping already applied inside each forecast function (samples >= 0)


# ---------------- 5. Auswertung und Plot ----------------
naive = np.full(HORIZON, context.iloc[-20:].mean())   # einfache Baseline
print(f"\nMAE Baseline (Mittel letzte 20 Tage): {np.mean(np.abs(truth.values - naive)):.5f}")
for name, fc in forecasts.items():
    mae = np.mean(np.abs(truth.values - fc["median"]))
    print(f"MAE {name:<16}: {mae:.5f}")

fig, ax = plt.subplots(figsize=(12, 4))
hist = context.iloc[-100:]
ax.plot(hist.index, hist.values, color="grey", label="Historie")
ax.plot(truth.index, truth.values, color="black", linewidth=1.5, label="Tatsaechlich")
for (name, fc), color in zip(forecasts.items(), ["tab:blue", "tab:orange"]):
    # draw every sample path as a thin translucent line
    for sample in fc["samples"]:
        ax.plot(truth.index, sample, color=color, alpha=0.15, linewidth=0.7)
    # median on top as a bold reference
    ax.plot(truth.index, fc["median"], color=color, linewidth=1.8, label=f"{name} (Median)")
ax.set_title(f"{TICKER}: absolute Log-Returns, Prognose {HORIZON} Tage")
ax.legend()
plt.tight_layout()
plt.savefig("prognose.png", dpi=120)
print("\nPlot gespeichert: prognose.png")