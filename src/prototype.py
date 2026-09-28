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
def forecast_chronos(context: pd.Series, horizon: int) -> pd.DataFrame:
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    # Chronos-2 braucht Zeitstempel mit regelmaessiger Frequenz. Handelstage haben
    # Luecken (Wochenenden, Feiertage), darum nummerieren wir sie als Pseudo-Tage durch.
    context_df = pd.DataFrame({
        "id": "serie_1",
        "timestamp": pd.date_range("2000-01-01", periods=len(context), freq="D"),
        "target": context.values,
    })
    pred = pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    return pd.DataFrame({
        "q10": pred["0.1"].values,
        "median": pred["0.5"].values,
        "q90": pred["0.9"].values,
    })


# ---------------- 4b. Toto 2.0 ----------------
def forecast_toto(context: pd.Series, horizon: int) -> pd.DataFrame:
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
    q = q[:, 0, 0, :].float().cpu().numpy()
    return pd.DataFrame({"q10": q[0], "median": q[4], "q90": q[8]})


forecasts = {
    "Chronos-2": forecast_chronos(context, HORIZON),
    "Toto 2.0 (313m)": forecast_toto(context, HORIZON),
}

# Absolute Returns sind nie negativ -> negative Prognosen auf 0 setzen
for fc in forecasts.values():
    fc[:] = fc.clip(lower=0)


# ---------------- 5. Auswertung und Plot ----------------
naive = np.full(HORIZON, context.iloc[-20:].mean())   # einfache Baseline
print(f"\nMAE Baseline (Mittel letzte 20 Tage): {np.mean(np.abs(truth.values - naive)):.5f}")
for name, fc in forecasts.items():
    mae = np.mean(np.abs(truth.values - fc["median"].values))
    print(f"MAE {name:<16}: {mae:.5f}")

fig, ax = plt.subplots(figsize=(12, 4))
hist = context.iloc[-100:]
ax.plot(hist.index, hist.values, color="grey", label="Historie")
ax.plot(truth.index, truth.values, color="black", label="Tatsaechlich")
for (name, fc), color in zip(forecasts.items(), ["tab:blue", "tab:orange"]):
    ax.plot(truth.index, fc["median"], color=color, label=f"{name} (Median)")
    ax.fill_between(truth.index, fc["q10"], fc["q90"], color=color, alpha=0.2)
ax.set_title(f"{TICKER}: absolute Log-Returns, Prognose {HORIZON} Tage")
ax.legend()
plt.tight_layout()
plt.savefig("prognose.png", dpi=120)
print("\nPlot gespeichert: prognose.png")