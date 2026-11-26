# Learning to Forecast Time Series with Multiple Experts

Bachelorprojekt (HSG). Mehrere Prognosemodelle ("Experten") sagen eine Zeitreihe voraus: die stündliche Volatilität von Kryptowährungen, Gold und EUR/USD, die stündliche Stromlast europäischer Länder und (in Arbeit) die Windstromerzeugung. Ein kleines neuronales Netz (das **Gate**) lernt, wie stark es jedem Experten je nach Situation vertrauen soll, und kombiniert ihre Prognosen.

**Forschungsfrage:** Ist eine gelernte, situationsabhängige Kombination von Experten (Foundation Models und klassische Modelle) besser als das beste Einzelmodell oder feste Gewichte? Und wovon hängt das ab?

**Kernergebnis (Stand Nov. 2026, Abschnitt 16):** Ob Kombinieren etwas bringt, hängt davon ab, wie ähnlich die Fehler der Foundation Models sind. Bei der Volatilität korrelieren ihre Fehler zu 0.93–0.98: Keine gelernte Kombination schlägt das beste Einzelmodell oder gleiche Gewichte signifikant (Forecast Combination Puzzle). Bei der Stromlast korrelieren sie nur zu 0.55–0.85: Gelernte Gewichte sind 2–12 % besser als das beste Einzelmodell, signifikant. Über alle Serien ist die Rangkorrelation zwischen Fehlerkorrelation und Kombinationsgewinn −0.76.

---

## Inhalt

1. [Überblick](#1-überblick)
2. [Projektstruktur](#2-projektstruktur)
3. [Installation und Ausführung](#3-installation-und-ausführung)
4. [Daten und Zielgrösse: realisierte Varianz (RV)](#4-daten-und-zielgrösse-realisierte-varianz-rv)
5. [Rolling-Origin-Prognosen](#5-rolling-origin-prognosen)
6. [Die Experten](#6-die-experten)
7. [Splits: train, val, test](#7-splits-train-val-test)
8. [Kombination: das Gate](#8-kombination-das-gate)
9. [Training des Gates](#9-training-des-gates)
10. [Evaluation und Metriken](#10-evaluation-und-metriken)
11. [Mehrere Domänen: Notebooks 06–09](#11-mehrere-domänen-notebooks-0609)
12. [Resource Check](#12-resource-check-00_resource_checkipynb)
13. [TimesFM-2.5 und Chronos-Check (Tagesdaten, Archiv)](#13-timesfm-25-und-chronos-check-tagesdaten-archiv)
14. [Live-Prognose-App](#14-live-prognose-app-app)
15. [Konfigurationsparameter](#15-konfigurationsparameter)
16. [Stand, Einschränkungen und nächste Schritte](#16-stand-einschränkungen-und-nächste-schritte)
17. [Glossar](#17-glossar)

---

## 1. Überblick

Die **Hauptpipeline** besteht aus fünf Notebooks, eines pro Data-Science-Schritt. Sie tauschen ihre Ergebnisse über Dateien aus:

```
01_daten        Binance oder synthetische Daten ──► stündliche RV pro Serie      data/<DATASET>.json
02_experten     RW, EWMA, HAR, GARCH, Chronos-2, TimesFM-2.5, t0, Timestral       results/forecasts/<DATASET>/<SID>__<MODELL>.parquet
                                                 (Mittelwert + 21 Quantile pro Ursprung und Horizont)
03_features     Prognosen + Stundendaten ──► Features, ein Panel pro Horizont     results/panels/<DATASET>/
04_training     Split (Walk-Forward) ──► λ wählen ──► Baselines und Gates fitten   results/runs/<RUN>/
05_evaluation   Out-of-Sample-Vergleich: Experten vs. Baselines vs. Gate, DM-Tests, Oracle, Gewichte
```

Dazu kommen vier **eigenständige Notebooks** für den Domänenvergleich (Abschnitt 11). Sie laden ihre Daten selbst, rechnen die Experten (auf Kaggle/Colab mit GPU) und trainieren die Gates in einem Notebook:

```
06_new_models_and_TS   Volatilität: PAXG (Gold), EUR/USDT, XRP, BNB; neu TiRex und FlowState
07_electricity         Stromlast DE/FR/IT/ES (Energy-Charts); klassisch SNaive und SeasReg
08_correlation_vs_win  Analyse über alle Datensätze: Fehlerkorrelation vs. Kombinationsgewinn
09_wind                Windstrom onshore DE/FR/ES/PL (noch nicht gerechnet)
```

**Datenquelle austauschbar:** `01_daten` schreibt echte (`binance_1h`) oder synthetische Daten (`syn_<Szenario>_s<Seed>`) in dasselbe Format. Die Notebooks 02–05 brauchen nur `DATASET = "..."` in der ersten Zelle. Der frühere Tagesdatensatz `binance` (5-Min-Kerzen, h = 1/5 Tage) liegt noch an den alten Orten (`data/<COIN>_daily.parquet`, `results/<COIN>__<MODELL>.parquet`, `results/runs/binance_walkforward/`).

**Aktueller Pool in 03–05 (`binance_1h`):** Chronos-2 (voll) und TimesFM-2.5. Der Random Walk dient als Baseline. HAR und GARCH werden gerechnet, sind aber nicht im Pool.

- **Stufe 1 (02)** ist teuer (Modelle laufen über Tausende Ursprünge) und wird einmal gerechnet. Die Ergebnisse werden gecacht.
- **Stufe 2 (03–05)** ist billig und arbeitet nur auf den gespeicherten Prognosen.
- **Quantile kappen:** Chronos-2 und TiRex liefern auf Volatilität vereinzelt absurde Quantile (q0.99 bis e⁴² × Median). `pipeline.common.cap_quantiles(fc, k=8)` begrenzt die Quantile beim Laden auf Median · e^±8 und rechnet in den betroffenen Zeilen den Mittelwert neu. Alle Notebooks, die Prognosen laden (03, 05, 06, 08, 09), wenden es an. Details in Abschnitt 6.

**Gemeinsamer Code:** Experten, Features und Kombination (Panel, Loss, Baselines, Gate, Folds) stehen in `src/pipeline/`. Die Notebooks 02–04 und die App importieren denselben Code. Was in den Notebooks gilt, gilt deshalb exakt auch in der App (Abschnitt 14). Die Notebooks 06–09 nutzen dieselben Module, ersetzen aber Teile davon im Notebook (eigene Experten-Klassen, eigener Loss für Strom und Wind).

Die früheren Notebooks liegen unverändert in `src/archiv/` (siehe Abschnitt 2). Mit `LEGACY = True` in 03 und 04 sowie `SPLIT = "fixed"` reproduzieren die neuen Notebooks deren Ergebnisse exakt.

---

## 2. Projektstruktur

```
.
├── README.md
├── CLAUDE.md                     Kontext für Claude Code (Stand, Entscheidungen, Stolpersteine)
├── requirements.txt
├── prognose.png                  Plot aus einem frühen Prototyp (Chronos-2 + Toto, S&P 500)
├── app/
│   ├── app.py                    Live-Prognose-App (Streamlit), siehe Abschnitt 14
│   └── .streamlit/config.toml    Farben und Darstellung der App
├── reports/
│   └── zwischenstand.html        Zwischenbericht mit Grafiken und Ergebnissen (Stand Tagesdaten)
└── src/
    ├── 00_resource_check.ipynb   laufen Chronos-2, TimesFM und Moirai lokal? RAM, Latenz
    ├── 01_daten.ipynb            Daten laden (Binance oder synthetisch), RV, exogene Features, Format prüfen
    ├── 02_experten.ipynb         Stufe 1: Rolling-Origin-Prognosen aller Experten (gecacht)
    ├── 03_features.ipynb         Features für das Gate, ein Panel pro Horizont
    ├── 04_training.ipynb         Split, Hyperparameterwahl, Baselines, Gates und Ablationen fitten
    ├── 05_evaluation.ipynb       Vergleich mit Experten und Baselines, DM-Tests, Oracle
    ├── 06_new_models_and_TS.ipynb  Volatilität neuer Serien, TiRex und FlowState (Abschnitt 11)
    ├── 07_electricity.ipynb      Stromlast als zweite Domäne (Abschnitt 11)
    ├── 08_correlation_vs_win.ipynb  Fehlerkorrelation vs. Kombinationsgewinn über alle Datensätze
    ├── 09_wind.ipynb             Windstrom als dritte Domäne (vorbereitet, noch nicht gerechnet)
    ├── pipeline/                 gemeinsamer Code von Notebooks und App
    │   ├── common.py             Quantile, Horizonte, Schätzfenster, cap_quantiles (gilt für Notebooks und App)
    │   ├── experts.py            alle Experten (02)
    │   ├── exo.py                exogene Features (Volumen, DVOL, VIX/S&P) und Kalender (Stunde, US-Börse, FOMC/CPI)
    │   ├── features.py           Features und Panels (03)
    │   ├── combine.py            Panel, Loss, Baselines, Gate, Walk-Forward-Folds (04)
    │   └── live.py               Live-Betrieb der App: Prognosen nachführen, Monatsmodelle
    ├── archiv/                   frühere Notebooks (Zwischenbericht), durch 01–05 ersetzt
    │   ├── volatilitaet_chronos2_har_garch.ipynb
    │   ├── 02_timesfm.ipynb      inkl. Chronos-Check (Abschnitt 13)
    │   ├── 01_kombination.ipynb
    │   └── 03_t0_timestral.ipynb erste Prüfung von t0 und Timestral (heute in 02_experten)
    ├── data/                     (nicht in Git, siehe .gitignore)
    │   ├── raw/<COIN>/<COIN>-1m-YYYY-MM.zip   Binance-Monatsdateien (1-Min-Kerzen, auch von 06 genutzt)
    │   ├── raw/energy_charts/     API-Antworten von Energy-Charts (07 und 09 teilen den Cache)
    │   ├── <DATASET>/<SID>_daily.parquet   Stundendaten pro Serie (Name "daily" historisch), Spalte rv = Zielgrösse
    │   │                          binance_1h, binance_1h_neu (06), strom_1h (07), wind_1h (09), syn_*
    │   ├── <DATASET>.json        Beschreibung des Datensatzes (Serien, Pfade, Testsperre)
    │   ├── <COIN>_daily.parquet  alter Tagesdatensatz binance (5-Min-Kerzen)
    │   └── live/binance_1h/      alles, was die App schreibt (Abschnitt 14)
    └── results/
        ├── forecasts/<DATASET>/<SID>__<MODELL>.parquet   Prognosen der Experten (nicht in Git)
        ├── forecasts/06_res.parquet, 06_wts.parquet      Kombinationsergebnisse und Gewichte aus 06
        ├── forecasts/07_res.parquet, 07_wts.parquet      dasselbe aus 07 (09 schreibt 09_res/09_wts)
        ├── panels/<DATASET>/          Panels aus 03 (h1.parquet, h24.parquet, panel.json; nicht in Git)
        ├── runs/<RUN>/                Ergebnisse aus 04 (Prognosen, Gewichte, λ, Koeffizienten, Konfiguration)
        │                              binance_1h_walkforward (aktuell), binance_walkforward (Tagesdaten),
        │                              syn_regime_signal_s0_walkforward
        ├── korrelation_vs_gewinn.png, oos_vs_obergrenze.png   Grafiken aus 08
        ├── <COIN>__<MODELL>.parquet   Prognosen des alten Tagesdatensatzes binance
        ├── runtimes.csv, runtimes_timesfm.csv   Laufzeiten der Experten (Tagesdaten)
        ├── chronos_check.json         welche Chronos-Variante steckt in Chronos-2.parquet?
        └── resource_check_*.csv/json  Ergebnisse des Resource Checks
```

`src/results/forecasts/` und `src/results/panels/` sind in `.gitignore`: Die Expertenprognosen der GPU-Läufe (06, 07, 09) kommen als ZIP von Kaggle/Colab und werden lokal nach `src/` entpackt. Die Notebooks im Archiv erwarten ihre Pfade relativ zu `src/`. Zum erneuten Ausführen zurück nach `src/` verschieben.

---

## 3. Installation und Ausführung

### Umgebung

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

> **Achtung:** `requirements.txt` ist nicht vollständig. Die Notebooks brauchen zusätzlich:
> ```bash
> pip install arch scipy requests tqdm psutil jupyter
> # für TimesFM-2.5 in 02_experten.ipynb (fehlt es, wird der Experte übersprungen):
> pip install timesfm
> # für die Live-Prognose-App:
> pip install streamlit
> # für t0 und Timestral (02_experten.ipynb und App, optional):
> pip install "tfc-t0[torch]>=0.6.0"
> pip install git+https://github.com/mistralai/timestral.git
> # für TiRex und FlowState (06, 07, 09):
> pip install tirex-ts granite-tsfm
> # nur für den Resource Check (Moirai):
> pip install uni2ts gluonts
> ```

**Jupyter-Kernel:** Der Kernel muss die `.venv` verwenden. Startet er die System-Python (Fehler `pyarrow` fehlt), Jupyter aus der aktivierten `.venv` starten oder beim Ausführen per Kommandozeile `.venv\Scripts` vorne in den `PATH` setzen.

### Reihenfolge

Die Notebooks verwenden relative Pfade (`data/`, `results/`) und müssen deshalb **aus dem Ordner `src/`** gestartet werden.

| Schritt | Datei | was passiert | Dauer |
|---|---|---|---|
| (optional) | `00_resource_check.ipynb` | prüft, ob die Foundation Models lokal laufen | Minuten |
| 1 | `01_daten.ipynb` | Binance-Daten herunterladen, stündliche RV und exogene Features berechnen, oder synthetische Reihen simulieren | Sekunden (mit Cache) |
| 2 | `02_experten.ipynb` | Prognosen aller Experten in `MODELS`, vorhandene werden nur geladen | Minuten bis Stunden (Foundation Models, besser auf GPU) |
| 3 | `03_features.ipynb` | Features und Panels bauen | Sekunden |
| 4 | `04_training.ipynb` | Walk-Forward: pro Monat λ wählen, Baselines, Gates und Ablationen fitten, out-of-sample prognostizieren | ca. 30 Min. (CPU) |
| 5 | `05_evaluation.ipynb` | Tabellen, Diebold-Mariano gegen die Baselines, Oracle, Gewichte, Koeffizienten | Sekunden |
| 6–9 | `06`, `07`, `09` | je eigenständig: Daten, Experten, Gates (Abschnitt 11). Expertenprognosen auf Kaggle/Colab mit GPU rechnen (ca. 4–5 h pro Datensatz), lokal läuft nur der Cache | lokal ca. 25 Min. (Gates) |
| 8 | `08_correlation_vs_win.ipynb` | Analyse über alle Datensätze, braucht die Ergebnisse von 06 und 07 | ca. 1 Min. |
| (optional) | `app/app.py` | Live-Prognose mit den neuesten Daten (Abschnitt 14) | ca. 1 Min. pro Klick |

**Datensatz wechseln:** in `01_daten` `SOURCE = "synthetic"` (und `SCENARIO`) setzen, danach in 02–05 `DATASET` auf den ausgegebenen Namen (z.B. `"syn_regime_signal_s0"`). In 05 zusätzlich `RUN` anpassen, falls in 04 ein anderer Name oder Split gewählt wurde.

**Caching:** Existiert eine Prognosedatei schon, wird sie nur geladen. Um einen Experten neu zu rechnen, die entsprechende Parquet-Datei löschen. Ändern sich die Daten eines Datensatzes (z.B. andere Szenario-Parameter), bricht `01_daten` ab, statt mit veralteten Prognosen weiterzurechnen.

Der Ordner `src/data/` ist nicht in Git. Auf einem neuen Rechner lädt `01_daten` die Daten automatisch von `data.binance.vision` (kostenlos, ohne API-Key).

---

## 4. Daten und Zielgrösse: realisierte Varianz (RV)

### Daten

| | aktuell: `binance_1h` | früher: `binance` |
|---|---|---|
| Quelle | Binance Spot, 1-Minuten-Kerzen (`data.binance.vision`) | Binance Spot, 5-Minuten-Kerzen |
| Coins | BTCUSDT, ETHUSDT, SOLUSDT | BTCUSDT, ETHUSDT, SOLUSDT |
| Zeitraum | 01/2023 – 08/2026 | 01/2020 (SOL 08/2020) – 08/2026 |
| Zielgrösse | RV pro Stunde (60 Bars) | RV pro Tag (288 Bars) |
| Qualitätsfilter | Stunden mit weniger als 54 Bars werden verworfen | Tage mit weniger als 270 Bars |
| Exogene Spalten | `exo_vol_*` (Volumen, Trades, Taker-Anteil), `exo_dvol_*` (Deribit DVOL, BTC/ETH), `exo_mkt_*` (VIX, S&P-500-Futures), siehe `pipeline/exo.py` | – |

Krypto handelt 24/7, es gibt keine Overnight-Lücke. Weitere Datensätze (Abschnitt 11): `binance_1h_neu` (PAXG, EUR, XRP, BNB, gleiche Berechnung), `strom_1h` (Stromlast) und `wind_1h` (Windstrom). Dort steht in der Spalte `rv` die jeweilige Zielgrösse (z.B. Last in MW), damit der Pipeline-Code unverändert bleibt.

### Was ist die RV?

Die **realisierte Varianz** fasst die Schwankung eines Zeitintervalls in **einer Zahl** zusammen. Für jede Stunde (früher: jeden UTC-Tag) *d*:

$$RV_d = \sum_{i} r_{d,i}^2, \qquad r_{d,i} = \ln P_{d,i} - \ln P_{d,i-1}$$

Dabei ist $P_{d,i}$ der Schlusskurs der *i*-ten Kerze im Intervall. Zusätzlich wird die **Rendite** $ret_d = \sum_i r_{d,i}$ gespeichert.

Code (`01_daten.ipynb`, ebenso `hourly_rv` in 06):

```python
r = np.log(close).diff().dropna()
hour = r.index.floor("1h")
rv  = (r**2).groupby(hour).sum()
ret = r.groupby(hour).sum()
```

**Warum RV und nicht einfach die Rendite?** Eine Stunde, die +1 % schliesst, kann ruhig gewesen sein oder zwischendurch −5 % und +5 % gesehen haben. Die Rendite ist in beiden Fällen gleich, die RV nicht. Die RV misst also, wie stark sich der Preis **während** des Intervalls bewegt hat. Die "wahre" Volatilität ist nicht beobachtbar, und die RV ist eine viel genauere Schätzung davon als die quadrierte Rendite.

**Warum 1 bzw. 5 Minuten?** Bei sehr kurzen Intervallen springt der Preis zwischen Geld- und Briefkurs (Mikrostruktur-Rauschen), und die RV wird künstlich zu gross. Für Tages-RV ist 5 Minuten der Standardkompromiss. Für Stunden-RV braucht es feinere Kerzen, sonst bestünde jede Stunde nur aus 12 Renditen.

### Grössenordnung (BTC, Tages-RV, Beispiel)

| Grösse | typischer Wert |
|---|---|
| 5-Minuten-Rendite | ca. ±0.15 % |
| Tagesvolatilität (Std.-Abw.) | ca. 2.5 % |
| **RV (Tagesvarianz)** | ca. 0.000625 |
| Annualisierte Volatilität $\sqrt{365 \cdot RV}$ | ca. 48 % |

Die RV ist stark **rechtsschief**: Die meisten Stunden und Tage sind ruhig, wenige haben extreme Spitzen. Deshalb rechnen die Modelle meist mit **log-RV**. Stunden-RV hat zusätzlich einen ausgeprägten **Tagesgang** (US-Börsenöffnung, Wochenende).

`data/<DATASET>/<SID>_daily.parquet` enthält die Spalten `rv`, `ret`, `n_bars` (und `exo_*`) mit dem Zeitstempel als Index. Der Dateiname "daily" ist historisch, die Zeilen sind Stunden.

### Gemeinsames Datenformat

Jede Datenquelle liefert pro Serie ein DataFrame mit Datum als Index und den Spalten `rv` und `ret` (Pflicht), optional `true_var` (wahre Varianz, nur synthetisch) und `exo_*` (externe Kovariaten, werden automatisch Gate-Features). `save_dataset` in `01_daten` prüft das Format und schreibt `data/<DATASET>.json`. Eigene Daten (Aktien, FX, Krypto mit DVOL, …) brauchen nur ein Dict `{sid: DataFrame}` und einen Aufruf von `save_dataset`.

### Synthetische Daten

Ein kontrolliertes Experiment, um die **Methodik** unabhängig von der Domäne zu testen. Keine Kontamination der Foundation Models, beliebig viele Daten, und die wahre Varianz ist bekannt.

- **Regime** $s_t$: Markov-Kette, Verweilwahrscheinlichkeit 0.97 (≈ 33 Tage im Mittel)
- **log-Varianz:** $l_t = \mu_k + \phi_k (l_{t-1} - \mu_k) + \eta_k \varepsilon_t$ mit $k = s_t$
- **Intraday:** 288 Renditen pro Tag mit t₅-Innovationen. Die RV wird wie bei echten Daten daraus berechnet und hat deshalb dasselbe Messrauschen.

| Regime | $\phi$ | $\eta$ | Verhalten | wer gewinnt (h = 5) |
|---|---|---|---|---|
| 0 „mean-reverting“ | 0.5 | 0.35 | Schocks klingen in wenigen Tagen ab | HAR |
| 1 „persistent“ | 0.999 | 0.25 | fast ein Random Walk in der log-Varianz | RW |

Beide Regimes haben dasselbe Niveau ($\mu = -7.4$, Tagesvola ≈ 2.5 % wie BTC), damit `level` das Regime nicht verrät. Pool auf synthetischen Daten: HAR, GARCH, RW.

| Szenario | Signal | Erwartung |
|---|---|---|
| `null` | ein Regime, kein Signal | Gate ≈ Statisch (Overfitting-Test) |
| `regime_signal` | `exo_regime_in1d`, `exo_regime_in5d` = 1{Regime in 1 bzw. 5 Tagen = 1} + Rauschen (σ = 0.25) | Gate besser als Statisch, vor allem bei h = 5 |
| `regime_nosignal` | – | Gate kann das Regime nur indirekt erkennen (`volvol`, `recent_*`) |

**Wie viel ist überhaupt zu holen?** Mit perfekter Kenntnis des Regimes **am Zieldatum** (pro Regime die optimalen statischen Gewichte) sinkt der Loss gegenüber Statisch bei h = 5 um ca. 7 %, bei h = 1 nur um ca. 1.6 % (HAR ist dort in beiden Regimes gut). Wer nur das Regime **am Ursprung** kennt, kommt auf ca. 2 %: Die grossen Fehler entstehen um die Regimewechsel herum. Das Signal kündigt deshalb das Regime am Zieldatum an, wie ein bekannter Termin im Kalender. Das Gate muss selbst lernen, welches Signal zu welchem Horizont gehört.

**Wie man es liest:** Gewinnt das Gate in `regime_signal` nicht gegen Statisch, stimmt etwas an der Methode nicht. Gewinnt es dort, auf den echten Daten aber nicht, fehlt auf den echten Daten das Signal.

---

## 5. Rolling-Origin-Prognosen

- Ein **Ursprung** *t* ist die letzte bekannte Stunde. Jeder Experte sieht nur Daten bis einschliesslich *t*.
- Für jeden Ursprung und jeden **Horizont** *h* ∈ {1, 24} ist das Ziel $RV_{t+h}$, also die RV in 1 bzw. 24 Stunden (Tagesdaten früher: *h* ∈ {1, 5} Tage).
- Das wird für **jede Stunde** wiederholt (rollierend), ab `ORIGIN_START` = 01.01.2025 und frühestens nach 60 Tagen Historie (`MIN_HISTORY` = 1440 Stunden).
- Alle Werte in `pipeline/common.py` sind in Zeitschritten der Daten. Fenster, die in Tagen gemeint sind (HAR, EWMA, Features), rechnen `experts.py` und `features.py` mit `steps_per_day()` um (Stundendaten: × 24).

### Was eine Prognose ist

Jeder Experte liefert pro Ursprung und Horizont keine einzelne Zahl, sondern eine **Verteilung**:

| Ausgabe | Bedeutung |
|---|---|
| `mean` | erwartete RV (Punktprognose) |
| `q0.01` … `q0.99` | 21 Quantile: 1 %, 5 %, 10 %, 15 %, …, 95 %, 99 % |

**RV vs. Prognose:** Die RV ist der **gemessene** Wert (Spalte `y`). Mittelwert und Quantile sind die **Prognose** dieses Werts. Die RV-Historie ist der Input der Experten, die zukünftige RV ist die Wahrheit, mit der verglichen wird.

**Beispiel** (Ursprung 10. Oktober, h = 1):

| | Wert | Bedeutung |
|---|---|---|
| q0.05 | 0.00030 | 5 % Wahrscheinlichkeit, dass die RV darunter liegt |
| q0.50 | 0.00058 | Median |
| q0.95 | 0.00125 | nur 5 % Wahrscheinlichkeit, dass die RV darüber liegt |
| mean | 0.00065 | Erwartungswert (über dem Median, weil die RV rechtsschief ist) |
| **y** | **0.00080** | tatsächlich gemessene RV am 11. Oktober |

### Format der Ergebnisdateien

`results/forecasts/<DATASET>/<SID>__<MODELL>.parquet`, eine Zeile pro Ursprung und Horizont:

| Spalte | Inhalt |
|---|---|
| `sid` | Serie (z.B. `BTCUSDT`) |
| `model` | Experte |
| `h` | Horizont in Zeitschritten (Stunden) |
| `origin` | Datum des Ursprungs *t* |
| `target_date` | Datum des Ziels *t + h* |
| `y` | tatsächliche RV am Zieldatum |
| `mean`, `q0.01` … `q0.99` | Prognose |
| `split` | nur in alten Dateien: train / val / test (wird ignoriert, die Splits entstehen in `04_training`) |

---

## 6. Die Experten

Alle Experten haben dieselbe Schnittstelle: eine Klasse mit `name` und `forecast(s, origins)`, die `mean` mit Form `(n_origins, n_horizons)` und `q` mit Form `(n_origins, n_horizons, 21)` zurückgibt.

### Gemeinsames Prinzip der ökonometrischen Modelle

RW, EWMA, HAR und GARCH prognostizieren einen **Punktwert in Logs**. Die Verteilung entsteht aus den **empirischen Residuen** im Schätzfenster (`EST_WINDOW`: letzte 90 Tage = 2160 Stunden; Tagesdaten früher 1000 Tage):

- Quantile: $\hat q_\tau = \exp(\text{Log-Prognose} + \text{Quantil}_\tau(\text{Residuen}))$
- Mittelwert: Smearing-Schätzer (Duan 1983), $\hat\mu = \exp(\text{Log-Prognose}) \cdot \overline{\exp(\text{Residuen})}$

### Übersicht

| Experte | Input | Idee | Details |
|---|---|---|---|
| **RW** (Random Walk) | log-RV | $RV_{t+h} = RV_t$, "morgen wie heute" | Streuung aus historischen h-Tages-Änderungen der log-RV. Auf echten Daten **nur Baseline**, auf synthetischen Daten auch im Pool. |
| **EWMA** | log-RV | exponentiell geglättete log-RV (Halbwertszeit 5 Tage) | günstiger, kontaminationsfreier Zusatzexperte. Streuung aus den h-Schritt-Fehlern im Schätzfenster. Bisher in keinem Standard-Pool. |
| **HAR** (log-HAR, Corsi 2009) | log-RV | $\ln RV_{t+h} = \beta_0 + \beta_1 \ln RV_t + \beta_2 \ln RV^{(a)}_t + \beta_3 \ln RV^{(b)}_t$ | Stundendaten: letzte Stunde, Tag (a = 24), Woche (b = 168). Tagesdaten: Tag, Woche (5), Monat (22). OLS im rollierenden Fenster, direkte Prognose pro Horizont. Nur Zeilen, deren Ziel bis *t* bekannt ist. |
| **GARCH(1,1)-t** | Renditen | Varianz reagiert auf vergangene Schocks und vergangene Varianz, t-verteilte Innovationen | GARCH modelliert die Varianz der Rendite, nicht die RV. Niveau und Streuung werden deshalb über die Residuen $\ln(RV_u / \sigma_u^2)$ auf die RV kalibriert. Neuschätzung alle 24 Ursprünge (täglich). |
| **Chronos-2** (Amazon, ca. Okt. 2025) | log-RV | vortrainiertes Foundation Model (Transformer), **zero-shot** (kein Training auf diesen Daten) | Kontext: letzte 1024 Schritte (≈ 43 Tage). Gibt die 21 Quantile direkt aus (in Logs, dann `exp`). Mittelwert durch Integration der Quantile. `cross_learning=False`, damit sich Kontexte im Batch nicht gegenseitig Information geben. |
| **TimesFM-2.5** (Google, 200M, ca. Sept. 2025) | log-RV | Foundation Model (Transformer), **zero-shot** | Gleiches Setup wie Chronos (Kontext 1024). Liefert nur die Quantile 0.1 … 0.9, die übrigen 12 werden ergänzt (`expand_quantiles`, siehe Abschnitt 13). Mittelwert durch Integration der Quantile. |
| **TiRex** (NX-AI, 35M, Mai 2025) | log-RV | Foundation Model, **xLSTM (rekurrent)**, zero-shot | nur in 06, 07, 09. Quantile 0.1 … 0.9, ergänzt wie TimesFM. Paket `tirex-ts`. |
| **FlowState** (IBM Granite, 9M, 2025) | log-RV | Foundation Model, **State-Space-Modell**, zero-shot | nur in 06, 07, 09. `scale_factor = 1.0` für Stundendaten mit Tageszyklus. Paket `granite-tsfm`. |
| **SNaive** | log-Wert | gleiche Stunde vor einer Woche | klassisches Lastmodell, nur in 07 und 09. Streuung aus den Wochen-Differenzen im Schätzfenster. |
| **SeasReg** | log-Wert | Regression auf letzten Wert, gleiche Stunde gestern und vor einer Woche, 24-h-Mittel | die "HAR-Version" für Last, nur in 07 und 09. Pro Horizont rollierend geschätzt. |

### Ausreisser in den Quantilen: `cap_quantiles`

Chronos-2 liefert auf allen sieben Volatilitätsreihen vereinzelt absurde obere Quantile (31–157 Prognosen pro Serie, q0.99 bis e⁴² × Median, Mittelwert bis e³⁹ × Median), TiRex in kleinerem Umfang (≤ 22 Zeilen, bis e¹³). Auf Strom gibt es keine. Auf den QLIKE des Mittelwerts wirkt sich das kaum aus, aber der **Pinball-Term im Gate-Loss** explodiert (mittlerer Pinball von Chronos im Training 1.5·10¹² statt ca. 0.7). Jedes Gewicht auf Chronos wurde dadurch bestraft, und "Statisch" gab Chronos 0 %, obwohl es das beste Einzelmodell ist.

`pipeline.common.cap_quantiles(fc, k=8.0)` begrenzt deshalb beim Laden jedes Quantil auf [Median · e⁻⁸, Median · e⁸] und rechnet den Mittelwert in den gekappten Zeilen neu aus den Quantilen. k = 8 (≈ 3000-faches des Medians) trifft nur die Ausreisser von Chronos und TiRex. Breite, aber echte Verteilungen (RW, HAR, GARCH, TimesFM: höchstens ca. e⁷) bleiben unverändert; bei k = 5 wären es über 1000 RW-Zeilen. Die gespeicherten Parquet-Dateien bleiben unverändert.

**Zwei Chronos-Varianten in `results/`:**

| Modellname in den Dateien | Modell | Status |
|---|---|---|
| `Chronos-2` | `autogluon/chronos-2-small` | aus `archiv/volatilitaet_chronos2_har_garch.ipynb` (auf CPU automatisch die kleine Variante); **nicht mehr im Pool** |
| `Chronos-2-full` | `amazon/chronos-2` | aus `archiv/02_timesfm.ipynb`, heute `02_experten.ipynb`; **im Pool** |

Dass `Chronos-2.parquet` vom kleinen Modell stammt, ist geprüft (`results/chronos_check.json`): Die Neuberechnung mit `chronos-2-small` stimmt exakt mit den gespeicherten Quantilen überein (Abweichung 0.0), die mit `chronos-2` nicht (0.084). **Alle Chronos-Ergebnisse im Zwischenbericht stammen vom kleinen Modell.**

### Laufzeiten (Tagesdaten, Sekunden pro Coin, CPU)

Auf Stundendaten gibt es ca. 24-mal mehr Ursprünge. Die Foundation Models laufen deshalb auf Kaggle/Colab mit GPU (T4: ca. 40–90 Min. pro Modell für vier Serien).

| | RW | HAR | GARCH | Chronos-2 (small) | Chronos-2 (voll) | TimesFM-2.5 |
|---|---|---|---|---|---|---|
| BTC | 0.4 | 1.4 | 9.9 | 28.9 | 180 | 545 |
| ETH | 0.5 | 0.8 | 6.7 | 32.7 | 188 | 457 |
| SOL | 0.2 | 0.5 | 5.7 | 28.6 | 174 | 353 |

Quellen: `results/runtimes.csv` und `results/runtimes_timesfm.csv`.

### Neuen Experten hinzufügen

1. In `src/pipeline/experts.py` eine Klasse mit `name` und `forecast(s, origins)` schreiben, die `mean` und `q` im Format oben zurückgibt (Vorlage: `TimesFM25`), und sie in `EXPERTS` eintragen.
2. Den Namen in `MODELS` aufnehmen und das Notebook ausführen. Fehlende Prognosen werden gerechnet, vorhandene geladen.
3. In `03_features.ipynb` den Namen in `POOL` eintragen. Features, Gates, Vergleiche und Auswertungen passen sich automatisch an.

---

## 7. Splits: train, val, test

Die Aufteilung richtet sich nach dem **Zieldatum**, damit kein Trainingslabel Information aus der Validierungsperiode enthält.

| Split | Periode | Zweck |
|---|---|---|
| **train** | bis 31.10.2025 | Gewichte und Gate-Parameter lernen |
| **val** | 01.11.2025 – 28.02.2026 | Hyperparameter (λ) wählen, Ergebnisse anschauen |
| **test** | 01.03.2026 – 08/2026 | **gesperrt** (`ALLOW_TEST = False`), erst für die finale Auswertung |

**Warum diese Grenzen?** Foundation Models wurden auf öffentlichen Daten vortrainiert, möglicherweise auch auf Krypto-Kursen. Liegt eine Evaluationsperiode vor dem Trainings-Cutoff eines Modells, könnte es die Werte schon gesehen haben (**Kontamination**), und der Vergleich wäre nicht mehr zero-shot. Der Cutoff ist meist nicht dokumentiert, das Release-Datum ist eine obere Schranke (Chronos-2 ca. Oktober 2025, TimesFM-2.5 ca. September 2025). Deshalb liegen val und test danach.

Diese Aufteilung entspricht `SPLIT = "fixed"` in `04_training`. Ihr Nachteil: val ist mit 4 Monaten kurz, und λ wird auf denselben Daten gewählt, auf denen ausgewertet wird (leicht optimistisch).

### Walk-Forward (Standard in `04_training`)

Jeden Monat (`FREQ = "MS"`) wird neu trainiert und der nächste Monat out-of-sample prognostiziert. Pro Fold:

| Zeilen | Zieldaten | wofür |
|---|---|---|
| `tune_fit` | bis `TUNE_DAYS` (90) Tage vor dem Trainingsende, minus *h* | jeden λ-Kandidaten fitten |
| `tune_eval` | die letzten 90 Tage des Trainings | λ wählen |
| `train` | alles bis *h* Tage vor Monatsbeginn | mit dem gewählten λ neu fitten |
| `test` | der Monat | auswerten |

- **Kein Leakage:** Die Prognose für Zieldatum *T* hat den Ursprung *T − h*. Dort sind nur Ziele ≤ *T − h* bekannt. Deshalb endet das Training *h* Tage (+ `GAP_DAYS`) vor dem ersten ausgewerteten Ziel.
- **λ wird nur mit Daten vor dem Testmonat gewählt**, die Auswertung ist also nicht optimistisch wie beim festen Split.
- **Kein k-fold CV:** Wegen der Autokorrelation und der überlappenden Ziele bei h = 24 würde Information aus der Zukunft ins Training gelangen.
- Bei Stundendaten sind *h* und die Lücke in Stunden gemeint; `TUNE_DAYS` bleibt in Tagen.
- **Optionen gegen Kontamination:** `TRAIN_START` beschränkt das Training auf Zieldaten nach einem Datum (z.B. dem Release der Foundation Models), `WINDOW_DAYS` macht das Fenster rollierend.
- Auf echten Daten wird standardmässig **Nov. 2025 – Feb. 2026** ausgewertet (= bisherige val-Periode). Die Testperiode bleibt gesperrt: 04 bricht ab, wenn ein Fold Zieldaten ab `test_start` auswertet und `ALLOW_TEST = False` ist.
- Auf synthetischen Daten gibt es keine Sperre. Ausgewertet werden die letzten 40 % der Daten mit quartalsweisem Neutraining.

> Die früheren Notebooks im Archiv verwenden teils noch die **alten** Grenzen (train ≤ 2023, val = 2024). Die gespeicherte `split`-Spalte in alten Prognosedateien wird ignoriert.

---

## 8. Kombination: das Gate

### Grundidee

Das Gate ist ein kleines neuronales Netz. Es macht **keine eigene RV-Prognose**, sondern berechnet pro Coin und Tag **ein Gewicht pro Experte**, bei *M* = 4 Experten also $w = (w_\text{Chronos}, w_\text{TimesFM}, w_\text{HAR}, w_\text{GARCH})$ mit $w_m \ge 0$ und $\sum w_m = 1$. Damit werden **Mittelwert und alle 21 Quantile** der Experten gemischt:

$$\hat\mu = \sum_m w_m \hat\mu_m, \qquad \hat q_\tau = \sum_m w_m \hat q_{m,\tau}$$

Die Kombination ist ein eigener Wert, nicht einfach das beste Modell. Sie liegt aber immer zwischen dem tiefsten und dem höchsten Expertenwert.

### Die Features (Input des Gates)

Die Features sind in **Gruppen** organisiert, die in `03_features` (`GROUPS`) einzeln an- und abgeschaltet werden können. Aktuell (`binance_1h`, Pool Chronos + TimesFM) sind es 6 Marktzustand + 1 Uneinigkeit + 2 jüngste Fehler + 7 Kalender + 3 Volumen + 3 DVOL + 3 Märkte = **25 Features**. Mit dem früheren 4er-Pool auf Tagesdaten waren es 16, mit `LEGACY = True` exakt die früheren 14.

Die Fenster der Features sind in Tagen definiert und werden bei Stundendaten mit 24 multipliziert (z.B. $RV^{(22)}$ = Mittel der letzten 22 · 24 Stunden).

> **Nur Zeitreihe, kein Feature Engineering:** In 06, 07 und 09 sieht das Gate keine dieser Marktfeatures, sondern nur die letzten 24 normierten log-Werte der Reihe (`raw_0` … `raw_23`) und die Uneinigkeit. In 04 testet die Ablation `[disagree, recent]` ("forecast only") dasselbe Prinzip: nur Features, die sich aus den Expertenprognosen selbst ableiten lassen.

Alle Features sind am Ursprungstag *t* bekannt, es gibt also kein Leakage. Sie werden pro Coin und Tag berechnet und **pro Fold** mit Mittelwert und Standardabweichung der Trainingszeilen standardisiert (0 = durchschnittlich, +1 = eine Standardabweichung darüber). Fehlende Werte (Aufwärmphase) werden auf 0 gesetzt.

Notation: $RV^{(k)}$ = gleitender Mittelwert der RV über *k* Tage.

**A. Marktzustand (6 Features)**

| Feature | Formel | misst | Interpretation |
|---|---|---|---|
| `logrv_dev` | $\ln RV_t - \ln RV^{(22)}_t$ | Abweichung der heutigen Vola vom Monatsschnitt | > 0: heute unruhiger als üblich |
| `trend` | $\ln RV^{(5)}_t - \ln RV^{(22)}_t$ | Richtung der Vola | > 0: Vola steigt |
| `level_rel` | $\ln RV^{(22)}_t - \overline{\ln RV}^{(250)}_t$ | Vola-Niveau relativ zum langfristigen Niveau der Serie | hoch: volatile Phase |
| `volvol` | Std.-Abw. von $\ln RV$ über 22 Tage | Volatilität der Volatilität | hoch: RV springt stark, schwer vorherzusagen |
| `ret1` | $ret_t / \sqrt{RV^{(22)}_t}$ | gestrige Rendite relativ zur üblichen Schwankung | gross (±): Schock gestern |
| `ret5` | $\sum_{5} ret / \sqrt{5\,RV^{(22)}_t}$ | Rendite der letzten 5 Tage, standardisiert | anhaltende Bewegung |

Früher war `level` = $\ln RV^{(22)}_t$ absolut. Das ist nicht stationär (das Niveau 2021 unterscheidet sich von 2026) und verrät bei einem gemeinsamen Gate, um welchen Coin es geht. Mit `LEVEL = "absolute"` lässt sich die alte Variante weiter verwenden.

**B. Uneinigkeit (M − 1 Features)**

| Feature | Formel | misst |
|---|---|---|
| `disagree_<Modell>` | $\ln\hat\mu_m - \frac{1}{M}\sum_j \ln\hat\mu_j$ | wie stark die heutige Prognose dieses Experten vom Durchschnitt abweicht (> 0: prognostiziert mehr Vola als die anderen) |

**C. Jüngste Fehler (M − 1 Features pro Halbwertszeit)**

| Feature | Formel | misst |
|---|---|---|
| `recent<HL>_<Modell>` | $\ln L_m - \frac{1}{M}\sum_j \ln L_j$ | wie gut der Experte **zuletzt** war, relativ zu den anderen (< 0: kleinerer Fehler) |

$L_m$ ist der exponentiell gewichtete QLIKE-Fehler (mind. 5 Werte), um *h* verschoben, damit nur Prognosen zählen, deren wahre RV am Tag *t* schon bekannt ist. Es gibt zwei Halbwertszeiten, **10 und 60 Prognosen**: Der QLIKE hat schwere Ausreisser, mit 10 allein ist das Feature sehr verrauscht.

**Warum M − 1?** `disagree_*` und `recent_*` summieren sich über die Experten zu 0. Der letzte Experte im Pool trägt also keine neue Information und wird weggelassen (`DROP_REDUNDANT`). Das ändert die Vorhersagen kaum, macht die Koeffizienten aber eindeutig.

**D. Kalender, externe Information, Serie**

| Gruppe | Feature | misst |
|---|---|---|
| `calendar` | `weekend` | Zielzeitpunkt ist Samstag oder Sonntag (Krypto-Vola ist am Wochenende tiefer) |
| `calendar` (nur Stundendaten) | `hour_sin`, `hour_cos`, `us_open`, `fomc`, `cpi`, `event_day` | Uhrzeit (Tagesgang), US-Börse offen, Stunde der FOMC- bzw. CPI-Veröffentlichung, Tag mit Termin. Alle am **Zielzeitpunkt**; kein Leakage, weil die Termine lange im Voraus feststehen (`pipeline/exo.py`) |
| `volume` | `exo_vol_logrel`, `exo_vol_trades_rel`, `exo_vol_taker` | Handelsvolumen und Trades relativ zum üblichen Niveau, Anteil aggressiver Käufer (aus den 1-Min-Kerzen) |
| `dvol` | `exo_dvol_level`, `exo_dvol_vrp`, `exo_dvol_proxy` | Deribit DVOL (implizite Vola BTC/ETH), Varianz-Risikoprämie (DVOL vs. realisierte Vola), Markierung für SOL (kein eigener DVOL, ETH als Ersatz) |
| `markets` | `exo_mkt_vix`, `exo_mkt_spx_ret`, `exo_mkt_closed` | VIX, Rendite der S&P-500-Futures, Markt geschlossen (Yahoo Finance) |
| `exo` | alle Spalten `exo_*` | Obergruppe von `volume`, `dvol`, `markets` (nie zusammen mit ihnen verwenden), auf synthetischen Daten das Regime-Signal. Werden in `01_daten` angehängt (`exo.add_exo`). Yahoo liefert Stundendaten nur für ca. 730 Tage, der Cache in `data/raw/` ist deshalb die einzige Quelle für ältere Werte. |
| `series` | `series_<SID>` (One-Hot) | Bias pro Coin. Standardmässig **aus**, in `GROUPS` aktivierbar. |

**Keine Features:** `scale` = $RV^{(22)}_t$ dient nur zur Normierung des Pinball-Loss.

### Architektur

**Gating linear** (Hauptmodell):

$$z = b + W x_t, \qquad w = \mathrm{softmax}(z), \quad w_m = \frac{e^{z_m}}{\sum_j e^{z_j}}$$

- $x_t$: die Features (standardisiert)
- $W$ (M × k): wie stark jedes Feature den Score jedes Experten erhöht oder senkt
- $b$ (4): wie stark man jedem Experten **grundsätzlich** vertraut
- Softmax: macht aus beliebigen Scores positive Gewichte mit Summe 1

**Gating MLP** (Ablation): Features → 16 Neuronen (tanh) → M Scores, plus $b$. Kann Kombinationen lernen, z.B. "nur wenn die Vola hoch **und** steigend ist". Bisher hat das MLP nie mehr gefunden als das lineare Gate (es wählt das stärkste λ und wird zu Statisch). Es wird deshalb nur noch als Kontrolle berichtet. Weil das Ergebnis von der zufälligen ersten Schicht abhängt, werden **3 Seeds** trainiert und ihre Gewichte gemittelt.

**Initialisierung:** $b = 0$ und letzte Schicht $= 0$. Jedes Gate startet also bei gleichen Gewichten (je ¼).

Es gibt **ein Gate pro Horizont**, gemeinsam für alle drei Coins. Die Coins teilen die Parameter, jede Zeile sieht aber nur die Features ihres eigenen Coins.

### Rechenbeispiel (ein Tag, erfundene Werte)

Zur Übersicht vereinfacht mit 3 Experten; mit 4 Experten kommt einfach eine Zeile dazu.

Features (Rest = 0): `logrv_dev` = 1.5, `recent_Chronos` = −1.0, `recent_HAR` = 0.5

| | b | W · logrv_dev | W · recent_Chronos | W · recent_HAR | Score z |
|---|---|---|---|---|---|
| Chronos | 0.3 | 0.1 · 1.5 | −0.4 · (−1.0) | 0.1 · 0.5 | **0.90** |
| HAR | 0.1 | −0.2 · 1.5 | 0.2 · (−1.0) | −0.3 · 0.5 | **−0.55** |
| GARCH | −0.4 | 0.3 · 1.5 | 0.1 · (−1.0) | 0.0 · 0.5 | **−0.05** |

Softmax → **w = 0.62 / 0.15 / 0.24**.
Prognosen 0.00070 / 0.00060 / 0.00090 → kombiniert **≈ 0.00073**.

### Zwei Arten von "Gewichten"

| | was | ändert sich wann? |
|---|---|---|
| **Parameter W, b** | die "Regeln" des Gates | nur im Training, danach **fix** |
| **Expertengewichte w** | wie stark jeder Experte heute zählt | **jeden Tag neu**, berechnet aus W, b und den heutigen Features |

Die vergangenen 5 bzw. 22 Tage passen also nicht die Parameter an. Sie beschreiben die aktuelle Marktsituation, auf die die festen Regeln angewendet werden.

### Kombinationsmethoden im Vergleich

| Methode | Gewichte | Parameter |
|---|---|---|
| **Gleichgewicht** | immer 1/M (bei 4 Experten je ¼) | 0 |
| **Auswahl** | 1 für den besten Experten im Training, sonst 0 | 0 |
| **Inverse Fehler** | $w_m \propto 1/\overline{\text{QLIKE}}_m$ im Training (Bates/Granger) | 0 |
| **Inverse Fehler 90T** | wie oben, nur die letzten 90 Tage | 0 |
| **Inverse jüngste Fehler** | $w_{m,t} \propto 1/L_{m,t}$ (EWMA-QLIKE, Halbwertszeit 10), jeden Tag neu, ohne Training | 0 |
| **Statisch** | $\mathrm{softmax}(b)$, gleich für alle Tage | M |
| **Statisch 365T** | wie oben, nur auf dem letzten Jahr (rollierend) | M |
| **Gating linear** | $\mathrm{softmax}(b + Wx_t)$, jeden Tag anders, λ gewählt | M + M·k |
| **Gating linear [Gruppe]** | lineares Gate mit nur bestimmten Feature-Gruppen (`ABLATIONS` in 04: `[disagree, recent]`, `[disagree]`, `[recent]`, `[market]`, `[exo]`, `[volume]`, `[dvol]`, `[markets]`, `[calendar]`): Woher kommt ein allfälliger Gewinn? | |
| **Gating MLP** | wie linear, mit versteckter Schicht, 3 Seeds | |

Die Methoden ohne Parameter sind klassische Kombinationen aus der Literatur und schwer zu schlagen. Das Gate muss zeigen, dass es **darüber hinaus** etwas bringt. In `05_evaluation` wird jede Methode per Diebold-Mariano gegen Gleichgewicht, Statisch und den besten Experten getestet.

"Statisch" ist ein Gate ohne Features. Ein Gate mit sehr grossem λ wird zu "Statisch". Statisch mit $b = 0$ ist "Gleichgewicht".

**Was wird gelernt?** In allen Varianten lernt das Training nur die Parameter des Gates ($b$ und gegebenenfalls $W$). Die Expertengewichte werden nie direkt gelernt, sondern aus diesen Parametern berechnet. Bei "Statisch" einmal für alle Tage, bei den Gates jeden Tag neu aus den Features.

---

## 9. Training des Gates

### Loss

$$\mathcal{L} = \overline{\text{QLIKE}(y, \hat\mu)} + \alpha \cdot \overline{\text{Pinball}(y, \hat q) / s_t} + \lambda \sum W^2$$

- **QLIKE** bewertet den kombinierten Mittelwert (siehe Abschnitt 10).
- **Pinball** bewertet die kombinierten Quantile.
- **$s_t$** = RV-Niveau der letzten 22 Tage: Ohne diese Normierung würden volatile Phasen und Coins das Training dominieren.
- **$\alpha$** = 1: Gewicht des Quantil-Teils.
- **$\lambda \sum W^2$**: L2-Strafe nur auf die Gewichtsmatrizen, nicht auf $b$ und nicht auf Biases im MLP. Sie hält das Gate nahe an statischen Gewichten, solange die Features den Fehler nicht deutlich senken. (Das alte Notebook hat beim MLP auch den Bias der ersten Schicht bestraft. Mit `LEGACY = True` in 04 ist das wieder so.)

### Ablauf

1. Start bei W = 0, b = 0 → gleiche Gewichte.
2. **Alle train-Zeilen auf einmal** (alle Coins, alle Tage) durch das Gate rechnen → Gewichte → kombinierte Prognose → Loss.
3. **Backpropagation**: Für jeden Parameter wird berechnet, in welche Richtung der Loss sinkt.
   *Beispiel:* Lag GARCH an Tagen mit hohem `logrv_dev` meist näher an der echten RV, steigt W[GARCH, logrv_dev]. Ist GARCH im Schnitt schlechter, sinkt b[GARCH].
4. **L-BFGS** passt die Parameter an (full-batch, bis 500 Iterationen, Strong-Wolfe-Linesearch). Schritte 2–4 wiederholen, bis der Loss nicht mehr sinkt.
5. **λ-Auswahl:** Training für jedes λ ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1} (04; in 06/07/09: 1e-4 … 1) auf `tune_fit`, behalten wird das λ mit dem kleinsten Loss auf `tune_eval` (Walk-Forward: die letzten 90 Tage vor dem Testmonat; fester Split: val).
6. Mit dem gewählten λ auf dem ganzen Trainingsfenster neu fitten, dann die Gewichte für den Testmonat berechnen.

Die Experten werden dabei **nicht** verändert, nur die Parameter des Gates. Im Walk-Forward wird das Gate jeden Monat neu trainiert (expandierendes Fenster), dazwischen bleibt es eingefroren. Gespeichert werden pro Fold das gewählte λ (`params.csv`), alle Tuning-Losses (`tuning.csv`), die Gewichte (`weights.parquet`) und die Koeffizienten des linearen Gates (`coef.parquet`).

---

## 10. Evaluation und Metriken

Alle Modelle (Experten als One-Hot-Gewichte und Kombinationen) werden auf **denselben Zeilen** ausgewertet: den Out-of-Sample-Zeilen aller Folds aus `04_training`. Der Vergleich gegen die Baselines ist **gepoolt**: Pro Ursprung wird der QLIKE über die Serien gemittelt, dann wird der Diebold-Mariano-Test auf der Zeitreihe der Differenzen gerechnet. Zusätzlich gibt es den Test pro Serie. Bei synthetischen Daten kann mit `TARGET = "true_var"` gegen die wahre Varianz ausgewertet werden.

| Metrik | Formel / Definition | misst | gut ist |
|---|---|---|---|
| **QLIKE** | $\frac{y}{\hat\mu} - \ln\frac{y}{\hat\mu} - 1$ | Fehler der Punktprognose. Standard für Varianzprognosen, robust gegenüber verrauschten Vola-Proxies (Patton 2011). Bestraft zu tiefe Prognosen stärker. | 0 = perfekt |
| **Pinball** | $\frac{1}{21}\sum_\tau \max\big(\tau(y-\hat q_\tau),\ (\tau-1)(y-\hat q_\tau)\big)$ | Qualität der ganzen Verteilung: kalibriert und möglichst eng | klein |
| **Cov90** | Anteil Tage mit $q_{0.05} \le y \le q_{0.95}$ | Kalibrierung des 90-%-Intervalls | ≈ 0.90 |
| **Pinball9 / Cov80** | wie Pinball bzw. Cov, aber nur mit den Quantilen 0.1 … 0.9 (Cov80: $q_{0.1} \le y \le q_{0.9}$) | Robustheitscheck ohne die ergänzten TimesFM-Quantile (nur in `archiv/02_timesfm.ipynb`) | Pinball9 klein, Cov80 ≈ 0.80 |
| **rel. QLIKE / rel. Pinball** | Metrik des Modells / Metrik des Random Walk, pro Coin und Horizont, dann geometrisches Mittel über die Coins | Vergleich mit der einfachsten Baseline | < 1 |
| **Diebold-Mariano** | Test auf Loss-Differenzen, Newey-West-Varianz | ist der Unterschied zwischen zwei Modellen signifikant? | DM < 0: erstes Modell besser; p < 0.05 |

### Oracle-Analyse (`05_evaluation.ipynb`)

Prüft, ob sich ein Gate überhaupt lohnen kann:

- **Oracle**: wählt rückblickend für jeden Tag den besten Experten. Obergrenze für jedes Gate.
- **max. Gewinn %**: Abstand zwischen bestem Einzelmodell und Oracle. Ist er klein, lohnt sich Routing nicht.
- **Winrate**: wie oft jeder Experte der beste ist.
- **Oracle-Beitrag**: wie viel ein Experte gegenüber dem Rest des Pools beiträgt. Nahe null heisst: überflüssig.
- **Korrelation der Log-Fehler**: Liegen alle Experten in dieselbe Richtung daneben, gibt es wenig zu routen.

---

## 11. Mehrere Domänen: Notebooks 06–09

Das Betreuer-Feedback (neuer Datensatz, weitere Foundation Models, kein Feature Engineering, Forecast Combination Puzzle als mögliches Finding) führte zu vier eigenständigen Notebooks. Gemeinsam ist allen:

- **Gleiche Perioden** wie 04: Ursprünge ab 01.01.2025, Auswertung Nov. 2025 – Feb. 2026 im monatlichen Walk-Forward, λ auf den letzten 90 Tagen vor dem Testmonat, Testperiode ab März 2026 gesperrt (Daten werden nur bis 28.02.2026 geladen).
- **Gleiches Gate, ohne Feature Engineering:** Features sind nur die letzten 24 normierten log-Werte der Reihe (`raw_*`) und die Uneinigkeit der Experten.
- **Pools:** "2 FMs (alt)" (Chronos, TimesFM), "4 FMs" (+ TiRex, FlowState) und "4 FMs + klassisch" (+ HAR/GARCH bzw. SNaive/SeasReg).
- **Methoden:** Gleichgewicht, Auswahl (bester Experte im Training), Statisch, Gate linear, Gate linear [nur disagree], Gate MLP.
- **Ablauf:** Expertenprognosen auf Kaggle (GPU T4, Internet an, "Save & Run All") oder Colab rechnen, das Ergebnis-ZIP lokal in `src/` entpacken. Lokal setzt `ONLY_CACHE` automatisch auf nur Cache, die Gates laufen auf der CPU. Ergebnisse: `results/forecasts/0X_res.parquet` (Prognosen aller Methoden) und `0X_wts.parquet` (Gewichte). Zum Neurechnen der Gates diese beiden Dateien löschen.

| Notebook | Datensatz | Serien | Experten | Loss im Gate | Metrik |
|---|---|---|---|---|---|
| `06_new_models_and_TS` | `binance_1h_neu` | PAXGUSDT (Gold), EURUSDT, XRPUSDT, BNBUSDT; stündliche RV aus 1-Min-Kerzen, 07/2024 – 02/2026 | RW, HAR, GARCH, 4 FMs | QLIKE + Pinball | QLIKE |
| `07_electricity` | `strom_1h` | Netzlast DE, FR, IT, ES (Energy-Charts-API, Fraunhofer ISE, CC BY 4.0), 15 Min → stündlich, 07/2024 – 02/2026 | RW, SNaive, SeasReg, 4 FMs | quadrierter Fehler / Niveau + Pinball | MAPE |
| `08_correlation_vs_win` | alle drei Volatilitäts- und Stromdatensätze | – | nur die 4 FMs | – | – |
| `09_wind` | `wind_1h` | Windstrom onshore DE, FR, ES, PL (Energy-Charts) | RW, SNaive, SeasReg, 4 FMs | wie 07 | sMAE = \|y − ŷ\| / mittleres Niveau |

**Warum Strom und Wind?** Gesucht sind Domänen mit anderer Fehlerstruktur der Foundation Models. Stromlast ist stark saisonal (Tag, Woche) und gut vorhersagbar; die FMs machen dort deutlich verschiedenere Fehler als bei der Volatilität. Wind soll dazwischen liegen: wetterbedingt, aber mit Persistenz und Tagesgang. Temperatur wurde verworfen (negative Werte, kein Log, MAPE sinnlos). Für Wind wird Italien durch Polen ersetzt (wenig Onshore-Wind in Italien), und statt MAPE gilt der skalierte MAE, weil die Erzeugung in windstillen Stunden nahe 0 liegen kann.

**Was 08 berechnet** (alle Volatilitätsanalysen mit `cap_quantiles`):

1. **Fehlerkorrelation vs. Kombinationsgewinn:** pro Serie und Horizont die mittlere paarweise Korrelation der Log-Fehler der FMs und der Gewinn optimaler fester Gewichte gegenüber dem besten FM (bewusst ex post, als Obergrenze). Grafik `results/korrelation_vs_gewinn.png`.
2. **Out-of-sample vs. Obergrenze:** wie viel von dieser Obergrenze die Walk-Forward-Methoden aus 06 und 07 tatsächlich erreichen (`results/oos_vs_obergrenze.png`).
3. **Robustheit Strom:** mit und ohne Feiertage (24.12.–06.01.).
4. **Zerlegung des Fehlers** in einen gemeinsamen Teil (Fehler des FM-Mittels) und einen modellspezifischen Teil (Abweichung jedes FMs vom Mittel). Kombinieren kann nur den modellspezifischen Teil verkleinern.
5. **Haupttest:** Gate linear vs. Auswahl, Diebold-Mariano, Holm-korrigiert über Domänen und Horizonte; dazu Ablationen Gate vs. Statisch und Statisch vs. Auswahl.
6. **Diagnosen:** Werden die Trainings-Losses von wenigen Stunden dominiert? Wie extrem sind die Quantile der FMs (führte zu `cap_quantiles`)?

Ergebnisse in Abschnitt 16.

---

## 12. Resource Check (`00_resource_check.ipynb`)

Prüft vor dem Ausbau, ob weitere Foundation Models lokal laufen. Die Experten bleiben eingefroren, es gibt nur Forward-Passes auf synthetischen Daten (Kontext 512, Horizont 64, 32 Serien).

Gemessen werden Ladezeit, Peak-RAM, Inferenzlatenz pro Batchgrösse und das Output-Format.

**Ergebnis** (`results/resource_check_summary.csv`, Windows 11, CPU, 32 GB RAM):

| Experte | Laden | Prognose 32 Serien | Peak-RAM | Status |
|---|---|---|---|---|
| Chronos-2 | 1 s | 1.0 s | ca. 0.9 GB | ok |
| TimesFM-2.5 (200M) | 27 s | 1.6 s | ca. 2.5 GB | ok |
| Moirai-2.0-R-small | 3 s | – | ca. 2.0 GB | **Fehler** (`ValueError: too many values to unpack`) |

**Wichtige Erkenntnisse:**
- TimesFM: `per_core_batch_size` steht per Default auf 1 und muss beim `compile` erhöht werden, sonst ist es etwa 8× langsamer.
- Moirai braucht einen Fix, bevor es in den Pool kann.
- Die Experten liefern unterschiedliche Quantil-Achsen (Chronos: frei wählbar, TimesFM/Moirai: 0.1 … 0.9). Für die Kombination braucht es ein gemeinsames Raster.
- Prognosen sollten gecacht werden: Alle Kombinationen trainieren auf denselben Expertenausgaben.

---

## 13. TimesFM-2.5 und Chronos-Check (Tagesdaten, Archiv)

> Notebook `archiv/02_timesfm.ipynb`, Stand **Tagesdaten** (h = 1/5 Tage, fester Split). Die Experten-Klassen sind jetzt in `pipeline/experts.py`. Der Chronos-Check, die Quantil-Ergänzung und die Auswertung der Einzelmodelle bleiben als Dokumentation hier.

Dieses Notebook ergänzt den Pool um **TimesFM-2.5** als vierten Experten und prüft, mit welcher Chronos-Variante die bisherigen Prognosen entstanden sind. Es verwendet exakt das Setup aus `archiv/volatilitaet_chronos2_har_garch.ipynb` (gleiche Ursprünge, Horizonte, 21 Quantile). **Bestehende Dateien werden nicht überschrieben**, neue Prognosen kommen als zusätzliche Dateien dazu.

### Ablauf

| Schritt | was passiert | Ergebnis |
|---|---|---|
| 1. Chronos-Check | `chronos-2-small` und `chronos-2` auf 60 BTC-Ursprüngen neu rechnen und mit `BTCUSDT__Chronos-2.parquet` vergleichen | `results/chronos_check.json` |
| 2. TimesFM-2.5 | zero-shot für alle Coins | `results/<COIN>__TimesFM-2.5.parquet` |
| 3. Chronos-2 voll | nur falls Schritt 1 ergibt, dass bisher die kleine Variante lief | `results/<COIN>__Chronos-2-full.parquet` |
| 4. Auswertung | Metriken, Oracle-Analyse, Fehlerkorrelation, Diebold-Mariano, Plot | Ausgaben im Notebook |

### Setup von TimesFM

TimesFM bekommt **dieselben Bedingungen wie Chronos**, damit der Vergleich fair ist:

| | Einstellung | Grund |
|---|---|---|
| Modell | `google/timesfm-2.5-200m-pytorch` | im Resource Check getestet |
| Input | log-RV | wie Chronos; die RV ist stark schief |
| Kontext | 1024 Tage | wie Chronos |
| Horizont | 5 prognostizieren, Schritt 1 und 5 verwenden | wie Chronos |
| `per_core_batch_size` | 128 | Default 1 wäre ca. 8× langsamer |
| `use_continuous_quantile_head` | True | Quantile statt nur Punktwert |
| `fix_quantile_crossing` | True | monotone Quantile |
| `infer_is_positive` | False | log-RV ist negativ, keine Positivitäts-Annahme |

### Von 9 auf 21 Quantile

TimesFM liefert nur die Quantile 0.1, 0.2, …, 0.9. Damit es ohne Änderung in die Pipeline passt, werden die übrigen 12 **in Logs** auf der Normal-Skala $z = \Phi^{-1}(\tau)$ ergänzt:

- **Innen** (0.15, 0.25, …, 0.85): lineare Interpolation zwischen den Nachbarn. Die 9 nativen Werte bleiben exakt erhalten.
- **Ränder** (0.01, 0.05, 0.95, 0.99): Extrapolation mit der lokalen Steigung am jeweiligen Rand (aus $q_{0.1}..q_{0.3}$ bzw. $q_{0.7}..q_{0.9}$). Das entspricht einem normalverteilten Rand der log-RV und erlaubt asymmetrische Ränder.

Das ist eine **Annahme**. Sie wird mit Pinball9 und Cov80 geprüft (nur native Quantile, siehe Abschnitt 10). Ergebnis: Beide Varianten ergeben dasselbe Ranking, und Cov90 von TimesFM auf train liegt bei 0.90. Die Ergänzung verzerrt also nichts.

**Mittelwert:** Wie bei Chronos aus den 21 Quantilen integriert. Den Punktwert von TimesFM verwenden wir **nicht**: Er ist eine Prognose in Logs, und `exp` davon liegt beim Median der RV, nicht beim Erwartungswert. TimesFM wäre damit systematisch zu tief, und QLIKE bestraft genau das stark.

### Ergebnisse der Einzelmodelle

Rel. QLIKE (< 1 = besser als Random Walk), geometrisches Mittel über die Coins:

| | val h=1 | val h=5 | train h=1 | train h=5 | Cov90 val h=1 |
|---|---|---|---|---|---|
| Chronos-2 (small, bisher) | 0.547 | 0.522 | 0.791 | 0.533 | 0.861 |
| **Chronos-2 (voll)** | 0.542 | **0.503** | 0.744 | 0.516 | 0.897 |
| **TimesFM-2.5** | **0.520** | 0.565 | 0.744 | 0.513 | 0.847 |
| HAR | 0.669 | 0.562 | 0.838 | 0.536 | 0.878 |
| GARCH | 0.917 | 0.587 | 1.032 | 0.549 | 0.853 |

- **Chronos voll ist besser als Chronos small**, vor allem auf train und bei der Kalibrierung.
- **TimesFM und Chronos voll sind etwa gleich gut.** Diebold-Mariano findet auf val keinen signifikanten Unterschied (alle p > 0.05).

### Fehlerkorrelation: zwei Gruppen von Experten

Korrelation von $\log(y / \hat\mu)$ auf val, gepoolt über die Coins:

| | h=1 | h=5 |
|---|---|---|
| TimesFM ↔ Chronos voll | **0.98** | **0.99** |
| HAR ↔ GARCH | 0.84 | 0.93 |
| Foundation Models ↔ HAR/GARCH | 0.69–0.81 | 0.80–0.88 |

Die beiden Foundation Models liegen fast immer gleichzeitig und in dieselbe Richtung daneben. Für das Gate ist TimesFM eher ein zweites Chronos als ein neuer Blickwinkel. Entsprechend klein ist sein **Oracle-Beitrag** (0.017 bei h=1, 0.032 bei h=5), und das Oracle verbessert sich mit 4 statt 3 Experten nur um etwa 8–10 %.

### Ergebnisse der Kombination mit 4 Experten (`archiv/01_kombination.ipynb`, val)

| | rel. QLIKE h=1 | rel. QLIKE h=5 | Cov90 h=1 |
|---|---|---|---|
| bestes Einzelmodell | 0.520 (TimesFM) | 0.503 (Chronos voll) | – |
| Gleichgewicht | 0.578 | 0.492 | 0.889 |
| **Statisch** | **0.509** | **0.484** | 0.892 |
| Gating linear | 0.482 | 0.481 | 0.894 |
| Gating MLP | 0.509 (= Statisch) | 0.484 (= Statisch) | 0.892 |

Gelernte statische Gewichte:

| | h=1 | h=5 |
|---|---|---|
| Chronos-2 (voll) | 0.39 | 0.16 |
| TimesFM-2.5 | 0.55 | 0.61 |
| HAR | 0.00 | 0.03 |
| GARCH | 0.07 | 0.20 |

- **Statische Gewichte sind mindestens so gut wie das beste Einzelmodell.** Im Durchschnitt über die Coins ist ihr QLIKE ca. 2 % (h=1) bzw. 4 % (h=5) tiefer als der des besten Einzelmodells. Das gilt aber nicht bei jedem Coin (BTC h=5: Chronos voll allein besser; SOL h=1: gleichauf mit TimesFM) und nicht beim Pinball h=1 (TimesFM 0.730 gegen Statisch 0.740). Diebold-Mariano: Statisch gegen TimesFM nirgends signifikant, gegen Chronos voll nur bei SOL h=1 (p = 0.006). Ein klarer Gewinn ist also nicht belegt. Der praktische Vorteil: Man muss nicht im Voraus wissen, welches Einzelmodell das beste ist (bei h=1 TimesFM, bei h=5 Chronos).
- **GARCH bekommt trotz der schlechtesten Einzelleistung Gewicht**, HAR fast keines. GARCH macht *andere* Fehler als die Foundation Models, HAR liegt zu nah an beiden Gruppen. Ein Experte lohnt sich in der Kombination nicht, weil er gut ist, sondern weil er anders falsch liegt.
- **Das lineare Gate ist nicht verlässlich besser als Statisch.** Der Vorsprung bei h=1 kommt fast nur von ETH, bei BTC und SOL ist das Gate schlechter. Diebold-Mariano Gate gegen Statisch: nirgends signifikant (alle p ≥ 0.14). Das Gate wählt bei h=1 das kleinste λ, und λ wird auf val gewählt. Das Ergebnis ist deshalb optimistisch. Beim Pinball ist das Gate schlechter als Statisch (0.752 gegen 0.740).
- **Das MLP** wählt so starke Regularisierung, dass es exakt zu Statisch wird.

---

## 14. Live-Prognose-App (`app/`)

Die Notebooks bewerten Prognosen für vergangene Zeitpunkte. Die App macht **echte Prognosen für die Zukunft**: Auf Knopfdruck lädt sie die neuesten 1-Minuten-Kerzen von Binance, berechnet die stündliche RV und prognostiziert sie **1 und 24 Stunden** voraus (Datensatz `binance_1h`). Zusätzlich zeigt sie den Kurs und eine **Kursspanne**, die aus der Vola-Prognose abgeleitet ist.

> **Stand:** Die App wurde von Tages- auf Stundendaten umgestellt. Voraussetzung ist, dass 01–04 mit `DATASET = "binance_1h"` gelaufen sind. Pool, Features und λ kommen aus `src/results/runs/binance_1h_walkforward/config.json`. Alles Live-Geschriebene liegt in `src/data/live/binance_1h/`; die alten Tagesdaten in `src/data/live/` bleiben unberührt. Die Beschreibung unten und die Kalibrierungstabelle der Kursspanne stammen noch aus der Tagesversion; die Logik (Monatsmodelle wie im Walk-Forward, Statisch als Hauptprognose, Gate parallel, Testperiode gesperrt, t0/Timestral nur separat) ist gleich geblieben, die Zeiteinheit ist jetzt die Stunde.

### Starten

```bash
.venv\Scripts\activate
pip install streamlit          # einmalig
cd app
streamlit run app.py
```

Die App öffnet sich im Browser. Voraussetzungen: die Prognosen der Experten des Pools in `src/results/forecasts/binance_1h/` (aus `02_experten.ipynb`) und `src/results/runs/binance_1h_walkforward/config.json` (aus `04_training.ipynb`, in Git). Aus dieser Konfiguration übernimmt die App Pool, Features, λ-Kandidaten und Tuning-Tage. Fehlt sie, gelten die Standardwerte.

### Was die App macht

Die App rechnet **mit demselben Code und derselben Logik wie der Walk-Forward in `04_training`**, nur in Echtzeit. Ein Klick auf **«Daten laden und Prognose berechnen»**:

1. **Neue Daten laden:** 5-Minuten-Kerzen seit dem letzten bekannten Tag über die öffentliche Binance-API (`data-api.binance.vision`, ohne Schlüssel). Fällt sie aus, werden die Tagesdateien von `data.binance.vision` geladen. Der heutige UTC-Tag ist unvollständig und wird weggelassen. Die RV wird exakt wie `daily_rv` in `01_daten.ipynb` berechnet.
2. **Experten nachführen:** Für jeden Ursprung nach dem Ende der Forschungsdaten (ab 27.08.2026) rechnen RW, HAR, GARCH, Chronos-2, TimesFM-2.5, t0 und Timestral ihre Prognose (Code aus `pipeline/experts.py`) und speichern sie in `src/data/live/forecasts/`. Beim ersten Klick sind das gut zwei Monate, danach nur die neuen Tage. Sobald ein Ziel bekannt ist, wird es nachgetragen.
3. **Features:** Forschungsprognosen und Live-Prognosen werden zusammengeführt, die Features entstehen wie in `03_features` (`pipeline/features.py`).
4. **Monatsmodell:** Für jeden Zielmonat werden **«Statisch»** und **«Gating linear»** einmal trainiert, genau wie ein Fold im Walk-Forward (`pipeline/combine.make_fold`): alle Ziele bis Monatsbeginn − h, λ auf den letzten 90 Tagen des Trainings gewählt. **Die gesperrte Testperiode (März bis August 2026) bleibt dabei aus dem Training.** Das Modell wird in `src/data/live/models/` gespeichert, alle Prognosen des Monats verwenden es. Beim ersten Klick eines neuen Monats wird neu trainiert (einige Sekunden).
5. **Kombinieren:** Die **Hauptprognose** («Kombination») mischt Chronos-2, TimesFM-2.5, HAR und GARCH mit den Gewichten von «Statisch». Das **Gate** läuft **parallel**: Es berechnet seine Gewichte jeden Tag neu aus den Features, fliesst aber nicht in die Hauptprognose ein. Fehlt ein Experte des Pools, rechnet die Kombination mit den übrigen und den letzten statischen Gewichten, das Gate setzt dann aus.
6. **Kursspanne** aus der Hauptprognose und dem letzten Schlusskurs (siehe unten).
7. **Im Logbuch speichern:** Kombination, Gate, alle Experten und die zweite Kombination mit t0 und Timestral. Ein erneuter Klick am selben Datenstand ersetzt die Zeilen dieses Tages.

Beim ersten Klick werden die Foundation Models geladen und die Experten für die Zeit seit September nachgerechnet. Das dauert einige Minuten, danach geht es schneller.

**Was die Testperiode betrifft:** Ziele von März bis August 2026 werden weder trainiert noch im Logbuch ausgewertet. Als Vergangenheit sind diese Tage aber sichtbar wie jede Historie: im Kontext der Experten und in gleitenden Features (z.B. dem Vola-Niveau der letzten 22 Tage). Für die Wahl von λ bleiben deshalb im November 2026 nur die 61 Tage September und Oktober statt 90.

### Ansichten

| Reiter | Inhalt |
|---|---|
| Bitcoin / Ethereum / Solana | zuletzt gemessene Vola, Prognose 1 und 5 Tage voraus mit 90-%-Bereich, Verlauf der letzten 90 Tage; Schlusskurs und Kursspanne mit Kursverlauf; Vergleich aller Modelle; darunter separat «Gate (parallel)» mit seinen heutigen Gewichten und «Neu, nur live: t0 und Timestral» |
| Logbuch | alle gespeicherten Vola-Prognosen und Kursspannen; sobald der Zieltag vorbei ist, mit tatsächlichem Wert, QLIKE und Trefferquote im 90-%-Bereich; für das Gate und für t0, Timestral und die zweite Kombination je ein Vergleich auf denselben Tagen |
| Methode | Erklärung des monatlichen Trainings und eine Tabelle aller Monatsmodelle (trainiert bis, Zeilen, λ des Gates, Gewichte von «Statisch») |

Angezeigt wird die **annualisierte Volatilität** $\sqrt{365 \cdot RV}$ in Prozent, weil sie leichter zu lesen ist als die RV.

### Kursspanne aus der Vola-Prognose

Die Richtung des Kurses ist praktisch nicht vorhersagbar, seine Schwankungsbreite schon. Die App prognostiziert deshalb nicht den Kurs selbst, sondern eine **Spanne**, in der er mit 90 % (bzw. 50 %) Wahrscheinlichkeit liegen wird.

- **Grundlage:** Die RV eines Tages misst die Varianz der Tagesrendite. Gegeben die Varianz *v* ist die Tagesrendite ungefähr $N(0, v)$.
- **Unsicherheit der Vola:** Statt einer einzigen Varianz wird über die 21 Quantile der kombinierten RV-Prognose gemischt (Gewichte wie beim Mittelwert). Sind sich die Modelle bei der Vola unsicher, wird die Kursspanne breiter.
- **5 Tage:** Die Varianzen der Tage 1 bis 5 werden addiert. Die Tage 2 bis 4 werden in Logs zwischen der 1- und der 5-Tages-Prognose interpoliert; gleiche Quantile werden addiert, weil die RV stark persistent ist.
- **Mitte:** Der letzte Schlusskurs, also keine Annahme über die Richtung.
- **Kurse:** Tages-Schlusskurse (UTC) von Binance, `src/data/live/<COIN>_price.parquet`. Geprüft: Die Renditen aus diesen Schlusskursen sind identisch mit `ret` aus der RV-Berechnung.

**Rückblick auf train und val** (Testperiode ausgeschlossen), kombinierte statische Prognose:

| | Treffer 90-%-Spanne | Treffer 50-%-Spanne | Breite 90 % | Breite 90 % mit historischer 22-Tage-Vola |
|---|---|---|---|---|
| train, 1 Tag | 89.7 % | 53.5 % | 12.2 % | 12.9 % |
| train, 5 Tage | 88.3 % | 51.5 % | 27.6 % | 29.0 % |
| val, 1 Tag | 88.6 % | 53.3 % | 11.1 % | 11.5 % |
| val, 5 Tage | 88.6 % | 50.8 % | 24.8 % | 25.8 % |

Die Spannen sind gut kalibriert und bei ähnlicher Trefferquote 3–5 % schmaler als eine Spanne aus der historischen Vola. Die Spanne ist eine Schwankungsbreite, keine Anlageempfehlung.

### Schutz der Forschungsdaten

- `src/data/<COIN>_daily.parquet` und `src/results/` werden **nur gelesen**. Neue Tage, Kurse und die Logbücher liegen in `src/data/live/` (wie `src/data/` nicht in Git).
- Die Monatsmodelle trainieren nie auf Zielen der Testperiode (März bis August 2026).
- Das Logbuch wertet nur Zieltage **nach** der Testperiode aus (ab 01.09.2026). Die Testperiode wird damit nicht informell geöffnet.

### Das Logbuch als saubere Evaluation

Jede Prognose wird gespeichert, **bevor** der Zieltag eintritt. Kontamination ist damit ausgeschlossen, auch für Modelle, die nach der Testperiode erschienen sind (z.B. t0, Timestral). Nach einigen Monaten regelmässiger Klicks entsteht so eine echte Out-of-Sample-Periode. Ein Klick pro Tag genügt. Mehrere Klicks am selben Datenstand überschreiben sich, statt doppelt zu zählen.

### Neue Modelle: t0 und Timestral (nur live)

**t0-gamma** (The Forecasting Company, `theforecastingcompany/t0-gamma`) und **Timestral-209M** (Mistral AI, `mistralai/Timestral-209M`) sind im September bzw. Oktober 2026 erschienen, also nach train, val und test. Einen Trainings-Cutoff nennen beide nicht. Sie könnten unsere Perioden im Pretraining gesehen haben. Deshalb gilt:

- **Separat:** Sie laufen in der App nur mit, als eigene Modelle und in einer **zweiten Kombination** («Kombination neu»). Die Hauptprognose, das Gate und die Kursspanne verwenden sie nicht. Geprüft: Die Hauptprognose ist mit und ohne die neuen Modelle identisch.
- **Keine gelernten Gewichte:** Aus der Vergangenheit lernen wir für sie nichts, weil diese Daten möglicherweise kontaminiert sind.
- **Gewichte der zweiten Kombination** (gesetzt, nicht gelernt): Die Foundation Models machen fast dieselben Fehler (Fehlerkorrelation 0.98). Deshalb wird ihr Gesamtanteil aus den Gewichten von «Statisch» des aktuellen Monats gleichmässig auf alle vier verteilt, HAR und GARCH behalten ihr Gewicht. Fehlt ein Foundation Model, wird sein Anteil auf die übrigen verteilt.
- **Saubere Auswertung nur im Logbuch:** Das Logbuch vergleicht die zweite Kombination und die neuen Modelle mit der Hauptprognose auf **denselben Tagen**, weil sie erst ab ihrem Start mitlaufen.
- **Setup wie bei Chronos und TimesFM:** log-RV, Kontext 1024 Tage, 21 Quantile (beide liefern genau dieses Raster), Mittelwert durch Integration der Quantile. t0 mit `symmetric_averaging=True` und `trim_leading_missing=True` (vom Hersteller empfohlen), Timestral mit `symmetric_averaging=True` (Default). `clip_nonnegative` ist bei beiden aus, weil die log-RV negativ ist.
- **Optional:** Sind die Pakete nicht installiert, läuft die App wie bisher und zeigt einen Hinweis.

**In den Notebooks:** Beide stehen als Experten in `pipeline/experts.py` (`T0Gamma`, `Timestral`) und in `MODELS` von `02_experten`. Dort wird ihre Historie gerechnet, sobald die Pakete installiert sind. Sie gilt als möglicherweise kontaminiert. Im Standard-Pool von `03_features` sind die beiden **nicht**. Wer sie dort aufnimmt, sollte in `04_training` mit `TRAIN_START` nur auf Zielen nach ihrem Release trainieren. Die erste Prüfung (Plausibilität, Laufzeit) liegt in `archiv/03_t0_timestral.ipynb`.

---

## 15. Konfigurationsparameter

Alle Einstellungen stehen in der ersten Code-Zelle des jeweiligen Notebooks. In 02–05 ist `DATASET` die wichtigste.

### `01_daten.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `SOURCE` | `"binance"` | `"binance"` oder `"synthetic"` |
| `SYMBOLS` | BTCUSDT, ETHUSDT, SOLUSDT | Coins |
| `KLINE_INTERVAL` / `FREQ` | 1m / 1h | Kerzenlänge für die RV, Länge eines RV-Intervalls |
| `FIRST_MONTH` / `LAST_MONTH` | 2023-01 / 2026-08 | Datenzeitraum |
| `MIN_BARS` | 54 | Mindestanzahl Bars pro Stunde (von 60), sonst wird die Stunde verworfen (Tagesdaten: `MIN_BARS_PER_DAY` = 270 von 288) |
| `DATASET` | `"binance_1h"` | Name des Datensatzes (synthetisch: `syn_<Szenario>_s<Seed>`) |
| `TEST_START` | 2026-03-01 | Zieldaten ab hier sind gesperrt |
| `SCENARIO` | `"regime_signal"` | synthetisches Szenario: `null`, `regime_signal`, `regime_nosignal` (Abschnitt 4) |
| `N_SERIES` / `N_DAYS` / `SEED` | 3 / 2500 / 0 | Anzahl und Länge der synthetischen Reihen, Zufallsseed |

### `02_experten.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `MODELS` | RW, EWMA, HAR, GARCH, Chronos-2-full, TimesFM-2.5, t0-gamma, Timestral-209M | Experten, deren Prognosen sichergestellt werden (t0 und Timestral nur als Historie, nicht im Pool) |
| `SKIP_MISSING_PACKAGES` | True | Foundation Model nicht installiert → überspringen |
| `HORIZONS` | [1, 24] | Prognosehorizonte in Stunden (wie die nächsten: in `pipeline/common.py`, gilt auch für die App) |
| `MIN_HISTORY` | 1440 | Stunden Historie vor dem ersten Ursprung (60 Tage) |
| `ORIGIN_START` | 2025-01-01 | erster Ursprung |
| `EST_WINDOW` | 2160 | rollierendes Schätzfenster für RW, EWMA, HAR, GARCH (90 Tage) |
| `GARCH_REFIT_EVERY` | 24 | GARCH alle 24 Ursprünge (täglich) neu schätzen |
| `CONTEXT` / `FM_BATCH` | 1024 / 128 | Kontext und Batchgrösse der Foundation Models (06/07/09 auf GPU: Batch 512) |

Änderungen in `common.py` rechnen gespeicherte Prognosen nicht neu: Die betroffenen Parquet-Dateien löschen.

### `03_features.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `POOL` | Chronos-2-full, TimesFM-2.5 (synthetisch: HAR, GARCH, RW) | Experten, die kombiniert werden |
| `GROUPS` | market, disagree, recent, calendar, volume, dvol, markets (synthetisch: … exo) | Feature-Gruppen (+ `series` für einen Bias pro Serie). `exo` nie zusammen mit `volume`/`dvol`/`markets` |
| `CALENDAR` | weekend, hour, us_open, events | Kalender-Features am Zielzeitpunkt (Tagesdaten: nur weekend) |
| `HALFLIVES` | [10, 60] | Halbwertszeiten der `recent`-Features in Tagen (Stundendaten: × 24) |
| `LEVEL` | `"relative"` | `level_rel` statt absolutem `level` |
| `DROP_REDUNDANT` | True | bei `disagree`/`recent` den letzten Experten weglassen |
| `LEGACY` | False | True: exakt die 14 Features aus `archiv/01_kombination.ipynb` |

### `04_training.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `SPLIT` | `"walkforward"` | oder `"fixed"` (train/val wie bisher) |
| `RUN` | `<DATASET>_<SPLIT>` | Ordner in `results/runs/` |
| `EVAL_START` / `EVAL_END` | 2025-11-01 / 2026-02-28 (synthetisch: letzte 40 %) | ausgewerteter Zeitraum (Zieldaten) |
| `FREQ` | `"MS"` (synthetisch `"QS"`) | Neutraining monatlich bzw. quartalsweise |
| `TRAIN_START` | None | frühestes Zieldatum im Training (gegen Kontamination) |
| `WINDOW_DAYS` | None | None = expandierendes Fenster, sonst rollierend |
| `TUNE_DAYS` | 90 | letzte Tage des Trainings für die Wahl von λ |
| `GAP_DAYS` | 0 | zusätzliche Lücke zwischen Training und Test (Mindestlücke ist h) |
| `TRAIN_END` / `VAL_END` | 2025-10-31 / 2026-02-28 | nur für `SPLIT = "fixed"` |
| `ALLOW_TEST` | False | Sperre der Testperiode |
| `ALPHA` | 1.0 | Gewicht des Pinball-Terms im Loss |
| `LAMBDAS` | 0, 1e-6 … 1 | Kandidaten für die L2-Regularisierung |
| `MLP` / `MLP_HIDDEN` / `MLP_SEEDS` | True / 16 / 3 | MLP-Gate an/aus, Neuronen, Anzahl Seeds |
| `ABLATIONS` | [disagree, recent], [disagree], [recent], [market], [exo], [volume], [dvol], [markets], [calendar] | lineare Gates mit nur diesen Feature-Gruppen; `[disagree, recent]` = "forecast only" (nur Features aus den Expertenprognosen) |
| `LEGACY` | False | True: Methoden genau wie im alten Notebook |

### `05_evaluation.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `RUN` | `<DATASET>_walkforward` | welcher Lauf aus 04 |
| `BASELINE` | RW | Referenz für die relativen Metriken |
| `TARGET` | `"y"` | `"true_var"`: gegen die wahre Varianz auswerten (nur synthetisch) |

---

## 16. Stand, Einschränkungen und nächste Schritte

Alle Zahlen: Walk-Forward mit monatlichem Neutraining, ausgewertet Nov. 2025 – Feb. 2026, λ nur mit Daten vor dem jeweiligen Monat gewählt, Volatilität mit `cap_quantiles` (k = 8). Stand 26.11.2026.

### Kernergebnis: Fehlerkorrelation bestimmt den Nutzen der Kombination (08)

| Domäne | mittlere Fehlerkorrelation der FMs | Gewinn fester Gewichte ex post (Obergrenze) | gemeinsamer Fehler | modellspezifischer Anteil |
|---|---|---|---|---|
| Volatilität (7 Serien) | 0.93 – 0.98 | 0 – 1.7 % | 58–60 % (h=1) / 81–84 % (h=24) | 2 – 5 % |
| Stromlast (4 Länder) | 0.55 – 0.85 | 1.4 – 11.6 % | 1 % / 5 % | 20 – 32 % |

- **Rangkorrelation** zwischen Fehlerkorrelation und Kombinationsgewinn über alle 22 Serie-Horizont-Paare: **−0.76**.
- Bei der Volatilität liegen alle FMs fast immer gleichzeitig und in dieselbe Richtung daneben. Kombinieren kann nur den modellspezifischen Teil des Fehlers verkleinern, und der ist dort winzig. Die FMs sind sich zwar in absoluten Zahlen *uneiniger* als beim Strom (8–19 % vs. 0.7–2.5 %), aber der gemeinsame Fehler ist um ein Vielfaches grösser.
- **Haupttest** (Gate linear vs. Auswahl = bester Experte im Training, Pool 4 FMs, Holm-korrigiert):

| | h=1 | h=24 |
|---|---|---|
| Strom | **−11.8 %** (p_Holm < 0.001) | **−2.4 %** (p_Holm = 0.024) |
| Volatilität | −2.4 % (p_Holm = 0.148) | +0.6 % (p_Holm = 0.744) |

- **Forecast Combination Puzzle:** gilt für die Volatilität (gleiche Gewichte oder das beste Modell sind nicht zu schlagen), nicht für den Strom.
- **Routing vs. Mischen:** Auch beim Strom ist das Gate kaum besser als feste Gewichte. Der Gewinn kommt vom Mischen verschiedener Modelle, nicht davon, je nach Situation umzuschalten.

### Volatilität BTC/ETH/SOL, Stundendaten (04/05, Pool Chronos + TimesFM)

rel. QLIKE gegenüber dem Random Walk (< 1 = besser):

| | h=1 | h=24 |
|---|---|---|
| Chronos-2 (voll) | 0.731 | 0.769 |
| TimesFM-2.5 | 0.752 | 0.736 |
| Gleichgewicht | 0.730 | 0.735 |
| Statisch | 0.732 | 0.734 |
| Gating linear | 0.723 | 0.755 |
| Gating linear [disagree, recent] ("forecast only") | 0.730 | 0.749 |
| Gating linear [market] | 0.727 | 0.778 |
| Gating linear [volume] | 0.727 | 0.728 |
| Gating MLP | 0.721 | 0.760 |

- **Gating linear vs. Gleichgewicht:** −0.94 % bei h=1 (p = 0.114), +2.75 % bei h=24 (p = 0.27). Kein Gate schlägt gleiche Gewichte signifikant.
- **"Forecast only"** (nur Uneinigkeit und jüngste Fehler, keine Marktdaten) ist gleich gut wie Gleichgewicht (+0.03 %, p = 0.82): Aus den Prognosen allein lässt sich nicht lernen, wann welches FM besser ist.
- **Vor `cap_quantiles`** sah es anders aus: Statisch gab TimesFM 100 % (rel. QLIKE 0.752), mehrere Ablationen ([calendar], [dvol], [recent]) schienen zu schaden, und das Gate schien bei h=1 signifikant besser als Gleichgewicht (−1.08 %, p = 0.043). Das waren Artefakte der Chronos-Ausreisser.

### Neue Serien und Foundation Models (06, `binance_1h_neu`)

QLIKE relativ zum besten Einzelexperten des Pools (negativ = besser):

| Pool 4 FMs | h=1 | h=24 |
|---|---|---|
| bester Experte | TiRex | Chronos-2 |
| Gleichgewicht | +0.7 % | +0.4 % |
| Statisch | −0.7 % | 0.0 % (100 % Chronos) |
| Gate linear | −0.2 % | +0.6 % |
| Gate linear [nur disagree] | −0.2 % | −1.6 % |
| Gate MLP | −0.5 % | +1.3 % |

- Auch mit vier FMs aus drei Architekturen (Transformer, xLSTM, State-Space) bleibt der Gewinn bei der Volatilität unter 2 %.
- Pool "4 FMs + HAR/GARCH" ist deutlich schlechter (Gleichgewicht +37 % bei h=1). GARCH hat auf diesen Serien ebenfalls einen auffällig hohen Pinball-Loss, den k = 8 nicht kappt. "Auswahl" wählt dort HAR, weil HAR im Training den kleinsten QLIKE hat, auf der Auswertung aber viel schlechter ist.
- Ausreisser out-of-sample: BNB h=24 −4 bis −6.5 % gegenüber dem besten FM, weil Statisch und Gate auf Chronos setzen, das für BNB nicht das beste ist. Das ist normales Auswahlrisiko.

### Stromlast (07, `strom_1h`)

MAPE der Einzelexperten (Mittel über DE, FR, IT, ES): h=1 TiRex 0.77 %, FlowState ≈ 0.78 %, Chronos ≈ 0.95 %, SeasReg 3.4 %, SNaive 7 %; h=24 Chronos 3.30 %, FlowState ≈ 3.6 %.

| Pool 4 FMs, MAPE vs. bester Experte | h=1 | h=24 |
|---|---|---|
| Gleichgewicht | −7.6 % | +2.5 % |
| Statisch | −10.1 % | −2.7 % |
| Gate linear | −10.8 % | −2.4 % |
| Gate linear [nur disagree] | −10.2 % | −2.4 % |

- Gelernte feste Gewichte: h=1 ca. TiRex 35 %, FlowState 40 %, Chronos 22 %, TimesFM 3 %; h=24 Chronos 72 %, TiRex und FlowState je 13 %.
- SNaive und SeasReg bekommen im Pool "4 FMs + klassisch" fast kein Gewicht. Gleichgewicht wird mit ihnen viel schlechter (+127 % bei h=1).
- Fehlerkorrelation bei h=1: zwischen den FMs 0.49–0.74, zwischen FMs und klassischen Modellen 0.04–0.26.
- Walk-Forward erreicht fast die ganze Obergrenze fester Gewichte ex post (h=1: −10.3 %); das Oracle (pro Stunde der beste Experte) läge bei −53 %.
- Robust ohne Feiertage bei h=1; der kleine Gewinn bei h=24 hängt teilweise an den Feiertagen.

### Windstrom (09, `wind_1h`)

Notebook vorbereitet, Expertenprognosen noch nicht gerechnet (Kaggle, ca. 4–5 h). Hypothese: Fehlerkorrelation zwischen Strom und Volatilität, damit ein dritter Punkt im Diagramm von 08.

### Frühere Ergebnisse auf Tagesdaten

Details und Grafiken: `reports/zwischenstand.html`. Die folgenden Abschnitte beziehen sich auf den Tagesdatensatz `binance` (h = 1/5 Tage) und wurden **vor** `cap_quantiles` gerechnet.

**Walk-Forward auf Tagesdaten** (`binance_walkforward`, 16 Features):

| | rel. QLIKE h=1 | rel. QLIKE h=5 | dQLIKE gegen Statisch h=1 / h=5 (p) |
|---|---|---|---|
| bester Experte | 0.520 (TimesFM) | 0.503 (Chronos voll) | |
| Gleichgewicht | 0.578 | 0.492 | |
| Inverse Fehler | 0.559 | 0.491 | |
| Inverse jüngste Fehler | 0.546 | 0.502 | |
| **Statisch** | **0.510** | **0.488** | |
| Gating linear | 0.498 | 0.506 | −2.5 % (0.36) / +3.8 % (0.27) |
| Gating linear [recent] | 0.507 | 0.486 | −0.6 % (0.86) / −0.4 % (0.93) |
| Gating MLP | 0.489 | 0.523 | −4.1 % (0.07) / +7.6 % (0.05) |

- **Kein Gate schlägt Statisch signifikant.** Das bestätigt den festen Split, jetzt aber ohne den optimistischen Effekt, λ auf den Auswertungsdaten zu wählen.
- **Die klassischen Baselines ohne Parameter** (Inverse Fehler, Inverse jüngste Fehler) sind bei h=1 schlechter als Statisch. Sie geben HAR und GARCH zu viel Gewicht.
- **Das MLP** wird ohne Strafe auf den Biases und mit 3 Seeds nicht mehr zu Statisch. Bei h=1 ist es knapp nicht signifikant besser, bei h=5 schlechter. Das spricht eher für Rauschen als für ein echtes nichtlineares Signal.

**Synthetische Daten** (`syn_regime_signal_s0`, Pool HAR, GARCH, RW, quartalsweises Neutraining, 2397 Testzeilen pro Horizont):

| | dQLIKE gegen Statisch h=1 (p) | dQLIKE gegen Statisch h=5 (p) |
|---|---|---|
| Gating linear | −2.4 % (0.04) | −5.6 % (0.09) |
| **Gating linear [exo]** | −1.4 % (< 0.001) | **−11.0 % (< 0.001)** |
| Gating linear [recent] | −1.0 % (0.003) | +1.7 % (0.19) |
| Gating MLP | +2.4 % (0.19) | +0.1 % (0.98) |

- **Die Methodik funktioniert, wenn es ein Signal gibt:** Das Gate mit dem Regime-Signal ist bei h = 5 deutlich und signifikant besser als statische Gewichte. Es erkennt die Regimes: RW-Gewicht 0.73 im persistenten gegen 0.16 im mean-reverting Regime (h = 5); Statisch bleibt fix bei 0.40.
- **Mit allen Features** ist der Gewinn kleiner (−5.6 %): Zusätzliche verrauschte Features kosten auch bei einem echten Signal.
- **Das MLP findet hier nichts**, obwohl ein Signal da ist. Auf Daten dieser Grösse ist das lineare Gate klar vorzuziehen.
- **Folgerung für die echten Daten:** Dass das Gate dort nicht gewinnt, liegt eher am fehlenden Signal in den Features als an der Methode.

**Mit 4 Experten, fester Split (Details in Abschnitt 13):**

- **Statische Gewichte sind mindestens so gut wie das beste Einzelmodell:** im Durchschnitt ca. 2–4 % tieferer QLIKE, aber nicht bei jedem Coin und kaum signifikant (siehe Abschnitt 13).
- **Das Gate bringt darüber hinaus keinen statistisch nachweisbaren Gewinn** (Diebold-Mariano gegen Statisch: alle p ≥ 0.14). Der Vorsprung des linearen Gates bei h=1 hängt fast nur an ETH. Das MLP wird zu Statisch.
- **Die Experten bilden zwei Gruppen:** Foundation Models (Fehlerkorrelation 0.98–0.99) und Ökonometrie (0.84–0.93). Der Nutzen der Kombination entsteht durch die Diversität zwischen den Gruppen, nicht durch tägliches Routing.
- Die bisherigen Chronos-Prognosen und alle Chronos-Zahlen im Zwischenbericht stammen von `chronos-2-small`. Ab jetzt wird `Chronos-2-full` verwendet.

**Mit dem früheren 3er-Pool (Chronos-2 small, HAR, GARCH; Zwischenbericht):**

- Das lineare Gate ist auf val kaum besser als statische Gewichte. Der Unterschied liegt im Rauschen (das Bootstrap-Intervall enthält 0).
- Das MLP wählt auf val das stärkste λ und wird damit faktisch zu statischen Gewichten.
- Kleines λ senkt nur den train-Fehler, nicht den val-Fehler (Overfitting).
- Das Gate verschiebt Gewicht vor allem zwischen Chronos-2 und HAR (h = 1). Bei h = 5 bleibt es fast statisch.

**Gründe in beiden Fällen:** Die Experten liegen meist in dieselbe Richtung daneben, weil alle nur die vergangene RV sehen. Dazu kommt eine kurze val-Periode (4 Monate).

- **Kontamination:** Chronos-2 ist 2021–2023 kaum besser als HAR, der grosse Vorsprung kommt erst 2024–2025, und im ersten Quartal nach dem Release ist es schlechter als HAR. Das passt zur Hypothese, dass Chronos die Daten im Pretraining gesehen hat, ist aber kein Beweis (es könnte auch ein Regimewechsel sein).

### Nächste Schritte

1. **Wind rechnen (09):** auf Kaggle ausführen, ZIP lokal entpacken, 08 um `wind_1h` (Loss sMAE) in `SETS` und `OOS` erweitern. Vorher prüfen, wie viele Stunden mit Erzeugung ≤ 0 wegfallen.
2. **Testperiode öffnen**, sobald die Methode feststeht (`ALLOW_TEST = True` in 04, `EVAL_END = None`; in 06/07/09 `LAST_MONTH`/`LAST_DAY` und `EVAL_END` erweitern). Erst damit gibt es einen echten Out-of-sample-Beleg für das Kernergebnis.
3. **Mehr Datenpunkte für 08:** weitere Domänen mit positiver Zielgrösse (z.B. Gaspreise, Verkehr), damit die Rangkorrelation −0.76 auf mehr als zwei Domänen beruht.
4. **Kontamination prüfen:** Hängt die hohe Fehlerkorrelation bei Krypto damit zusammen, dass alle FMs ähnliche Daten im Pretraining gesehen haben? Vergleich mit Serien, die kaum in Trainingskorpora stehen dürften.
5. **A-priori-Vorhersage:** Lässt sich der Kombinationsgewinn aus der Fehlerkorrelation auf den **Trainingsdaten** vorhersagen (statt ex post)?
6. **GARCH-Pinball** in `binance_1h_neu` untersuchen (mittlerer Pinball ca. 27 statt ca. 0.7, von k = 8 nicht erfasst).
7. Synthetische Daten: `null` und `regime_nosignal` laufen lassen, mehrere Seeds.
8. **Gate verfeinern** (nur falls es irgendwo Routing-Signal gibt): Sparse Top-k-Gating (Shazeer et al. 2017), ein Gate über alle Horizonte mit h als Feature, Gewichte pro Quantil.
9. App: nach einigen Monaten im Logbuch prüfen, ob das Gate live besser ist als «Statisch».

Erledigt: TimesFM-2.5, Walk-Forward, synthetische Daten, Stundendaten mit exogenen Features, weitere FMs (TiRex, FlowState), neue Serien (Gold, EUR/USD, XRP, BNB), zweite Domäne (Strom), Gate ohne Feature Engineering, Korrelation vs. Gewinn, `cap_quantiles`.

### Offene technische Punkte

- `requirements.txt` vervollständigen (`psutil`, `timesfm`, `tirex-ts`, `granite-tsfm`, `uni2ts`, `gluonts`). `arch`, `scipy`, `requests`, `tqdm` und `jupyter` sind ergänzt.
- Die Markdown-Zellen in `08_correlation_vs_win.ipynb` nennen teilweise noch Zahlen von **vor** `cap_quantiles` (z.B. Volatilität −0.7 % / +0.8 % im Haupttest). Die Code-Ausgaben sind aktuell.
- `src/results/runs/binance_1h_walkforward/forecasts.parquet` ist ca. 60 MB (GitHub warnt ab 50 MB). Langfristig in `.gitignore` oder Git LFS.
- ~~Dokumentieren, welche Chronos-2-Variante die gespeicherten Prognosen erzeugt hat~~ (erledigt: `chronos-2-small`, siehe `results/chronos_check.json`).
- ~~`CHRONOS_MODEL` fest auf `amazon/chronos-2` setzen~~ (erledigt: `02_experten` unterscheidet `Chronos-2-full` und `Chronos-2` explizit).
- Modellversionen festhalten: Chronos-2, TimesFM-2.5 und Moirai-2.0 sind neuere Generationen als die in den Originalpapern evaluierten.

---

## 17. Glossar

| Begriff | Bedeutung |
|---|---|
| **Experte** | ein einzelnes Prognosemodell (Chronos-2, HAR, GARCH, …) |
| **Gate / Gating** | neuronales Netz, das die Gewichte der Experten berechnet |
| **Mixture of Experts (MoE)** | Architektur, bei der ein Gate mehrere Experten kombiniert |
| **Foundation Model** | grosses, auf vielen Zeitreihen vortrainiertes Modell, das ohne weiteres Training (zero-shot) prognostiziert |
| **Zero-shot** | Prognose ohne Training auf den Zieldaten |
| **Ursprung (origin)** | letzter bekannter Tag einer Prognose |
| **Horizont (h)** | wie viele Tage voraus prognostiziert wird |
| **RV** | realisierte Varianz: Summe der quadrierten 5-Minuten-Renditen eines Tages |
| **Quantil** | Wert, unter dem die RV mit einer bestimmten Wahrscheinlichkeit liegt |
| **Softmax** | wandelt Scores in positive Gewichte mit Summe 1 um |
| **L-BFGS** | Optimierungsverfahren, das alle Daten gleichzeitig verwendet (full-batch) |
| **λ (Lambda)** | Stärke der Regularisierung: hält das Gate nahe an statischen Gewichten |
| **Leakage** | Information aus der Zukunft gelangt unbemerkt ins Training |
| **Kontamination** | ein Foundation Model hat die Evaluationsdaten im Pretraining gesehen |
| **Forecast Combination Puzzle** | empirischer Befund (Stock & Watson 2004, Timmermann 2006): einfache gleiche Gewichte schlagen geschätzte Gewichte oft |
| **sMAE** | skalierter MAE, \|y − ŷ\| / mittleres Niveau; Metrik für Windstrom (robust bei Werten nahe 0) |
| **Oracle** | wählt rückblickend jeden Tag den besten Experten; theoretische Obergrenze |
| **Fehlerkorrelation** | wie stark zwei Experten gleichzeitig in dieselbe Richtung danebenliegen; hoch = wenig Nutzen in der Kombination |
| **Kovariate** | zusätzliche Zeitreihe, die ein Modell neben der Zielgrösse sieht (z.B. implizite Vola, Volumen) |

### Literatur

- Corsi (2009): A Simple Approximate Long-Memory Model of Realized Volatility (HAR)
- Bollerslev (1986): Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
- Patton (2011): Volatility Forecast Comparison Using Imperfect Volatility Proxies (QLIKE)
- Duan (1983): Smearing Estimate (Rücktransformation aus Logs)
- Diebold & Mariano (1995): Comparing Predictive Accuracy
- Shazeer et al. (2017): Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- Bates & Granger (1969): The Combination of Forecasts
- Stock & Watson (2004): Combination Forecasts of Output Growth in a Seven-Country Data Set
- Timmermann (2006): Forecast Combinations (Handbook of Economic Forecasting)
- Holm (1979): A Simple Sequentially Rejective Multiple Test Procedure
- Ansari et al.: Chronos (Amazon), Das et al.: TimesFM (Google), Woo et al.: Moirai (Salesforce)
