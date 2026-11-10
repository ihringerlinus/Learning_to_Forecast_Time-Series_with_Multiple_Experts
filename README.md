# Learning to Forecast Time Series with Multiple Experts

Bachelorprojekt (HSG). Mehrere Prognosemodelle ("Experten") sagen die tägliche Volatilität von Kryptowährungen voraus. Ein kleines neuronales Netz (das **Gate**) lernt, wie stark es jedem Experten je nach Marktsituation vertrauen soll, und kombiniert ihre Prognosen.

**Forschungsfrage:** Ist eine gelernte, situationsabhängige Kombination von Experten (Foundation Models und klassische Ökonometrie) besser als das beste Einzelmodell oder feste Gewichte?

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
11. [Resource Check](#11-resource-check-00_resource_checkipynb)
12. [TimesFM-2.5 und Chronos-Check](#12-timesfm-25-und-chronos-check-archiv02_timesfmipynb)
13. [Live-Prognose-App](#13-live-prognose-app-app)
14. [Konfigurationsparameter](#14-konfigurationsparameter)
15. [Stand, Einschränkungen und nächste Schritte](#15-stand-einschränkungen-und-nächste-schritte)
16. [Glossar](#16-glossar)

---

## 1. Überblick

Die Pipeline besteht aus fünf Notebooks, eines pro Data-Science-Schritt. Sie tauschen ihre Ergebnisse über Dateien aus:

```
01_daten        Binance oder synthetische Daten ──► tägliche RV pro Serie        data/<DATASET>.json
02_experten     RW, EWMA, HAR, GARCH, Chronos-2, TimesFM-2.5 (Stufe 1, gecacht)   <SID>__<MODELL>.parquet
                                                 (Mittelwert + 21 Quantile pro Ursprung und Horizont)
03_features     Prognosen + Tagesdaten ──► Features, ein Panel pro Horizont        results/panels/<DATASET>/
04_training     Split (Walk-Forward) ──► λ wählen ──► Baselines und Gates fitten   results/runs/<RUN>/
05_evaluation   Out-of-Sample-Vergleich: Experten vs. Baselines vs. Gate, DM-Tests, Oracle, Gewichte
```

**Datenquelle austauschbar:** `01_daten` schreibt echte (`binance`) oder synthetische Daten (`syn_<Szenario>_s<Seed>`) in dasselbe Format. Die Notebooks 02–05 brauchen nur `DATASET = "..."` in der ersten Zelle.

**Aktueller Pool (echte Daten):** Chronos-2 (voll), TimesFM-2.5, HAR, GARCH. Der Random Walk dient als Baseline.

- **Stufe 1 (02)** ist teuer (Modelle laufen über Tausende Tage) und wird einmal gerechnet. Die Ergebnisse werden gecacht.
- **Stufe 2 (03–05)** ist billig und arbeitet nur auf den gespeicherten Prognosen.

Die früheren Notebooks liegen unverändert in `src/archiv/` (siehe Abschnitt 2). Mit `LEGACY = True` in 03 und 04 sowie `SPLIT = "fixed"` reproduzieren die neuen Notebooks deren Ergebnisse exakt.

---

## 2. Projektstruktur

```
.
├── README.md
├── requirements.txt
├── prognose.png                  Plot aus einem frühen Prototyp (Chronos-2 + Toto, S&P 500)
├── app/
│   ├── app.py                    Live-Prognose-App (Streamlit), siehe Abschnitt 13
│   └── .streamlit/config.toml    Farben und Darstellung der App
├── reports/
│   └── zwischenstand.html        Zwischenbericht mit Grafiken und Ergebnissen
└── src/
    ├── 00_resource_check.ipynb   laufen Chronos-2, TimesFM und Moirai lokal? RAM, Latenz
    ├── 01_daten.ipynb            Daten laden (Binance oder synthetisch), RV, Format prüfen
    ├── 02_experten.ipynb         Stufe 1: Rolling-Origin-Prognosen aller Experten (gecacht)
    ├── 03_features.ipynb         Features für das Gate, ein Panel pro Horizont
    ├── 04_training.ipynb         Split, Hyperparameterwahl, Baselines und Gates fitten
    ├── 05_evaluation.ipynb       Vergleich mit Experten und Baselines, DM-Tests, Oracle
    ├── archiv/                   frühere Notebooks (Zwischenbericht), durch 01–05 ersetzt
    │   ├── volatilitaet_chronos2_har_garch.ipynb
    │   ├── 02_timesfm.ipynb      inkl. Chronos-Check (Abschnitt 12)
    │   └── 01_kombination.ipynb
    ├── data/                     (nicht in Git, siehe .gitignore)
    │   ├── raw/<COIN>/<COIN>-5m-YYYY-MM.zip   Binance-Monatsdateien
    │   ├── <COIN>_daily.parquet  tägliche RV, Rendite, Anzahl Bars (binance)
    │   ├── <DATASET>/            tägliche Daten anderer Datensätze (z.B. synthetisch)
    │   └── <DATASET>.json        Beschreibung des Datensatzes (Serien, Pfade, Testsperre)
    └── results/
        ├── <COIN>__<MODELL>.parquet   Prognosen der Experten auf echten Daten
        │                              (RW, EWMA, HAR, GARCH, Chronos-2, Chronos-2-full, TimesFM-2.5)
        ├── forecasts/<DATASET>/       Prognosen der Experten auf anderen Datensätzen
        ├── panels/<DATASET>/          Panels aus 03 (h1.parquet, h5.parquet, panel.json)
        ├── runs/<RUN>/                Ergebnisse aus 04 (Prognosen, Gewichte, λ, Koeffizienten, Konfiguration)
        ├── runtimes.csv               Laufzeiten der Experten
        ├── runtimes_timesfm.csv       Laufzeiten aus archiv/02_timesfm.ipynb
        ├── chronos_check.json         welche Chronos-Variante steckt in Chronos-2.parquet?
        └── resource_check_*.csv/json  Ergebnisse des Resource Checks
```

Die echten Daten bleiben an den bisherigen Orten (`data/<COIN>_daily.parquet`, `results/<COIN>__<MODELL>.parquet`), weil die App sie liest. Die Notebooks im Archiv erwarten ihre Pfade relativ zu `src/`. Zum erneuten Ausführen zurück nach `src/` verschieben.

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
> # für t0 und Timestral (03_t0_timestral.ipynb und App, optional):
> pip install "tfc-t0[torch]>=0.6.0"
> pip install git+https://github.com/mistralai/timestral.git
> # nur für den Resource Check (Moirai):
> pip install uni2ts gluonts
> ```

### Reihenfolge

Die Notebooks verwenden relative Pfade (`data/`, `results/`) und müssen deshalb **aus dem Ordner `src/`** gestartet werden.

| Schritt | Datei | was passiert | Dauer |
|---|---|---|---|
| (optional) | `00_resource_check.ipynb` | prüft, ob die Foundation Models lokal laufen | Minuten |
| 1 | `01_daten.ipynb` | Binance-Daten herunterladen und RV berechnen, oder synthetische Reihen simulieren | Sekunden (mit Cache) |
| 2 | `02_experten.ipynb` | Prognosen aller Experten in `MODELS`, vorhandene werden nur geladen | Sekunden bis 45 Min. (Foundation Models auf CPU) |
| 3 | `03_features.ipynb` | Features und Panels bauen | Sekunden |
| 4 | `04_training.ipynb` | Walk-Forward: pro Monat λ wählen, Baselines und Gates fitten, out-of-sample prognostizieren | Minuten (das MLP ist der langsamste Teil) |
| 5 | `05_evaluation.ipynb` | Tabellen, Diebold-Mariano gegen die Baselines, Oracle, Gewichte, Koeffizienten | Sekunden |
| (optional) | `app/app.py` | Live-Prognose mit den neuesten Daten (Abschnitt 13) | ca. 1 Min. pro Klick |

**Datensatz wechseln:** in `01_daten` `SOURCE = "synthetic"` (und `SCENARIO`) setzen, danach in 02–05 `DATASET` auf den ausgegebenen Namen (z.B. `"syn_regime_signal_s0"`). In 05 zusätzlich `RUN` anpassen, falls in 04 ein anderer Name oder Split gewählt wurde.

**Caching:** Existiert eine Prognosedatei schon, wird sie nur geladen. Um einen Experten neu zu rechnen, die entsprechende Parquet-Datei löschen. Ändern sich die Daten eines Datensatzes (z.B. andere Szenario-Parameter), bricht `01_daten` ab, statt mit veralteten Prognosen weiterzurechnen.

Der Ordner `src/data/` ist nicht in Git. Auf einem neuen Rechner lädt `01_daten` die Daten automatisch von `data.binance.vision` (kostenlos, ohne API-Key).

---

## 4. Daten und Zielgrösse: realisierte Varianz (RV)

### Daten

| | |
|---|---|
| Quelle | Binance Spot, 5-Minuten-Kerzen (`data.binance.vision`) |
| Coins | BTCUSDT, ETHUSDT (ab 01/2020), SOLUSDT (ab 08/2020) |
| Zeitraum | bis 08/2026 (letzter vollständiger Monat) |
| Bars pro Tag | 288 (Krypto handelt 24/7, keine Overnight-Lücke) |
| Qualitätsfilter | Tage mit weniger als 270 Bars werden verworfen |

### Was ist die RV?

Die **realisierte Varianz** fasst die Schwankung eines Tages in **einer Zahl** zusammen. Für jeden UTC-Tag *d*:

$$RV_d = \sum_{i} r_{d,i}^2, \qquad r_{d,i} = \ln P_{d,i} - \ln P_{d,i-1}$$

Dabei ist $P_{d,i}$ der Schlusskurs der *i*-ten 5-Minuten-Kerze. Zusätzlich wird die **Tagesrendite** $ret_d = \sum_i r_{d,i}$ gespeichert.

Code (`daily_rv` in `01_daten.ipynb`):

```python
r = np.log(close).diff().dropna()
rv  = (r**2).groupby(day).sum()
ret = r.groupby(day).sum()
```

**Warum RV und nicht einfach die Tagesrendite?** Ein Tag, der +1 % schliesst, kann ruhig gewesen sein oder zwischendurch −7 % und +7 % gesehen haben. Die Tagesrendite ist in beiden Fällen gleich, die RV nicht. Die RV misst also, wie stark sich der Preis **während** des Tages bewegt hat. Die "wahre" Volatilität ist nicht beobachtbar, und die RV ist eine viel genauere Schätzung davon als die quadrierte Tagesrendite.

**Warum 5 Minuten?** Bei sehr kurzen Intervallen springt der Preis zwischen Geld- und Briefkurs (Mikrostruktur-Rauschen), und die RV wird künstlich zu gross. 5 Minuten ist der Standardkompromiss in der Literatur.

### Grössenordnung (BTC, Beispiel)

| Grösse | typischer Wert |
|---|---|
| 5-Minuten-Rendite | ca. ±0.15 % |
| Tagesvolatilität (Std.-Abw.) | ca. 2.5 % |
| **RV (Tagesvarianz)** | ca. 0.000625 |
| Annualisierte Volatilität $\sqrt{365 \cdot RV}$ | ca. 48 % |

Die RV ist stark **rechtsschief**: Die meisten Tage sind ruhig, wenige haben extreme Spitzen. Deshalb rechnen die Modelle meist mit **log-RV**.

`data/<COIN>_daily.parquet` enthält die Spalten `rv`, `ret`, `n_bars` mit dem Datum als Index.

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

- Ein **Ursprung** *t* ist der letzte bekannte Tag. Jeder Experte sieht nur Daten bis einschliesslich *t*.
- Für jeden Ursprung und jeden **Horizont** *h* ∈ {1, 5} ist das Ziel $RV_{t+h}$, also die RV in 1 bzw. 5 Tagen.
- Das wird für **jeden Tag** wiederholt (rollierend), ab Tag 500 der Reihe (`MIN_HISTORY`).

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

`results/<COIN>__<MODELL>.parquet`, eine Zeile pro Ursprung und Horizont:

| Spalte | Inhalt |
|---|---|
| `sid` | Serie (z.B. `BTCUSDT`) |
| `model` | Experte |
| `h` | Horizont in Tagen |
| `origin` | Datum des Ursprungs *t* |
| `target_date` | Datum des Ziels *t + h* |
| `y` | tatsächliche RV am Zieldatum |
| `mean`, `q0.01` … `q0.99` | Prognose |
| `split` | nur in alten Dateien: train / val / test (wird ignoriert, die Splits entstehen in `04_training`) |

---

## 6. Die Experten

Alle Experten haben dieselbe Schnittstelle: eine Klasse mit `name` und `forecast(s, origins)`, die `mean` mit Form `(n_origins, n_horizons)` und `q` mit Form `(n_origins, n_horizons, 21)` zurückgibt.

### Gemeinsames Prinzip der ökonometrischen Modelle

RW, EWMA, HAR und GARCH prognostizieren einen **Punktwert in Logs**. Die Verteilung entsteht aus den **empirischen Residuen** im Schätzfenster (letzte 1000 Tage):

- Quantile: $\hat q_\tau = \exp(\text{Log-Prognose} + \text{Quantil}_\tau(\text{Residuen}))$
- Mittelwert: Smearing-Schätzer (Duan 1983), $\hat\mu = \exp(\text{Log-Prognose}) \cdot \overline{\exp(\text{Residuen})}$

### Übersicht

| Experte | Input | Idee | Details |
|---|---|---|---|
| **RW** (Random Walk) | log-RV | $RV_{t+h} = RV_t$, "morgen wie heute" | Streuung aus historischen h-Tages-Änderungen der log-RV. Auf echten Daten **nur Baseline**, auf synthetischen Daten auch im Pool. |
| **EWMA** | log-RV | exponentiell geglättete log-RV (Halbwertszeit 5 Tage) | günstiger, kontaminationsfreier Zusatzexperte. Streuung aus den h-Schritt-Fehlern im Schätzfenster. Bisher in keinem Standard-Pool. |
| **HAR** (log-HAR, Corsi 2009) | log-RV | $\ln RV_{t+h} = \beta_0 + \beta_1 \ln RV_t + \beta_2 \ln RV^{(5)}_t + \beta_3 \ln RV^{(22)}_t$ | Tag, Woche, Monat. OLS im rollierenden 1000-Tage-Fenster, direkte Prognose pro Horizont. Nur Zeilen, deren Ziel bis *t* bekannt ist. |
| **GARCH(1,1)-t** | Tagesrenditen | Varianz reagiert auf vergangene Schocks und vergangene Varianz, t-verteilte Innovationen | GARCH modelliert die Varianz der Tagesrendite, nicht die RV. Niveau und Streuung werden deshalb über die Residuen $\ln(RV_u / \sigma_u^2)$ auf die RV kalibriert. Neuschätzung alle 5 Ursprünge. |
| **Chronos-2** (Amazon) | log-RV | vortrainiertes Foundation Model, **zero-shot** (kein Training auf diesen Daten) | Kontext: letzte 1024 Tage. Gibt die 21 Quantile direkt aus (in Logs, dann `exp`). Mittelwert durch Integration der Quantile. `cross_learning=False`, damit sich Kontexte im Batch nicht gegenseitig Information geben. |
| **TimesFM-2.5** (Google, 200M) | log-RV | vortrainiertes Foundation Model, **zero-shot** | Gleiches Setup wie Chronos (Kontext 1024). Liefert nur die Quantile 0.1 … 0.9, die übrigen 12 werden ergänzt (siehe Abschnitt 12). Mittelwert durch Integration der Quantile. |

**Zwei Chronos-Varianten in `results/`:**

| Modellname in den Dateien | Modell | Status |
|---|---|---|
| `Chronos-2` | `autogluon/chronos-2-small` | aus `archiv/volatilitaet_chronos2_har_garch.ipynb` (auf CPU automatisch die kleine Variante); **nicht mehr im Pool** |
| `Chronos-2-full` | `amazon/chronos-2` | aus `archiv/02_timesfm.ipynb`, heute `02_experten.ipynb`; **im Pool** |

Dass `Chronos-2.parquet` vom kleinen Modell stammt, ist geprüft (`results/chronos_check.json`): Die Neuberechnung mit `chronos-2-small` stimmt exakt mit den gespeicherten Quantilen überein (Abweichung 0.0), die mit `chronos-2` nicht (0.084). **Alle Chronos-Ergebnisse im Zwischenbericht stammen vom kleinen Modell.**

### Laufzeiten (Sekunden pro Coin, CPU)

| | RW | HAR | GARCH | Chronos-2 (small) | Chronos-2 (voll) | TimesFM-2.5 |
|---|---|---|---|---|---|---|
| BTC | 0.4 | 1.4 | 9.9 | 28.9 | 180 | 545 |
| ETH | 0.5 | 0.8 | 6.7 | 32.7 | 188 | 457 |
| SOL | 0.2 | 0.5 | 5.7 | 28.6 | 174 | 353 |

Quellen: `results/runtimes.csv` und `results/runtimes_timesfm.csv`.

### Neuen Experten hinzufügen

1. In `02_experten.ipynb` eine Klasse mit `name` und `forecast(s, origins)` schreiben, die `mean` und `q` im Format oben zurückgibt (Vorlage: `TimesFM25`), und sie in `EXPERTS` eintragen.
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
- **Kein k-fold CV:** Wegen der Autokorrelation und der überlappenden Ziele bei h = 5 würde Information aus der Zukunft ins Training gelangen.
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

Die Features sind in **Gruppen** organisiert, die in `03_features` (`GROUPS`) einzeln an- und abgeschaltet werden können. Mit dem 4er-Pool und den Standardeinstellungen sind es 6 Marktzustand + 3 Uneinigkeit + 2 × 3 jüngste Fehler + 1 Kalender = **16 Features** (plus allfällige `exo_*`). Mit `LEGACY = True` entstehen exakt die früheren 14.

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
| `calendar` | `weekend` | Zieldatum ist Samstag oder Sonntag (Krypto-Vola ist am Wochenende tiefer) |
| `exo` | alle Spalten `exo_*` der Tagesdaten | externe Kovariaten, z.B. DVOL, Volumen, Funding-Rates oder das synthetische Regime-Signal. Werden in `01_daten` an die Tagesdaten gehängt und automatisch übernommen. |
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
| **Gating linear [Gruppe]** | lineares Gate mit nur einer Feature-Gruppe (`recent`, `market`, `exo`): Woher kommt ein allfälliger Gewinn? | |
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
5. **λ-Auswahl:** Training für jedes λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} auf `tune_fit`, behalten wird das λ mit dem kleinsten Loss auf `tune_eval` (Walk-Forward: die letzten 90 Tage vor dem Testmonat; fester Split: val).
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

## 11. Resource Check (`00_resource_check.ipynb`)

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

## 12. TimesFM-2.5 und Chronos-Check (`archiv/02_timesfm.ipynb`)

> Die Experten-Klassen dieses Notebooks sind jetzt in `02_experten.ipynb`. Der Chronos-Check und die Auswertung der Einzelmodelle bleiben im Archiv.

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

## 13. Live-Prognose-App (`app/`)

Die Notebooks bewerten Prognosen für vergangene Tage. Die App macht **echte Prognosen für die Zukunft**: Auf Knopfdruck lädt sie die neuesten Binance-Daten und prognostiziert die Volatilität für den nächsten Tag und in 5 Tagen. Zusätzlich zeigt sie den Kurs und eine **Kursspanne**, die aus der Vola-Prognose abgeleitet ist.

### Starten

```bash
.venv\Scripts\activate
pip install streamlit          # einmalig
cd app
streamlit run app.py
```

Die App öffnet sich im Browser. Voraussetzung: Die Prognosen von `Chronos-2-full` und `TimesFM-2.5` liegen in `src/results/` (heute aus `02_experten.ipynb`, ursprünglich aus `archiv/02_timesfm.ipynb`) Die App braucht sie, um die statischen Gewichte zu bestimmen.

### Was die App macht

Ein Klick auf **«Daten laden und Prognose berechnen»**:

1. **Neue Daten laden:** 5-Minuten-Kerzen seit dem letzten bekannten Tag über die öffentliche Binance-API (`data-api.binance.vision`, ohne Schlüssel). Fällt sie aus, werden die Tagesdateien von `data.binance.vision` geladen. Der heutige UTC-Tag ist unvollständig und wird weggelassen.
2. **RV berechnen:** exakt wie `daily_rv` in `01_daten.ipynb`. Geprüft: Für August 2026 ergibt die App dieselben Werte wie die gespeicherten Daten (Abweichung 0).
3. **Experten rechnen** mit dem letzten Tag als Ursprung: RW, HAR, GARCH, Chronos-2 voll und TimesFM-2.5 (Setup wie in `02_experten.ipynb`). RW und HAR sind identisch zu den Notebooks (Abweichung ca. 10⁻¹⁵). GARCH weicht um 1–3 % ab, weil die Notebooks nur alle 5 Tage neu schätzt und die App jedes Mal frisch.
4. **Kombinieren** mit den statischen Gewichten aus train (bis 31.10.2025), gleiche Zielfunktion wie «Statisch» in `04_training.ipynb`. Die App findet dieselben Gewichte wie das Notebook. Fällt ein Experte aus (z.B. Modell nicht ladbar), werden die übrigen neu gewichtet und die App zeigt einen Hinweis.
5. **Kursspanne berechnen** aus der kombinierten Vola-Prognose und dem letzten Schlusskurs (siehe unten).
6. **Im Logbuch speichern**, Vola-Prognose und Kursspanne getrennt.

Beim ersten Klick werden die Modelle geladen (ca. 30 Sekunden), danach geht es schneller.

### Ansichten

| Reiter | Inhalt |
|---|---|
| Bitcoin / Ethereum / Solana | zuletzt gemessene Vola, Prognose 1 und 5 Tage voraus mit 90-%-Bereich, Verlauf der letzten 90 Tage; Schlusskurs und Kursspanne mit Kursverlauf; Vergleich aller Modelle; darunter separat «Neu, nur live: t0 und Timestral» |
| Logbuch | alle gespeicherten Vola-Prognosen und Kursspannen; sobald der Zieltag vorbei ist, mit tatsächlichem Wert, QLIKE und Trefferquote im 90-%-Bereich; für t0, Timestral und die zweite Kombination ein Vergleich auf denselben Tagen |
| Methode | kurze Erklärung, die statischen Gewichte und die Startgewichte der zweiten Kombination |

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
- Die statischen Gewichte kommen nur aus train.
- Das Logbuch wertet nur Zieltage **nach** der Testperiode aus (ab 01.09.2026). Die Testperiode wird damit nicht informell geöffnet.

### Das Logbuch als saubere Evaluation

Jede Prognose wird gespeichert, **bevor** der Zieltag eintritt. Kontamination ist damit ausgeschlossen, auch für Modelle, die nach der Testperiode erschienen sind (z.B. t0, Timestral). Nach einigen Monaten regelmässiger Klicks entsteht so eine echte Out-of-Sample-Periode. Ein Klick pro Tag genügt. Mehrere Klicks am selben Datenstand überschreiben sich, statt doppelt zu zählen.

### Neue Modelle: t0 und Timestral (nur live, `03_t0_timestral.ipynb`)

**t0-gamma** (The Forecasting Company, `theforecastingcompany/t0-gamma`) und **Timestral-209M** (Mistral AI, `mistralai/Timestral-209M`) sind im September bzw. Oktober 2026 erschienen, also nach train, val und test. Einen Trainings-Cutoff nennen beide nicht. Sie könnten unsere Perioden im Pretraining gesehen haben. Deshalb gilt:

- **Separat:** Sie laufen in der App nur mit, als eigene Modelle und in einer **zweiten Kombination** («Kombination neu»). Die Hauptprognose (Chronos-2, TimesFM-2.5, HAR, GARCH mit statischen Gewichten) und die Kursspanne verwenden sie nicht. Geprüft: Die Hauptprognose ist mit und ohne die neuen Modelle identisch.
- **Keine gelernten Gewichte:** Aus der Vergangenheit lernen wir für sie nichts, weil diese Daten möglicherweise kontaminiert sind.
- **Startgewichte der zweiten Kombination** (gesetzt, nicht gelernt): Die Foundation Models machen fast dieselben Fehler (Fehlerkorrelation 0.98). Deshalb wird ihr Gesamtanteil aus den statischen Gewichten gleichmässig auf alle vier verteilt, HAR und GARCH behalten ihr Gewicht: bei h=1 je ca. 0.23 für Chronos, TimesFM, t0 und Timestral, 0.00 HAR, 0.07 GARCH; bei h=5 je ca. 0.19, 0.03 HAR, 0.20 GARCH. Fehlt ein Foundation Model, wird sein Anteil auf die übrigen verteilt.
- **Saubere Auswertung nur im Logbuch:** Das Logbuch vergleicht die zweite Kombination und die neuen Modelle mit der Hauptprognose auf **denselben Tagen**, weil sie erst ab ihrem Start mitlaufen.
- **Setup wie bei Chronos und TimesFM:** log-RV, Kontext 1024 Tage, 21 Quantile (beide liefern genau dieses Raster), Mittelwert durch Integration der Quantile. t0 mit `symmetric_averaging=True` und `trim_leading_missing=True` (vom Hersteller empfohlen), Timestral mit `symmetric_averaging=True` (Default). `clip_nonnegative` ist bei beiden aus, weil die log-RV negativ ist.
- **Optional:** Sind die Pakete nicht installiert, läuft die App wie bisher und zeigt einen Hinweis.

**Inbetriebnahme:** Pakete installieren (Abschnitt 3), `03_t0_timestral.ipynb` aus `src/` ausführen. Es lädt die Gewichte von Hugging Face (zusammen ca. 2 GB), prüft Plausibilität und Laufzeit und rechnet die Prognose für den aktuellen Datenstand der App. Danach in der App neu berechnen: Die Werte für t0 und Timestral sollten mit dem Notebook übereinstimmen.

Die Historie (Abschnitt 5 im Notebook, `RUN_HISTORY = True`) ist optional und gilt als möglicherweise kontaminiert. Sie schreibt `results/<COIN>__t0-gamma.parquet` und `results/<COIN>__Timestral-209M.parquet` mit `split = "kontaminiert?"`. Weder `01_kombination.ipynb` noch die App verwenden diese Dateien, solange die Modelle nicht in `POOL` stehen.

---

## 14. Konfigurationsparameter

Alle Einstellungen stehen in der ersten Code-Zelle des jeweiligen Notebooks. In 02–05 ist `DATASET` die wichtigste.

### `01_daten.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `SOURCE` | `"binance"` | `"binance"` oder `"synthetic"` |
| `SYMBOLS` | BTCUSDT, ETHUSDT, SOLUSDT | Coins |
| `KLINE_INTERVAL` | 5m | Kerzenlänge für die RV |
| `FIRST_MONTH` / `LAST_MONTH` | 2020-01 / 2026-08 | Datenzeitraum |
| `MIN_BARS_PER_DAY` | 270 | Mindestanzahl Bars (von 288), sonst wird der Tag verworfen |
| `TEST_START` | 2026-03-01 | Zieldaten ab hier sind gesperrt |
| `SCENARIO` | `"regime_signal"` | synthetisches Szenario: `null`, `regime_signal`, `regime_nosignal` (Abschnitt 4) |
| `N_SERIES` / `N_DAYS` / `SEED` | 3 / 2500 / 0 | Anzahl und Länge der synthetischen Reihen, Zufallsseed |

### `02_experten.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `MODELS` | RW, EWMA, HAR, GARCH, Chronos-2-full, TimesFM-2.5 | Experten, deren Prognosen sichergestellt werden |
| `SKIP_MISSING_PACKAGES` | True | Foundation Model nicht installiert → überspringen |
| `HORIZONS` | [1, 5] | Prognosehorizonte in Tagen |
| `MIN_HISTORY` | 500 | Tage Historie vor dem ersten Ursprung |
| `EST_WINDOW` | 1000 | rollierendes Schätzfenster für RW, EWMA, HAR, GARCH |
| `GARCH_REFIT_EVERY` | 5 | GARCH alle 5 Ursprünge neu schätzen |
| `CONTEXT` / `FM_BATCH` | 1024 / 128 | Kontext und Batchgrösse der Foundation Models |

### `03_features.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `POOL` | Chronos-2-full, TimesFM-2.5, HAR, GARCH (synthetisch: HAR, GARCH, RW) | Experten, die kombiniert werden |
| `GROUPS` | market, disagree, recent, calendar, exo | Feature-Gruppen (+ `series` für einen Bias pro Serie) |
| `HALFLIVES` | [10, 60] | Halbwertszeiten der `recent`-Features |
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
| `LAMBDAS` | 1e-4 … 1 | Kandidaten für die L2-Regularisierung |
| `MLP` / `MLP_HIDDEN` / `MLP_SEEDS` | True / 16 / 3 | MLP-Gate an/aus, Neuronen, Anzahl Seeds |
| `ABLATIONS` | [recent], [market], [exo] | lineare Gates mit nur einer Feature-Gruppe |
| `LEGACY` | False | True: Methoden genau wie im alten Notebook |

### `05_evaluation.ipynb`

| Parameter | Wert | Bedeutung |
|---|---|---|
| `RUN` | `<DATASET>_walkforward` | welcher Lauf aus 04 |
| `BASELINE` | RW | Referenz für die relativen Metriken |
| `TARGET` | `"y"` | `"true_var"`: gegen die wahre Varianz auswerten (nur synthetisch) |

---

## 15. Stand, Einschränkungen und nächste Schritte

Details und Grafiken: `reports/zwischenstand.html`.

### Bisherige Ergebnisse

**Neue Pipeline, Walk-Forward auf echten Daten** (`binance_walkforward`: monatliches Neutraining, ausgewertet Nov. 2025 – Feb. 2026, λ nur mit Daten vor dem jeweiligen Monat gewählt, 16 Features):

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

**Mit 4 Experten, fester Split (Details in Abschnitt 12):**

- **Statische Gewichte sind mindestens so gut wie das beste Einzelmodell:** im Durchschnitt ca. 2–4 % tieferer QLIKE, aber nicht bei jedem Coin und kaum signifikant (siehe Abschnitt 12).
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

1. ~~TimesFM-2.5 als Experten hinzufügen~~ (erledigt, siehe Abschnitt 12).
2. ~~Walk-Forward statt eines festen Splits~~ (erledigt, `04_training`). Offen: eine zweite, längere Auswertungsperiode mit einem Pool ohne Kontamination (ältere Foundation Models wie Chronos-Bolt oder TimesFM 1.0, Release-Daten prüfen), damit die saubere Periode früher beginnt.
3. ~~Synthetische Daten zum Testen der Methodik~~ (erledigt, `01_daten`). Offen: `null` und `regime_nosignal` laufen lassen, mehrere Seeds.
4. **Experten und Features mit anderer Information:** Weitere Modelle, die nur die vergangene RV sehen (Toto, Moirai), bringen vermutlich wenig neue Diversität. Vielversprechender sind zusätzliche Informationen, z.B. implizite Volatilität (Deribit DVOL für BTC und ETH), Handelsvolumen oder Funding-Rates. Chronos-2 kann solche Zusatzreihen (Kovariaten) direkt verarbeiten. Als Gate-Features lassen sie sich als Spalten `exo_*` in `01_daten` anhängen.
5. **Mehr Serien und Domänen:** weitere Coins von Binance (geringer Aufwand), danach Aktien, Indizes, FX, öffentliche Benchmarks.
6. **Längere saubere Evaluation:** Mit jedem Monat neuer Daten wird die Walk-Forward-Auswertung aussagekräftiger.
7. **Zeitkomponente:** ~~Wochenende als Feature~~ und ~~regelmässiges Nachtrainieren~~ (erledigt). Monats-Saisonalität eher nicht (zu wenige Jahre, Overfitting-Gefahr).
8. **Gate verfeinern:** Sparse Top-k-Gating (Shazeer et al. 2017), ein Gate über alle Horizonte mit h als Feature, Gewichte pro Quantil.
9. **Testperiode** erst ganz am Ende öffnen (`ALLOW_TEST = True` in `04_training`, `EVAL_END = None`).

### Offene technische Punkte

- `requirements.txt` vervollständigen (`psutil`, `timesfm`, `uni2ts`, `gluonts`). `arch`, `scipy`, `requests`, `tqdm` und `jupyter` sind ergänzt.
- ~~Dokumentieren, welche Chronos-2-Variante die gespeicherten Prognosen erzeugt hat~~ (erledigt: `chronos-2-small`, siehe `results/chronos_check.json`).
- ~~`CHRONOS_MODEL` fest auf `amazon/chronos-2` setzen~~ (erledigt: `02_experten` unterscheidet `Chronos-2-full` und `Chronos-2` explizit).
- Modellversionen festhalten: Chronos-2, TimesFM-2.5 und Moirai-2.0 sind neuere Generationen als die in den Originalpapern evaluierten.

---

## 16. Glossar

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
- Ansari et al.: Chronos (Amazon), Das et al.: TimesFM (Google), Woo et al.: Moirai (Salesforce)
