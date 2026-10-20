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
12. [Prototyp](#12-prototyp-prototypepy)
13. [Konfigurationsparameter](#13-konfigurationsparameter)
14. [Stand, Einschränkungen und nächste Schritte](#14-stand-einschränkungen-und-nächste-schritte)
15. [Glossar](#15-glossar)

---

## 1. Überblick

Das Projekt läuft in zwei Stufen:

```
STUFE 1  (volatilitaet_chronos2_har_garch.ipynb)
  Binance 5-Min-Kerzen ──► tägliche RV pro Coin ──► Experten prognostizieren RV für t+1 und t+5
                                                     (Chronos-2, HAR, GARCH, Random Walk)
                                                     ──► results/<COIN>__<MODELL>.parquet
                                                         (Mittelwert + 21 Quantile pro Tag)

STUFE 2  (01_kombination.ipynb)
  gespeicherte Prognosen + 12 Features ──► Gate ──► Gewichte pro Experte und Tag
                                                ──► gewichteter Durchschnitt der Prognosen
                                                ──► Vergleich mit der tatsächlichen RV
```

- **Stufe 1** ist teuer (Modelle laufen über Tausende Tage) und wird einmal gerechnet. Die Ergebnisse werden gecacht.
- **Stufe 2** ist billig und arbeitet nur auf den gespeicherten Prognosen. Dort werden die Kombinationsstrategien trainiert und verglichen.

---

## 2. Projektstruktur

```
.
├── README.md
├── requirements.txt
├── prognose.png                  Plot aus prototype.py
├── reports/
│   └── zwischenstand.html        Zwischenbericht mit Grafiken und Ergebnissen
└── src/
    ├── prototype.py              erster Prototyp (S&P 500, Chronos-2 + Toto)
    ├── 00_resource_check.ipynb   laufen Chronos-2, TimesFM und Moirai lokal? RAM, Latenz
    ├── volatilitaet_chronos2_har_garch.ipynb   Stufe 1: Daten, RV, Experten, Forecasts
    ├── 01_kombination.ipynb      Stufe 2: Kombination der Experten, Gate, Evaluation
    ├── data/                     (nicht in Git, siehe .gitignore)
    │   ├── raw/<COIN>/<COIN>-5m-YYYY-MM.zip   Binance-Monatsdateien
    │   └── <COIN>_daily.parquet  tägliche RV, Rendite, Anzahl Bars
    └── results/
        ├── <COIN>__<MODELL>.parquet   Prognosen jedes Experten
        ├── runtimes.csv               Laufzeit pro Coin und Experte
        └── resource_check_*.csv/json  Ergebnisse des Resource Checks
```

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
> pip install arch scipy requests tqdm psutil
> # nur für den Resource Check:
> pip install timesfm uni2ts gluonts
> ```

### Reihenfolge

Die Notebooks verwenden relative Pfade (`data/`, `results/`) und müssen deshalb **aus dem Ordner `src/`** gestartet werden.

| Schritt | Datei | was passiert | Dauer |
|---|---|---|---|
| (optional) | `00_resource_check.ipynb` | prüft, ob die Foundation Models lokal laufen | Minuten |
| 1 | `volatilitaet_chronos2_har_garch.ipynb` | lädt die Binance-Daten herunter, berechnet die RV, lässt alle Experten laufen, speichert die Prognosen | einmalig länger, danach Cache |
| 2 | `01_kombination.ipynb` | trainiert und evaluiert die Kombinationen | Sekunden bis Minuten |

**Caching:** Existiert eine Datei in `data/` oder `results/` schon, wird sie nur geladen. Um einen Experten neu zu rechnen, die entsprechende Parquet-Datei löschen oder `run_expert(..., overwrite=True)` verwenden.

Der Ordner `src/data/` ist nicht in Git. Auf einem neuen Rechner lädt Notebook 1 die Daten automatisch von `data.binance.vision` (kostenlos, ohne API-Key).

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

Code (`daily_rv` in Notebook 1):

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
| `split` | train / val / test (in Notebook 2 neu berechnet) |

---

## 6. Die Experten

Alle Experten haben dieselbe Schnittstelle: eine Klasse mit `name` und `forecast(s, origins)`, die `mean` mit Form `(n_origins, n_horizons)` und `q` mit Form `(n_origins, n_horizons, 21)` zurückgibt.

### Gemeinsames Prinzip der ökonometrischen Modelle

RW, HAR und GARCH prognostizieren einen **Punktwert in Logs**. Die Verteilung entsteht aus den **empirischen Residuen** im Schätzfenster (letzte 1000 Tage):

- Quantile: $\hat q_\tau = \exp(\text{Log-Prognose} + \text{Quantil}_\tau(\text{Residuen}))$
- Mittelwert: Smearing-Schätzer (Duan 1983), $\hat\mu = \exp(\text{Log-Prognose}) \cdot \overline{\exp(\text{Residuen})}$

### Übersicht

| Experte | Input | Idee | Details |
|---|---|---|---|
| **RW** (Random Walk) | log-RV | $RV_{t+h} = RV_t$, "morgen wie heute" | Streuung aus historischen h-Tages-Änderungen der log-RV. **Nur Baseline**, nicht im Pool. |
| **HAR** (log-HAR, Corsi 2009) | log-RV | $\ln RV_{t+h} = \beta_0 + \beta_1 \ln RV_t + \beta_2 \ln RV^{(5)}_t + \beta_3 \ln RV^{(22)}_t$ | Tag, Woche, Monat. OLS im rollierenden 1000-Tage-Fenster, direkte Prognose pro Horizont. Nur Zeilen, deren Ziel bis *t* bekannt ist. |
| **GARCH(1,1)-t** | Tagesrenditen | Varianz reagiert auf vergangene Schocks und vergangene Varianz, t-verteilte Innovationen | GARCH modelliert die Varianz der Tagesrendite, nicht die RV. Niveau und Streuung werden deshalb über die Residuen $\ln(RV_u / \sigma_u^2)$ auf die RV kalibriert. Neuschätzung alle 5 Ursprünge. |
| **Chronos-2** (Amazon) | log-RV | vortrainiertes Foundation Model, **zero-shot** (kein Training auf diesen Daten) | Kontext: letzte 1024 Tage. Gibt die 21 Quantile direkt aus (in Logs, dann `exp`). Mittelwert durch Integration der Quantile. `cross_learning=False`, damit sich Kontexte im Batch nicht gegenseitig Information geben. |

> Hinweis zu Chronos-2: Auf GPU wird `amazon/chronos-2` geladen, auf CPU automatisch das kleinere `autogluon/chronos-2-small`. Für die Dokumentation festhalten, mit welchem Modell die gespeicherten Prognosen erzeugt wurden.

### Laufzeiten (`results/runtimes.csv`, Sekunden pro Coin)

| | RW | HAR | GARCH | Chronos-2 |
|---|---|---|---|---|
| BTC | 0.4 | 1.4 | 9.9 | 28.9 |
| ETH | 0.5 | 0.8 | 6.7 | 32.7 |
| SOL | 0.2 | 0.5 | 5.7 | 28.6 |

### Neuen Experten hinzufügen

1. In Notebook 1 eine Klasse `class NeuesModell(Expert)` mit `name` und `forecast(s, origins)` schreiben.
2. `run_expert(NeuesModell(), s)` für alle Reihen aufrufen.
3. In Notebook 2 den Namen in `POOL` eintragen. Alle Auswertungen nehmen ihn automatisch mit.

---

## 7. Splits: train, val, test

Die Aufteilung richtet sich nach dem **Zieldatum**, damit kein Trainingslabel Information aus der Validierungsperiode enthält.

| Split | Periode (Notebook 2) | Zweck |
|---|---|---|
| **train** | bis 31.10.2025 | Gewichte und Gate-Parameter lernen |
| **val** | 01.11.2025 – 28.02.2026 | Hyperparameter (λ) wählen, Ergebnisse anschauen |
| **test** | 01.03.2026 – 08/2026 | **gesperrt** (`ALLOW_TEST = False`), erst für die finale Auswertung |

**Warum diese Grenzen?** Foundation Models wurden auf öffentlichen Daten vortrainiert, möglicherweise auch auf Krypto-Kursen. Liegt eine Evaluationsperiode vor dem Trainings-Cutoff eines Modells, könnte es die Werte schon gesehen haben (**Kontamination**), und der Vergleich wäre nicht mehr zero-shot. Der Cutoff ist meist nicht dokumentiert, das Release-Datum ist eine obere Schranke (Chronos-2 ca. Oktober 2025). Deshalb liegen val und test danach.

> Notebook 1 verwendet noch die **alten** Grenzen (train ≤ 2023, val = 2024). Notebook 2 überschreibt die gespeicherte `split`-Spalte mit den neuen Grenzen. Für die Kombinationen zählen nur die neuen.

---

## 8. Kombination: das Gate

### Grundidee

Das Gate ist ein kleines neuronales Netz. Es macht **keine eigene RV-Prognose**, sondern berechnet pro Coin und Tag **3 Gewichte** $w = (w_\text{Chronos}, w_\text{HAR}, w_\text{GARCH})$ mit $w_m \ge 0$ und $\sum w_m = 1$. Damit werden **Mittelwert und alle 21 Quantile** der Experten gemischt:

$$\hat\mu = \sum_m w_m \hat\mu_m, \qquad \hat q_\tau = \sum_m w_m \hat q_{m,\tau}$$

Die Kombination ist ein eigener Wert, nicht einfach das beste Modell. Sie liegt aber immer zwischen dem tiefsten und dem höchsten Expertenwert.

### Die 12 Features (Input des Gates)

Alle Features sind am Ursprungstag *t* bekannt, es gibt also kein Leakage. Sie werden pro Coin und Tag berechnet und mit Mittelwert und Standardabweichung aus **train** standardisiert (0 = durchschnittlich, +1 = eine Standardabweichung darüber). Fehlende Werte (Aufwärmphase) werden auf 0 gesetzt.

Notation: $RV^{(k)}$ = gleitender Mittelwert der RV über *k* Tage.

**A. Marktzustand (6 Features)**

| Feature | Formel | misst | Interpretation |
|---|---|---|---|
| `logrv_dev` | $\ln RV_t - \ln RV^{(22)}_t$ | Abweichung der heutigen Vola vom Monatsschnitt | > 0: heute unruhiger als üblich |
| `trend` | $\ln RV^{(5)}_t - \ln RV^{(22)}_t$ | Richtung der Vola | > 0: Vola steigt |
| `level` | $\ln RV^{(22)}_t$ | allgemeines Vola-Niveau | hoch: volatile Phase bzw. volatiler Coin |
| `volvol` | Std.-Abw. von $\ln RV$ über 22 Tage | Volatilität der Volatilität | hoch: RV springt stark, schwer vorherzusagen |
| `ret1` | $ret_t / \sqrt{RV^{(22)}_t}$ | gestrige Rendite relativ zur üblichen Schwankung | gross (±): Schock gestern |
| `ret5` | $\sum_{5} ret / \sqrt{5\,RV^{(22)}_t}$ | Rendite der letzten 5 Tage, standardisiert | anhaltende Bewegung |

**B. Uneinigkeit (3 Features, eines pro Experte)**

| Feature | Formel | misst |
|---|---|---|
| `disagree_<Modell>` | $\ln\hat\mu_m - \frac{1}{M}\sum_j \ln\hat\mu_j$ | wie stark die heutige Prognose dieses Experten vom Durchschnitt abweicht (> 0: prognostiziert mehr Vola als die anderen) |

**C. Jüngste Fehler (3 Features, eines pro Experte)**

| Feature | Formel | misst |
|---|---|---|
| `recent_<Modell>` | $\ln L_m - \frac{1}{M}\sum_j \ln L_j$ | wie gut der Experte **zuletzt** war, relativ zu den anderen (< 0: kleinerer Fehler) |

$L_m$ ist der exponentiell gewichtete QLIKE-Fehler (Halbwertszeit 10 Prognosen, mind. 5 Werte), um *h* verschoben, damit nur Prognosen zählen, deren wahre RV am Tag *t* schon bekannt ist.

**Keine Features:** `scale` = $RV^{(22)}_t$ dient nur zur Normierung des Pinball-Loss. Es gibt **kein Kalender-Feature** (Wochentag, Monat). Die zeitliche Information steckt in den rollenden Fenstern.

### Architektur

**Gating linear** (39 Parameter):

$$z = b + W x_t, \qquad w = \mathrm{softmax}(z), \quad w_m = \frac{e^{z_m}}{\sum_j e^{z_j}}$$

- $x_t$: die 12 Features
- $W$ (3 × 12): wie stark jedes Feature den Score jedes Experten erhöht oder senkt
- $b$ (3): wie stark man jedem Experten **grundsätzlich** vertraut
- Softmax: macht aus beliebigen Scores positive Gewichte mit Summe 1

**Gating MLP** (259 Parameter): 12 Features → 16 Neuronen (tanh) → 3 Scores, plus $b$. Kann Kombinationen lernen, z.B. "nur wenn die Vola hoch **und** steigend ist".

**Initialisierung:** $b = 0$ und letzte Schicht $= 0$. Jedes Gate startet also bei gleichen Gewichten (⅓, ⅓, ⅓).

Es gibt **ein Gate pro Horizont**, gemeinsam für alle drei Coins. Die Coins teilen die Parameter, jede Zeile sieht aber nur die Features ihres eigenen Coins.

### Rechenbeispiel (ein Tag, erfundene Werte)

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

| Methode | Gewichte | Parameter | gelernt auf |
|---|---|---|---|
| **Gleichgewicht** | immer ⅓ / ⅓ / ⅓ | 0 | – |
| **Auswahl** | 1 für den besten Experten auf train, sonst 0 | 0 | train |
| **Statisch** | $\mathrm{softmax}(b)$, gleich für alle Tage | 3 | train |
| **Gating linear** | $\mathrm{softmax}(b + Wx_t)$, jeden Tag anders | 39 | train, λ auf val |
| **Gating MLP** | wie linear, mit versteckter Schicht | 259 | train, λ auf val |

"Statisch" ist ein Gate ohne Features. Ein Gate mit sehr grossem λ wird zu "Statisch".

---

## 9. Training des Gates

### Loss

$$\mathcal{L} = \overline{\text{QLIKE}(y, \hat\mu)} + \alpha \cdot \overline{\text{Pinball}(y, \hat q) / s_t} + \lambda \sum W^2$$

- **QLIKE** bewertet den kombinierten Mittelwert (siehe Abschnitt 10).
- **Pinball** bewertet die kombinierten Quantile.
- **$s_t$** = RV-Niveau der letzten 22 Tage: Ohne diese Normierung würden volatile Phasen und Coins das Training dominieren.
- **$\alpha$** = 1: Gewicht des Quantil-Teils.
- **$\lambda \sum W^2$**: L2-Strafe nur auf die Feature-Gewichte, nicht auf $b$. Sie hält das Gate nahe an statischen Gewichten, solange die Features den Fehler nicht deutlich senken.

### Ablauf

1. Start bei W = 0, b = 0 → gleiche Gewichte.
2. **Alle train-Zeilen auf einmal** (alle Coins, alle Tage) durch das Gate rechnen → Gewichte → kombinierte Prognose → Loss.
3. **Backpropagation**: Für jeden Parameter wird berechnet, in welche Richtung der Loss sinkt.
   *Beispiel:* Lag GARCH an Tagen mit hohem `logrv_dev` meist näher an der echten RV, steigt W[GARCH, logrv_dev]. Ist GARCH im Schnitt schlechter, sinkt b[GARCH].
4. **L-BFGS** passt die Parameter an (full-batch, bis 500 Iterationen, Strong-Wolfe-Linesearch). Schritte 2–4 wiederholen, bis der Loss nicht mehr sinkt.
5. **λ-Auswahl:** Training für jedes λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}. Behalten wird das λ mit dem kleinsten Loss auf **val**.

Die Experten werden dabei **nicht** verändert, nur die Parameter des Gates. Das Gate wird einmal trainiert und dann eingefroren, es gibt kein laufendes Nachtrainieren.

---

## 10. Evaluation und Metriken

Alle Modelle (Experten als One-Hot-Gewichte und Kombinationen) werden auf **denselben Zeilen** der val-Periode ausgewertet.

| Metrik | Formel / Definition | misst | gut ist |
|---|---|---|---|
| **QLIKE** | $\frac{y}{\hat\mu} - \ln\frac{y}{\hat\mu} - 1$ | Fehler der Punktprognose. Standard für Varianzprognosen, robust gegenüber verrauschten Vola-Proxies (Patton 2011). Bestraft zu tiefe Prognosen stärker. | 0 = perfekt |
| **Pinball** | $\frac{1}{21}\sum_\tau \max\big(\tau(y-\hat q_\tau),\ (\tau-1)(y-\hat q_\tau)\big)$ | Qualität der ganzen Verteilung: kalibriert und möglichst eng | klein |
| **Cov90** | Anteil Tage mit $q_{0.05} \le y \le q_{0.95}$ | Kalibrierung des 90-%-Intervalls | ≈ 0.90 |
| **rel. QLIKE / rel. Pinball** | Metrik des Modells / Metrik des Random Walk, pro Coin und Horizont, dann geometrisches Mittel über die Coins | Vergleich mit der einfachsten Baseline | < 1 |
| **Diebold-Mariano** | Test auf Loss-Differenzen, Newey-West-Varianz | ist der Unterschied zwischen zwei Modellen signifikant? | DM < 0: erstes Modell besser; p < 0.05 |

### Oracle-Analyse (Notebook 1)

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


## 13. Konfigurationsparameter

### Notebook 1 (`volatilitaet_chronos2_har_garch.ipynb`)

| Parameter | Wert | Bedeutung |
|---|---|---|
| `SYMBOLS` | BTCUSDT, ETHUSDT, SOLUSDT | Coins |
| `KLINE_INTERVAL` | 5m | Kerzenlänge für die RV |
| `FIRST_MONTH` / `LAST_MONTH` | 2020-01 / 2026-08 | Datenzeitraum |
| `MIN_BARS_PER_DAY` | 270 | Mindestanzahl Bars (von 288), sonst wird der Tag verworfen |
| `HORIZONS` | [1, 5] | Prognosehorizonte in Tagen |
| `QUANTILES` | 0.01, 0.05, 0.10, …, 0.95, 0.99 | 21 Quantilniveaus |
| `MIN_HISTORY` | 500 | Tage Historie vor dem ersten Ursprung |
| `EST_WINDOW` | 1000 | rollierendes Schätzfenster für RW, HAR, GARCH |
| `GARCH_REFIT_EVERY` | 5 | GARCH alle 5 Ursprünge neu schätzen |
| `TRAIN_END` / `VAL_END` | 2023-12-31 / 2024-12-31 | alte Splits (in Notebook 2 ersetzt) |
| `CHRONOS_MODEL` | `amazon/chronos-2` (GPU) / `autogluon/chronos-2-small` (CPU) | Chronos-Variante |
| `CHRONOS_CONTEXT` | 1024 | Tage Kontext für Chronos |
| `CHRONOS_BATCH` | 128 | Batchgrösse |

### Notebook 2 (`01_kombination.ipynb`)

| Parameter | Wert | Bedeutung |
|---|---|---|
| `POOL` | Chronos-2, HAR, GARCH | Experten, die kombiniert werden |
| `BASELINE` | RW | Referenz für relative Metriken |
| `TRAIN_END` | 2025-10-31 | Ende train (nach Release der Foundation Models) |
| `VAL_END` | 2026-02-28 | Ende val |
| `ALLOW_TEST` | False | Sperre der Testperiode |
| `ALPHA` | 1.0 | Gewicht des Pinball-Terms im Loss |
| `PERF_HALFLIFE` | 10 | Halbwertszeit für die `recent_*`-Features |
| `LAMBDAS` | 1e-4 … 1 | Kandidaten für die L2-Regularisierung |
| `MLP_HIDDEN` | 16 | Neuronen in der versteckten Schicht des MLP-Gates |

---

## 14. Stand, Einschränkungen und nächste Schritte

Details und Grafiken: `reports/zwischenstand.html`.

### Bisherige Ergebnisse

- Das lineare Gate ist auf val kaum besser als statische Gewichte. Der Unterschied liegt im Rauschen (das Bootstrap-Intervall enthält 0).
- Das MLP wählt auf val das stärkste λ und wird damit faktisch zu statischen Gewichten.
- Kleines λ senkt nur den train-Fehler, nicht den val-Fehler (Overfitting).
- Das Gate verschiebt Gewicht vor allem zwischen Chronos-2 und HAR (h = 1). Bei h = 5 bleibt es fast statisch.
- **Grund:** Die Experten liegen meist in dieselbe Richtung daneben, besonders HAR und GARCH, die auf denselben vergangenen Werten aufbauen. Dazu kommt eine kurze val-Periode (4 Monate).
- **Kontamination:** Chronos-2 ist 2021–2023 kaum besser als HAR, der grosse Vorsprung kommt erst 2024–2025, und im ersten Quartal nach dem Release ist es schlechter als HAR. Das passt zur Hypothese, dass Chronos die Daten im Pretraining gesehen hat, ist aber kein Beweis (es könnte auch ein Regimewechsel sein).

### Nächste Schritte

1. **Mehr und unterschiedlichere Experten:** TimesFM-2.5 (läuft lokal), danach Toto und Moirai (nach Fix).
2. **Mehr Serien und Domänen:** Aktien, Indizes, FX, öffentliche Benchmarks.
3. **Längere saubere Evaluation:** Mit jedem Monat neuer Daten wird val aussagekräftiger.
4. **Zeitkomponente:**
   - Wochentag des Zieldatums als Feature (Krypto-Vola ist am Wochenende tiefer)
   - regelmässiges Nachtrainieren des Gates (rollierendes oder expandierendes Fenster)
   - Monats-Saisonalität eher nicht (zu wenige Jahre, Overfitting-Gefahr)
5. **Gate verfeinern:** Sparse Top-k-Gating (Shazeer et al. 2017), ein Gate über alle Horizonte mit h als Feature, Gewichte pro Quantil.
6. **Testperiode** erst ganz am Ende öffnen (`ALLOW_TEST = True`).

### Offene technische Punkte

- `requirements.txt` vervollständigen (`arch`, `scipy`, `requests`, `tqdm`, `psutil`, `timesfm`, `uni2ts`, `gluonts`).
- Dokumentieren, welche Chronos-2-Variante die gespeicherten Prognosen erzeugt hat.
- Modellversionen festhalten: Chronos-2, TimesFM-2.5 und Moirai-2.0 sind neuere Generationen als die in den Originalpapern evaluierten.

---

## 15. Glossar

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

### Literatur

- Corsi (2009): A Simple Approximate Long-Memory Model of Realized Volatility (HAR)
- Bollerslev (1986): Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
- Patton (2011): Volatility Forecast Comparison Using Imperfect Volatility Proxies (QLIKE)
- Duan (1983): Smearing Estimate (Rücktransformation aus Logs)
- Diebold & Mariano (1995): Comparing Predictive Accuracy
- Shazeer et al. (2017): Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- Ansari et al.: Chronos (Amazon), Das et al.: TimesFM (Google), Woo et al.: Moirai (Salesforce)
