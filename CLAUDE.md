# CLAUDE.md

Kontext für Claude Code in diesem Repo. Ausführliche Doku: `README.md`. Stand: 2026-11-26.

## Projekt

Bachelorprojekt HSG: "Learning to Forecast Time Series with Multiple Experts". Mehrere Experten (Foundation Models + klassische Modelle) prognostizieren eine Zeitreihe; ein Gate `w = softmax(b + W x)` mischt Mittelwert und 21 Quantile. Frage: Wann schlägt eine gelernte Kombination das beste Einzelmodell bzw. gleiche Gewichte?

Kommunikation mit dem User auf **Deutsch**. Code-Kommentare und Notebooks sind deutsch.

### Zusammenarbeit
- Team von Studierenden; mehrere Personen pushen parallel auf `main` und rechnen FM-Läufe auf Kaggle/Colab bzw. eigenen Rechnern. Neue Notebooks/Ergebnisse kommen oft per Pull, danach fragt der User "was hat sich geändert / ist das korrekt gelaufen?".
- Der User will **Intuition und Einordnung**, nicht nur Tabellen: Warum klappt etwas (Strom) und warum nicht (Volatilität), was ist der Mehrwert für die Arbeit. Ergebnisse immer in Bezug auf die Forschungsfrage und das Betreuer-Feedback setzen.
- Texte von aussen (Betreuer, andere KI) werden oft als Paste geschickt ("verstehst du das & was soll ich tun?"): erst am echten Code/Daten prüfen, dann umsetzen.
- `README.md` ist die ausführliche Doku (17 Abschnitte, Ergebnisse in Abschnitt 16, Domänen 06–09 in Abschnitt 11). Bei neuen Ergebnissen oder Notebooks README und diese Datei mit aktualisieren.

### Betreuer-Feedback (Leitlinie für alles Weitere)
- neuer Datensatz mit allgemeinen Features
- weitere Foundation Models
- kein Feature Engineering, nur auf den Zeitreihen selbst
- Forecast Combination Puzzle als mögliches Finding

## Kernergebnis (nach Quantil-Cap, Nov. 2026)

- **Mechanismus:** Kombinationsgewinn hängt von der Fehlerkorrelation der FMs ab. Spearman(Korrelation, ex-post-Gewinn fester Gewichte) = **−0.76** über 22 Serie×Horizont-Punkte (08).
- **Volatilität:** FM-Fehlerkorrelation 0.93–0.98, gemeinsamer Fehler 58–84 %, modellspezifischer Anteil nur 2–5 %. Ex-post-Obergrenze 0–1.7 %. Keine gelernte Kombination signifikant besser als Auswahl/Gleichgewicht → Puzzle gilt.
  - BTC/ETH/SOL (04/05, Pool Chronos+TimesFM): Gating linear vs. Gleichgewicht h=1 −0.94 % (p=0.114), h=24 +2.75 % (p=0.27). Forecast-only-Gate [disagree+recent] ≈ Gleichgewicht.
  - 08 Haupttest Gate linear vs. Auswahl (Holm): h=1 −2.4 % (p_Holm 0.148), h=24 +0.6 % (0.744).
- **Strom (07):** FM-Fehlerkorrelation 0.55–0.85. Gate linear vs. Auswahl h=1 −11.8 % (p<0.001), h=24 −2.4 % (p_Holm 0.024). Gate ≈ Statisch → Gewinn kommt vom Mischen, nicht vom Routing. Klassische Modelle (SNaive, SeasReg) bekommen ~0 Gewicht; Gleichgewicht mit ihnen im Pool explodiert (+127 %).
- **Warum:** Volatilität = strukturloses Rauschen, FMs irren gemeinsam (+ mögliche Kontamination). Strom = starke Saisonalität, verschiedene Architekturen (TiRex xLSTM, FlowState SSM, Chronos Transformer) machen verschiedene Fehler; TiRex/FlowState besser bei h=1, Chronos bei h=24.

### Weitere Zahlen (für Rückfragen)
- BTC/ETH/SOL rel. QLIKE vs. RW (h=1 / h=24): Chronos 0.731/0.769, TimesFM 0.752/0.736, Gleichgewicht 0.730/0.735, Statisch 0.732/0.734, Gating linear 0.723/0.755, MLP 0.721/0.760, [volume] 0.727/0.728. Statisch-Gewichte nach Cap h=1 41 % Chronos / 59 % TimesFM (vorher 0/100).
- 06, Pool 4 FMs, Δ QLIKE vs. bester Experte (h=1 TiRex, h=24 Chronos): Gleichgewicht +0.7/+0.4 %, Statisch −0.7/0.0 % (h=24: 100 % Chronos), Gate linear −0.2/+0.6 %. Ausreisser BNB h=24 (−4 bis −6.5 % vs. bestes FM) = Auswahlrisiko, kein Bug.
- 07 MAPE h=1: TiRex 0.77 %, FlowState ~0.78, Chronos ~0.95, SeasReg 3.4, SNaive 7; h=24 Chronos 3.30 %. Statisch-Gewichte h=1 TiRex 35 / FlowState 40 / Chronos 22 / TimesFM 3; h=24 Chronos 72 %. Oracle h=1 −53 %, feste Gewichte ex post −10.3 % ≈ Walk-Forward.
- 08: Fehlerzerlegung Strom gemeinsamer Fehler 1 % (h=1) / 5 % (h=24), modellspezifischer Anteil 20–32 %; FMs bei Volatilität absolut uneiniger (8–19 % vs. 0.7–2.5 %), aber gemeinsamer Fehler dominiert.

### Domänenwahl (Entscheidungen)
- Ziel: Domänen mit anderer FM-Fehlerstruktur, um mehr Punkte für die Korrelation-vs.-Gewinn-Analyse zu bekommen. Mehrwert von Strom ist nicht "Last ist vorhersagbar", sondern der **Kontrast** zur Volatilität.
- **Temperatur verworfen:** negative Werte → kein Log, Pipeline (Experten, Quantile, Mittelwert) rechnet auf Log-Skala; MAPE sinnlos.
- **Wind gewählt:** positiv, gleiche Quelle wie Strom (Energy-Charts, `"Wind onshore"`), Cache `data/raw/energy_charts/` geteilt. PL statt IT (wenig Onshore-Wind). sMAE statt MAPE. Daten (Juli 24–Feb 26) liegen lokal in `data/wind_1h/`: praktisch keine Stunden mit Wind ≤ 0 (nur FR 39 h Lücken = 0.3 %), Minimum 19–271 MW.
- Weitere Kandidaten, falls nötig: Gaspreise, Verkehr (positiv, stündlich).

## Struktur

- `src/01–05`: Hauptpipeline auf `DATASET = "binance_1h"` (stündliche RV aus 1-Min-Kerzen, BTC/ETH/SOL). Pool in 03: Chronos-2-full + TimesFM-2.5. 04 = Walk-Forward (monatlich, λ auf letzten 90 Tagen), schreibt `results/runs/binance_1h_walkforward/`.
- `src/06_new_models_and_TS`: `binance_1h_neu` (PAXG, EURUSDT, XRP, BNB), FMs Chronos-2-full, TimesFM-2.5, TiRex, FlowState + RW/HAR/GARCH. Gate nur `raw_0..23` (24h log-RV) + disagree. Pools "2 FMs (alt)", "4 FMs", "4 FMs + HAR/GARCH". Ergebnisse `results/forecasts/06_res.parquet`, `06_wts.parquet`.
- `src/07_electricity`: `strom_1h`, Last DE/FR/IT/ES (Energy-Charts API), Experten RW, SNaive, SeasReg + 4 FMs. Loss: quadratischer Fehler / Niveau + Pinball, Metrik MAPE. `07_res/07_wts.parquet`.
- `src/08_correlation_vs_win`: Analyse über alle Datensätze (Korrelation vs. Gewinn, OOS vs. Obergrenze, Fehlerzerlegung, Haupttest mit Holm, Statisch-Diagnose, Quantil-Check).
- `src/09_wind`: `wind_1h`, Wind onshore DE/FR/ES/PL, Metrik skalierter MAE (|y−ŷ|/E[y], nicht MAPE wegen kleiner Werte). Von Claude erstellt, **noch nie gelaufen** (braucht Kaggle-GPU, ~4–5 h). Erwartung: FM-Korrelation zwischen Strom und Volatilität.
- `src/pipeline/`: gemeinsamer Code (common, experts, features, combine, exo, live). Auch von `app/app.py` (Streamlit, jetzt ebenfalls `binance_1h`) genutzt.
- `src/archiv/`: alte Notebooks (Tagesdaten), nicht anfassen.
- Gitignored: `src/data/`, `src/results/panels/`, `src/results/forecasts/` (Expertenprognosen + 06/07_res liegen nur lokal!). `results/runs/` ist in Git.

## Wichtige Konventionen

- **Splits:** train ≤ 2025-10-31, val/Eval 2025-11-01–2026-02-28, **Test ab 2026-03-01 gesperrt** (`ALLOW_TEST = False`). Grund: FM-Releases (Chronos-2 ~Okt 25, TimesFM-2.5 ~Sep 25, Moirai-2 ~Aug 25) → Kontamination. Testperiode nie ohne ausdrückliche Anweisung öffnen.
- `common.py`: `HORIZONS=[1,24]` (Stunden), `MIN_HISTORY=24*60`, `EST_WINDOW=24*90`, `CONTEXT=1024`, `ORIGIN_START="2025-01-01"`. Änderungen invalidieren gecachte Prognosen.
- **`cap_quantiles(fc, k=8)`** in `common.py` muss beim Laden von Expertenprognosen immer angewendet werden. Grund: Chronos-2 (bis q99 = e⁴²·Median) und TiRex (bis e¹³) liefern auf Volatilität absurde Quantile; sie haben den Pinball-Term im Gate-Loss gesprengt (Chronos-Pinball im Training 1.5e12 statt ~0.7) und Chronos auf Gewicht 0 gedrückt ("Statisch-Pathologie"). k=8 trifft nur Chronos/TiRex, nicht RW/HAR/GARCH (max ~e⁷). QLIKE ändert sich durch den Cap kaum. Strom: keine Ausreißer.
- Expertenprognosen werden nie lokal neu gerechnet (kein GPU; `ONLY_CACHE` in 06/07/09). FM-Läufe passieren auf Kaggle/Colab, Ergebnisse kommen als ZIP/Pull zurück.
- `FeatureConfig.forecast_only()` = Gruppen `disagree` + `recent` (keine Marktdaten); Ablation `["disagree", "recent"]` in 04.
- Neue Notebooks für Domänen: Vorlage 07 (Daten laden → Experten mit Cache → Gate-Loss per `cb.comb_loss_torch` / `cb.qlike` überschreiben → Walk-Forward → DM-Tests → Oracle).

## Arbeitsumgebung (Windows)

- Python immer aus der venv: `.venv/Scripts/python.exe` (System-Python hat kein pyarrow). Notebooks aus `src/` starten (relative Pfade).
- Notebooks headless ausführen: venv vor den PATH setzen, sonst startet der Jupyter-Kernel System-Python:
  `export PATH="$(cd .venv/Scripts && pwd):$PATH"; cd src; python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 X.ipynb`
- Laufzeiten lokal: 03 ~15 s, 04 ~30 min, 05 ~10 s, 06 (Gates neu) ~25 min, 08 ~1 min.
- Notebooks editieren: `NotebookEdit` (braucht vorher Read) oder kleines Python-Skript mit exakten String-Ersetzungen pro Zell-ID (json, `indent=1, ensure_ascii=False`). Edit-Tool funktioniert nicht auf .ipynb.
- In PowerShell-Here-Strings keine Python-f-Strings mit `{x:.2f}`; lieber Bash-Heredoc mit `PYTHONIOENCODING=utf-8`.
- In 08 Zelle 0 wird `C` als Korrelationsmatrix überschrieben → dort `pipeline.common` nicht als `C` importieren.
- `git push` scheitert aus Claude heraus (SSH-Key mit Passphrase). User pusht selbst; vorher oft `git pull --rebase origin main` nötig (Remote wird parallel von Teammitgliedern geändert). `forecasts.parquet` im Run-Ordner ist ~60 MB (GitHub-Warnung, kein Fehler).

## Offene Punkte

- **08 Markdown-Zellen** enthalten teils noch Zahlen von vor dem Cap (z.B. "Volatility: −0.7 % / +0.8 %"); Code-Outputs sind aktuell. Bei Gelegenheit Texte angleichen.
- **GARCH** auf `binance_1h_neu` hat ebenfalls auffälligen Pinball (Mittel 27 statt ~0.7), wird von k=8 nicht erfasst; betrifft Pool "4 FMs + HAR/GARCH" (dort sind Gates und Gleichgewicht schlecht).
- **Auswahl** im Pool "4 FMs + HAR/GARCH" wählt HAR (bester Trainings-QLIKE), das im Eval +76 % schlechter ist → Regimewechsel.
- **App-Abschnitt im README** (Abschnitt 14) beschreibt Details und die Kursspannen-Kalibrierung noch aus der Tagesversion; die App selbst läuft auf `binance_1h`. Bei Gelegenheit nachführen.
- `requirements.txt` fehlen u.a. `timesfm`, `tirex-ts`, `granite-tsfm`, `psutil`.
- **09 Wind** auf Kaggle laufen lassen, Ergebnis als dritte Domäne in 08 (`SETS`, `OOS`) aufnehmen.
- Testperiode erst ganz am Schluss öffnen.
- Mögliche Erweiterungen: A-priori-Vorhersage des Kombinationsgewinns aus Trainings-Korrelation; Kontaminations-Hypothese testen; Moirai-2 als weiteres FM.
