"""Exogene Features für Stundendaten (Datensatz binance_1h).

Zwei Arten:

1. Spalten `exo_*` in den Stundendaten (01_daten ruft `add_exo` vor `save_dataset` auf). Sie gehören zum
   Ursprung t und dürfen nur Information bis zum Ende der Stunde t nutzen (Zeile t = Stunde [t, t+1h)).
   features.py sortiert sie nach Präfix in Gruppen:
     exo_vol_*   -> "volume"   Volumen, Trades, Taker-Anteil aus den 1-Min-Kerzen von Binance
     exo_dvol_*  -> "dvol"     Deribit DVOL (implizite Vola BTC/ETH), Varianz-Risikoprämie
     exo_mkt_*   -> "markets"  VIX und S&P-500-Futures (Yahoo Finance)
2. Kalender-Features (`calendar_features`). Sie hängen am Zielzeitpunkt, nicht am Ursprung, und werden
   in features.build_panel pro Horizont berechnet (wie `weekend`). Das ist kein Leakage: Uhrzeit,
   Börsenzeiten und Termine von FOMC und CPI stehen lange im Voraus fest.

rv und ret werden hier nie verändert, gecachte Expertenprognosen bleiben also gültig.
Rohdaten werden gecacht (data/raw/...). Yahoo liefert Stundendaten nur für die letzten ~730 Tage:
der Cache ist darum die einzige Quelle für ältere Werte und sollte nicht gelöscht werden.
"""
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365 * 24
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]
VOLUME_VERSION = 1          # erhöhen, wenn sich die Berechnung ändert (neuer Cache-Name)


def _step(index):
    idx = pd.DatetimeIndex(index)
    return pd.Series(idx).diff().median() if len(idx) > 1 else pd.Timedelta(hours=1)


def _asof(row_end, avail_index, values, tolerance):
    """Letzter Wert, der zum Zeitpunkt row_end bekannt ist (avail <= row_end), höchstens `tolerance` alt.
    Gibt (Wert, Verfügbarkeitszeitpunkt) zurück."""
    left = pd.DataFrame({"t": pd.DatetimeIndex(row_end)})
    right = pd.DataFrame({"t": pd.DatetimeIndex(avail_index), "v": np.asarray(values, float),
                          "avail": pd.DatetimeIndex(avail_index)}).dropna(subset=["v"]).sort_values("t")
    m = pd.merge_asof(left, right, on="t", direction="backward", tolerance=tolerance)
    return m["v"].to_numpy(), pd.DatetimeIndex(m["avail"])


# =====================================================================================
# 1. Volumen aus den 1-Min-Kerzen
# =====================================================================================
def read_kline_volume(path):
    """Eine Monatsdatei von data.binance.vision -> Quote-Volumen, Trades, Taker-Kaufvolumen je Minute."""
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(f, header=None)
    if not str(df.iloc[0, 0]).strip().isdigit():     # manche Dateien haben eine Kopfzeile
        df = df.iloc[1:]
    df = df.iloc[:, :len(KLINE_COLS)]
    df.columns = KLINE_COLS
    t = pd.to_numeric(df["open_time"]).astype("int64").to_numpy()
    unit = "us" if t.max() > 1e14 else "ms"          # ab 2025: Mikrosekunden
    idx = pd.to_datetime(t, unit=unit, utc=True)
    return pd.DataFrame({c: pd.to_numeric(df[c]).to_numpy(float)
                         for c in ["quote_volume", "n_trades", "taker_quote"]}, index=idx)


def hourly_volume(zip_paths, freq="1h"):
    """Summen je Stunde (Zuordnung wie bei der RV: Startzeit der Kerze, abgerundet auf die Stunde)."""
    v = pd.concat([read_kline_volume(p) for p in zip_paths]).sort_index()
    v = v[~v.index.duplicated()]
    out = v.groupby(v.index.floor(freq).tz_localize(None)).sum()
    out.index.name = "date"
    return out


def volume_features(vol, index, window_days=30):
    """vol: Stundensummen. Relativ zum Mittel derselben Uhrzeit der letzten `window_days` Tage
    (ohne die aktuelle Stunde, also nur Vergangenheit). Das entfernt das Tagesmuster des Volumens."""
    index = pd.DatetimeIndex(index)
    v = vol.reindex(index)
    hour = index.hour

    def rel(x):
        lx = pd.Series(np.log(x.where(x > 0)).to_numpy(), index=index)
        base = lx.groupby(hour).transform(
            lambda z: z.shift(1).rolling(window_days, min_periods=max(5, window_days // 3)).mean())
        return (lx - base).to_numpy()

    qv = v["quote_volume"]
    return pd.DataFrame({"exo_vol_logrel": rel(qv),
                         "exo_vol_trades_rel": rel(v["n_trades"]),
                         "exo_vol_taker": (v["taker_quote"] / qv.where(qv > 0)).to_numpy()}, index=index)


# =====================================================================================
# 2. Deribit DVOL
# =====================================================================================
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DVOL_CURRENCY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
DVOL_PROXY = {"SOLUSDT": "ETH"}           # kein DVOL für SOL: ETH als Ersatz, markiert mit exo_dvol_proxy


def _fetch_dvol_range(currency, start, end, session, pause=0.1):
    """Kerzen [ts, open, high, low, close] rückwärts paginiert (continuation = nächstes end_timestamp)."""
    rows, end_ms, start_ms = [], int(end.timestamp() * 1000), int(start.timestamp() * 1000)
    while True:
        r = session.get(DERIBIT_URL, params={"currency": currency, "start_timestamp": start_ms,
                                             "end_timestamp": end_ms, "resolution": "3600"}, timeout=30)
        r.raise_for_status()
        res = r.json()["result"]
        rows += res["data"]
        cont = res.get("continuation")
        if not res["data"] or cont is None or cont <= start_ms or cont >= end_ms:
            break
        end_ms = cont
        time.sleep(pause)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")      # UTC, naiv
    return df[~df.index.duplicated()].sort_index()


def load_dvol(currency, start, end, cache_dir):
    """DVOL-Stundenkerzen mit Cache (data/raw/deribit/<CUR>_dvol_3600.parquet), lädt nur fehlende Ränder."""
    import requests

    path = Path(cache_dir) / f"{currency}_dvol_3600.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    todo = [(start, end)] if old is None or old.empty else \
        [(a, b) for a, b in [(start, old.index.min()), (old.index.max(), end)] if b - a > pd.Timedelta(hours=2)]
    if todo:
        with requests.Session() as s:
            new = [_fetch_dvol_range(currency, a.tz_localize("UTC"), b.tz_localize("UTC"), s) for a, b in todo]
        df = pd.concat(([old] if old is not None else []) + new)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        old = df
    return old


def dvol_features(sid, d, dvol, proxy, rv_window_days=30):
    """Zeile t nutzt die letzte DVOL-Kerze mit Zeitstempel T <= t. Ob T Anfang oder Ende der Kerze ist,
    ist nicht dokumentiert. Beides ist sicher: Die Kerze ist spätestens um T + 1h = Ende von Stunde t fertig.
    Werte älter als 6 Stunden (Lücken) -> NaN."""
    index = pd.DatetimeIndex(d.index)
    close, _ = _asof(index, dvol.index, dvol["close"], pd.Timedelta(hours=6))
    iv_var = (close / 100) ** 2 / HOURS_PER_YEAR                       # implizite Varianz pro Stunde
    k = max(1, int(round(pd.Timedelta(days=1) / _step(index))))
    rv_mean = d["rv"].rolling(rv_window_days * k, min_periods=5 * k).mean().to_numpy()   # bis inkl. t
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({"exo_dvol_level": np.log(iv_var),
                             "exo_dvol_vrp": np.log(iv_var) - np.log(rv_mean),
                             "exo_dvol_proxy": float(proxy)}, index=index)


# =====================================================================================
# 3. Andere Märkte: VIX, S&P-500-Futures
# =====================================================================================
MARKETS = {"vix": "^VIX", "spx": "ES=F"}


def load_yf_hourly(ticker, cache_dir):
    """Stunden-Schlusskurse von Yahoo (nur ~730 Tage zurück). Index = Beginn der Kerze, UTC naiv.
    Neue Werte werden mit dem Cache zusammengeführt, damit ältere Daten nicht verloren gehen."""
    path = Path(cache_dir) / f"{ticker.replace('^', '').replace('=', '_')}_1h.parquet"
    old = pd.read_parquet(path)["close"] if path.exists() else None
    try:
        import yfinance as yf
        df = yf.download(ticker, period="730d", interval="1h", auto_adjust=False, progress=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):                       # neuere yfinance: Spalte pro Ticker
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            raise RuntimeError(f"Yahoo lieferte keine Daten für {ticker}")
        idx = pd.DatetimeIndex(close.index)
        idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
        close = pd.Series(close.to_numpy(float), index=idx, name="close")
        close = close if old is None else close.combine_first(old)
    except Exception:
        if old is None:
            raise
        print(f"  {ticker}: Download fehlgeschlagen, nutze Cache bis {old.index.max()}")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    close.sort_index().to_frame().to_parquet(path)
    return close.sort_index()


def market_features(index, vix, spx, bar=pd.Timedelta(hours=1), std_days=30):
    """Eine Yahoo-Kerze mit Beginn s gilt erst ab s + 1h als bekannt. Zeile t (Ende t + 1h) nimmt den
    letzten bis dahin bekannten Wert. Ausserhalb der Handelszeiten wird er weitergetragen (höchstens
    4 Tage, Wochenende + Feiertag), exo_mkt_closed = 1, wenn in Stunde t keine neue Futures-Kerze fertig wurde."""
    index = pd.DatetimeIndex(index)
    row_end = index + _step(index)
    tol = pd.Timedelta(days=4)
    v, _ = _asof(row_end, vix.index + bar, vix.to_numpy(), tol)
    p, avail = _asof(row_end, spx.index + bar, spx.to_numpy(), tol)
    lp = pd.Series(np.log(p), index=index)
    r = lp.diff()                                                     # 0, solange keine neue Kerze kommt
    sd = r.where(r != 0).rolling(std_days * 24, min_periods=100).std()
    closed = np.where(np.isnan(p), np.nan, (avail <= index).astype(float))   # letzte Kerze vor Beginn von t
    return pd.DataFrame({"exo_mkt_vix": np.log(v),
                         "exo_mkt_spx_ret": (r / sd).to_numpy(),
                         "exo_mkt_closed": closed}, index=index)


# =====================================================================================
# 4. Kalender (am Zielzeitpunkt, siehe features.build_panel)
# =====================================================================================
# FOMC: zweiter Sitzungstag, Statement 14:00 ET, Pressekonferenz 14:30 ET.
# Quelle: Federal Reserve, FOMC-Kalender (übernommen von fedratecalc.com/fomc-meeting-schedule/, Abruf 15.11.2026)
FOMC_DATES = ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
              "2025-10-29", "2025-12-10",
              "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
              "2026-10-28", "2026-12-09"]
# CPI: Veröffentlichung 08:30 ET. Tatsächliche Termine inkl. Shutdown 2025: September-CPI am 24.10.2025
# (statt 15.10.), Oktober-CPI gestrichen (13.11. entfällt), November-CPI am 18.12.2025 (statt 10.12.).
# Quellen: BLS-Pläne 2025 (Caxton "US Economic Announcements by Qtr"), BLS 2025-lapse-revised-release-dates,
# Bloomberg 10.10.2025 (September-CPI am 24.10.), BLS/fedratecalc.com Plan 2026. Abruf 15.11.2026.
CPI_DATES = ["2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
             "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-12-18",
             "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10",
             "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10"]
CALENDAR_OPTIONS = {"weekend": ["weekend"], "hour": ["hour_sin", "hour_cos"], "us_open": ["us_open"],
                    "events": ["fomc", "cpi", "event_day"]}


def nyse_holidays(start, end):
    """NYSE-Feiertage (ohne verkürzte Handelstage)."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                        USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                        USThanksgivingDay, nearest_workday, sunday_to_monday)

    class NYSE(AbstractHolidayCalendar):
        rules = [Holiday("Neujahr", month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr,
                 USPresidentsDay, GoodFriday, USMemorialDay,
                 Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
                 Holiday("Independence", month=7, day=4, observance=nearest_workday), USLaborDay,
                 USThanksgivingDay, Holiday("Weihnachten", month=12, day=25, observance=nearest_workday)]

    extra = pd.DatetimeIndex(["2025-01-09"])            # Staatstrauer Jimmy Carter, NYSE geschlossen
    h = NYSE().holidays(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=7))
    return h.union(extra)


def _overlap(start_min, a, b):
    # Stunde [start_min, start_min + 60) überschneidet das Fenster [a, b) in Minuten nach Mitternacht
    return (start_min < b) & (start_min + 60 > a)


def calendar_features(target, which):
    """target: Zielzeitpunkte (Beginn der Zielstunde, UTC naiv). which: Teilmenge von CALENDAR_OPTIONS
    ohne 'weekend' (das macht build_panel selbst). Gibt ein DataFrame mit gleicher Länge zurück."""
    t = pd.DatetimeIndex(pd.to_datetime(target))
    out = pd.DataFrame(index=range(len(t)))
    if "hour" in which:
        ang = 2 * np.pi * (t.hour + 0.5) / 24
        out["hour_sin"], out["hour_cos"] = np.sin(ang), np.cos(ang)
    et = t.tz_localize("UTC").tz_convert("America/New_York")
    start_min = np.asarray(et.hour * 60 + et.minute)
    day = pd.DatetimeIndex(et.tz_localize(None).normalize())
    if "us_open" in which:
        trading = (day.dayofweek < 5) & ~day.isin(nyse_holidays(day.min(), day.max()))
        out["us_open"] = (trading & _overlap(start_min, 9 * 60 + 30, 16 * 60)).astype(float)
    if "events" in which:
        fomc_day = day.isin(pd.DatetimeIndex(FOMC_DATES))
        cpi_day = day.isin(pd.DatetimeIndex(CPI_DATES))
        out["fomc"] = (fomc_day & _overlap(start_min, 14 * 60, 16 * 60)).astype(float)
        out["cpi"] = (cpi_day & _overlap(start_min, 8 * 60 + 30, 10 * 60)).astype(float)
        out["event_day"] = (fomc_day | cpi_day).astype(float)
        lo, hi = pd.Timestamp(min(FOMC_DATES + CPI_DATES)), pd.Timestamp(max(FOMC_DATES + CPI_DATES))
        if len(t) and (t.min() < lo - pd.Timedelta(days=60) or t.max() > hi + pd.Timedelta(days=60)):
            print(f"Warnung: Termine FOMC/CPI nur {lo.date()} bis {hi.date()} hinterlegt (pipeline/exo.py)")
    return out


# =====================================================================================
# Alles zusammen
# =====================================================================================
def add_exo(daily, which=("volume", "dvol", "markets"), kline_paths=None, raw_dir=Path("data/raw"),
            cache_dir=Path("data"), dataset="binance_1h", freq="1h", skip_missing=True):
    """Ergänzt {sid: DataFrame} um exo_*-Spalten. rv/ret bleiben bitgleich (wird geprüft).
    Quellen, die fehlen (Paket, Netz), werden mit Meldung übersprungen, wenn skip_missing=True."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    out = {sid: d.drop(columns=[c for c in d.columns if c.startswith("exo_")]).copy() for sid, d in daily.items()}

    def run(name, fn):
        try:
            fn()
        except Exception as e:                                         # noqa: BLE001
            if not skip_missing:
                raise
            print(f"exo {name}: übersprungen ({type(e).__name__}: {e})")

    def volume():
        for sid, d in out.items():
            cache = cache_dir / f"{dataset}_{sid}_volume_v{VOLUME_VERSION}.parquet"
            if cache.exists():
                vol = pd.read_parquet(cache)
            else:
                paths = (kline_paths or {}).get(sid)
                if not paths:
                    raise FileNotFoundError(f"keine Kerzen-Dateien für {sid}")
                vol = hourly_volume(paths, freq)
                vol.to_parquet(cache)
            f = volume_features(vol, d.index)
            out[sid] = d.join(f)
        print("exo volume: exo_vol_logrel, exo_vol_trades_rel, exo_vol_taker")

    def dvol():
        cur = {sid: DVOL_CURRENCY.get(sid, DVOL_PROXY.get(sid)) for sid in out}
        start = min(d.index.min() for d in out.values())
        end = max(d.index.max() for d in out.values()) + pd.Timedelta(hours=2)
        data = {c: load_dvol(c, start, end, raw_dir / "deribit") for c in set(cur.values()) if c}
        for sid, d in out.items():
            if cur[sid] is None:
                print(f"exo dvol: kein DVOL für {sid}")
                continue
            out[sid] = d.join(dvol_features(sid, d, data[cur[sid]], proxy=sid not in DVOL_CURRENCY))
        print(f"exo dvol: exo_dvol_level, exo_dvol_vrp, exo_dvol_proxy (Proxy: {DVOL_PROXY})")

    def markets():
        vix = load_yf_hourly(MARKETS["vix"], raw_dir / "markets")
        spx = load_yf_hourly(MARKETS["spx"], raw_dir / "markets")
        for sid, d in out.items():
            out[sid] = d.join(market_features(d.index, vix, spx))
        first = max(vix.index.min(), spx.index.min())
        print(f"exo markets: exo_mkt_vix, exo_mkt_spx_ret, exo_mkt_closed (Daten ab {first:%Y-%m-%d})")

    for name, fn in [("volume", volume), ("dvol", dvol), ("markets", markets)]:
        if name in which:
            run(name, fn)

    for sid, d in daily.items():                                        # Schutz der gecachten Prognosen
        assert out[sid].index.equals(d.index), sid
        for c in ["rv", "ret"]:
            assert np.array_equal(out[sid][c].to_numpy(), d[c].to_numpy(), equal_nan=True), (sid, c)
    for sid, d in out.items():
        exo = [c for c in d.columns if c.startswith("exo_")]
        if exo:
            share = d[exo].notna().mean().round(2).to_dict()
            print(f"  {sid}: Anteil vorhandener Werte {share}")
    return out
