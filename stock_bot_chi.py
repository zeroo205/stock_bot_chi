# --- Imports ---
import re
import io
import html
from urllib.parse import urlparse, quote

import numpy as np
import pandas as pd
import requests
import streamlit as st

import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Clear cache safely (no-op if not available)
try:
    st.cache_data.clear()
except Exception:
    pass

# --- Config ---
st.set_page_config(page_title="📊 Stock Bot", layout="wide")

# --- News section styles ---
st.markdown("""
<style>
.news-item { margin: 0.6rem 0 1.0rem 0; }

/* Headline row used as a toggle */
.news-details { margin-left: 0; }
.news-details summary { cursor: pointer; list-style: none; }
.news-details summary::-webkit-details-marker { display: none; }
.news-details summary::before { content: "▸ "; color: #6b7280; }
.news-details[open] summary::before { content: "▾ "; }

/* Headline row layout */
.news-headline { 
    font-size: 1.05rem; 
    font-weight: 600; 
    line-height: 1.35;
    display: inline;
}

/* Date next to headline */
.news-bracket { color: #6b7280; font-size: 0.92rem; margin-left: 6px; }
         
.news-body li {
    margin-bottom: 0.25rem;
    position: relative;
    font-size: 0.92rem;
    color: #374151;
}

.news-body li::before {
    content: "•";
    color: #6b7280;
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -1em;
    position: absolute;
    left: 0;
}

</style>
""", unsafe_allow_html=True)

# ================================
# Helpers
# ================================
def extract_domain(url: str | None) -> str:
    """Return 'example.com' from a URL (no scheme or path)."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc or ""
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def normalize_url(url: str | None) -> str | None:
    """Return a normalized URL with scheme, or None."""
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u

def safe_dt_str(dt_str):
    """Format a date string safely as 'YYYY-MM-DD HH:MM' or return raw string."""
    if not dt_str:
        return ""
    try:
        return pd.to_datetime(dt_str).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt_str)

# Reject obvious non-name labels that sometimes appear in titles/headers
_GENERIC_BAD_TOKENS = {
    "港股報價", "香港即時股票股價", "經濟通", "etnet", "AASTOCKS", "阿思達克", "阿思達克財經網",
    "Detail Quote", "Interactive Chart", "Real-time Quote", "詳細報價", "互動圖表", "即時報價",
    "香港新聞財經資訊和生活平台"
}

def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    if not text:
        return False
    chinese_ranges = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x3007, 0x3007),   # Ideographic number zero
    ]
    for char in text:
        cp = ord(char)
        for a, b in chinese_ranges:
            if a <= cp <= b:
                return True
    return False

def _best_chinese_segment(text: str) -> str:
    """
    Split by common separators and pick the segment with the highest Chinese-character ratio,
    preferring shorter segments when ratios tie.
    """
    if not text:
        return ""
    parts = re.split(r"[\|｜\-—–－]+", text)
    candidates = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lowered = p.lower()
        if any(tok.lower() in lowered for tok in _GENERIC_BAD_TOKENS):
            continue
        if _contains_chinese(p):
            chinese_chars = sum(1 for ch in p if "\u4e00" <= ch <= "\u9fff")
            ratio = chinese_chars / max(len(p), 1)
            candidates.append((ratio, len(p), p))
    if candidates:
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][2]
    return text.strip()

def _clean_company_name(txt: str, *, prefer_chinese_segment: bool = False) -> str:
    """Strip code in parentheses, trim site suffixes, optionally pick best Chinese segment."""
    if not txt:
        return ""
    # remove code in parentheses: "(0001)", "(0001.HK)", "（0001）", etc.
    txt = re.sub(r"[\(（]\s*0*\d{4,5}\s*(?:\.HK)?\s*[\)）]", "", txt)
    if prefer_chinese_segment:
        txt = _best_chinese_segment(txt)
    txt = re.sub(r"\s*[\-—–]\s*.*$", "", txt).strip()
    return txt

def _valid_company_name(txt: str) -> bool:
    """Heuristic: contains Chinese, not generic labels, reasonable length."""
    if not txt:
        return False
    if not _contains_chinese(txt):
        return False
    if any(tok.lower() in txt.lower() for tok in _GENERIC_BAD_TOKENS):
        return False
    return len(txt) >= 2

# ================================
# Data Fetching (cached) - yfinance
# ================================
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_stock_data(symbol: str, is_hk_stock: bool = False) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Use yfinance. For HK stocks, append .HK to the ticker symbol.
    Returns tuple: (daily_data_for_charts, intraday_data_for_timestamp)
    """
    ticker_symbol = f"{symbol}.HK" if is_hk_stock else symbol
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Get daily data for charts
        daily_hist = ticker.history(period="6mo", interval="1d")
        # Get intraday data for latest timestamp
        intraday_hist = ticker.history(period="1d", interval="1m")
        # Rename columns to match the expected format
        daily_hist = daily_hist.rename(
            columns={
                "Open": "1. open",
                "High": "2. high",
                "Low": "3. low",
                "Close": "4. close",
                "Volume": "5. volume",
            }
        )
        intraday_hist = intraday_hist.rename(
            columns={
                "Open": "1. open",
                "High": "2. high",
                "Low": "3. low",
                "Close": "4. close",
                "Volume": "5. volume",
            }
        )
        return daily_hist, intraday_hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# ================================
# Company name helpers (HK = Chinese first)
# ================================
def _pad_hk(symbol: str) -> str:
    """Zero-pad numeric HK codes to 5 digits (e.g., '5' -> '00005', '9988' -> '09988'). If non-numeric, return as-is."""
    s = str(symbol).strip()
    if re.fullmatch(r"\d+", s):
        return s.zfill(5)
    return s

def _get_from_aastocks(symbol: str) -> str | None:
    """Try AASTOCKS Detail Quote page for Traditional Chinese name."""
    try:
        code = _pad_hk(symbol)
        url = f"https://www.aastocks.com/tc/stocks/quote/detail-quote.aspx?symbol={code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "zh-HK,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.6,en;q=0.5",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) Prefer the <title> ONLY if it contains the code pattern '(09988.HK)'
        title = soup.find("title")
        if title:
            t = title.get_text(strip=True)
            m = re.search(r'([^\(\（]+?)\s*[\(（]\s*0*\d{4,5}\.HK\s*[\)）]', t)
            if m:
                cand = _clean_company_name(m.group(1))
                if _valid_company_name(cand):
                    return cand

        # 2) Common containers
        selectors = [
            '#stockName',
            'h1', 'h2', 'div[class*="name"]', 'span[class*="name"]',
            '.quote-header h1', '.stock-name',
        ]
        for sel in selectors:
            for el in soup.select(sel):
                cand = _clean_company_name(el.get_text(strip=True))
                if _valid_company_name(cand):
                    return cand

        # 3) Table label fallback
        label = soup.find(string=re.compile(r"公司名稱"))
        if label:
            td = label.find_parent("td")
            if td and td.find_next_sibling("td"):
                cand = _clean_company_name(td.find_next_sibling("td").get_text(strip=True))
                if _valid_company_name(cand):
                    return cand
    except Exception as e:
        print(f"AASTOCKS error for {symbol}: {e}")
    return None

def _get_from_etnet(symbol: str) -> str | None:
    """Try ETNet realtime quote for Chinese name."""
    try:
        code = _pad_hk(symbol)
        url = f"https://www.etnet.com.hk/www/tc/stocks/realtime/quote.php?code={code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "zh-HK,zh-TW;q=0.9,zh;q=0.8,en;q=0.7,en;q=0.6",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("title")
        if title:
            title_text = title.get_text(strip=True)
            cand = _clean_company_name(title_text, prefer_chinese_segment=True)
            if _valid_company_name(cand):
                return cand

        for selector in ['h1', 'h2', '.stockname', 'div[class*="StockName"]', 'div[class*="name"]']:
            for el in soup.select(selector):
                txt = el.get_text(strip=True)
                txt = re.sub(r"\([^\)]+\)", "", txt)  # remove (09988)
                txt = txt.replace("即時報價", "").replace("經濟通", "").strip()
                if _contains_chinese(txt):
                    return txt
    except Exception as e:
        print(f"ETNet error for {symbol}: {e}")
    return None

def _get_from_yahoo_hk(symbol: str) -> str | None:
    """Try Yahoo Finance HK page for Chinese name."""
    try:
        code = _pad_hk(symbol)
        code_num = str(int(code)) if code.isdigit() else code  # "09988" -> "9988"
        url = f"https://hk.finance.yahoo.com/quote/{code_num}.HK"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "zh-HK,zh-TW;q=0.9,zh;q=0.8,en;q=0.7,en;q=0.6",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        if title:
            name = re.split(r"\s*\(", title.get_text(strip=True), 1)[0]
            if _contains_chinese(name):
                return name
    except Exception as e:
        print(f"Yahoo HK error for {symbol}: {e}")
    return None

@st.cache_data(ttl=86400)
def _get_hkex_dual_counter_map() -> dict[str, str]:
    """
    Optional official fallback: HKEX Dual-Counter list (contains Chinese short names for dual counters).
    Cached daily. Not every HK stock is included.
    """
    try:
        url = "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Trading/Securities/Securities-Lists/Dual_Counter_Security_List.xlsx"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))
        # Guess columns (HKEX may change headers slightly)
        code_col = next(c for c in df.columns if "HKD" in c and "Code" in c)
        name_col = next(c for c in df.columns if "Chinese" in c and "Name" in c)
        df["code5"] = (
            df[code_col].astype(str).str.extract(r"\((\d+)\)").fillna("").astype(str).str.zfill(5)
        )
        mapping = dict(zip(df["code5"], df[name_col].fillna("").astype(str)))
        return mapping
    except Exception:
        return {}

def _get_from_hkex_dual(symbol: str) -> str | None:
    mp = _get_hkex_dual_counter_map()
    return mp.get(_pad_hk(symbol))

def _get_english_name_from_yfinance(symbol: str) -> str:
    """Fallback to English name from yfinance."""
    try:
        ticker = yf.Ticker(f"{symbol}.HK")
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or info.get("displayName")
        return name or f"{symbol}.HK"
    except Exception:
        return f"{symbol}.HK"

def _get_us_stock_name(symbol: str) -> str:
    """Get English name for US stocks."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or info.get("displayName")
        return name or symbol
    except Exception:
        return symbol

@st.cache_data(ttl=86400)
def get_company_name(symbol: str, is_hk_stock: bool = False) -> str:
    """
    Returns the company name. For HK stocks, prioritizes Traditional Chinese names via:
    AASTOCKS → ETNet → Yahoo HK → HKEX dual-counter → yfinance English.
    """
    if is_hk_stock:
        for fn in (_get_from_aastocks, _get_from_etnet, _get_from_yahoo_hk, _get_from_hkex_dual):
            name = fn(symbol)
            if name:
                return name
        return _get_english_name_from_yfinance(symbol)
    else:
        return _get_us_stock_name(symbol)

# ================================
# Indicators

@st.cache_data(ttl=300)
def fetch_financial_info(symbol: str, is_hk_stock: bool = False):
    ticker_symbol = f"{symbol}.HK" if is_hk_stock else symbol
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "analyst_rating": info.get("recommendationMean")
        }
    except Exception as e:
        st.error(f"Error fetching financial info: {e}")
        return {}

# ================================
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close_series = df["4. close"]
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close_series = df["4. close"]
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def detect_macd_signals(macd: pd.Series, signal: pd.Series):
    macd = macd.dropna()
    signal = signal.reindex(macd.index).dropna()
    macd = macd.reindex(signal.index)
    buy_signals, sell_signals = [], []
    for i in range(1, len(macd)):
        pm, ps = macd.iloc[i - 1], signal.iloc[i - 1]
        cm, cs = macd.iloc[i], signal.iloc[i]
        if pm < ps and cm > cs:
            buy_signals.append((macd.index[i], cm))
        elif pm > ps and cm < cs:
            sell_signals.append((macd.index[i], cm))
    return buy_signals, sell_signals

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def bollinger_bands(close_series: pd.Series, window: int = 20, mult: float = 2.0):
    mid = close_series.rolling(window).mean()
    std = close_series.rolling(window).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower

def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    high = df["2. high"]
    low = df["3. low"]
    close = df["4. close"]
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def atr(df: pd.DataFrame, period: int = 14):
    high = df["2. high"]
    low = df["3. low"]
    close = df["4. close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period).mean()

# ================================
# Traditional Chinese news (NO sentiment)
# ================================
def _strip_html_to_text(s: str) -> str:
    if not s:
        return ""
    try:
        soup = BeautifulSoup(s, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return s

def _parse_google_rss(xml_text: str, limit: int = 8) -> list[dict]:
    """Parse Google News RSS XML -> list of {title,url,published_at,description,source}."""
    soup = BeautifulSoup(xml_text, "xml")
    out = []
    for it in soup.select("item"):
        title = (it.title.get_text(strip=True) if it.title else "") or "No title"
        link = it.link.get_text(strip=True) if it.link else None
        pub = it.pubDate.get_text(strip=True) if it.pubDate else None
        desc = it.description.get_text(strip=True) if it.description else ""
        source_tag = it.find("source")
        source_name = source_tag.get_text(strip=True) if source_tag else ""
        out.append({
            "title": title,
            "url": normalize_url(link),
            "published_at": pub,
            "description": _strip_html_to_text(desc),
            "source": source_name or extract_domain(link) or "Google News",
        })
        if len(out) >= limit:
            break
    return out

@st.cache_data(ttl=300)
def _fetch_tc_news_google(query: str, limit: int = 8) -> list[dict]:
    """Search-focused zh-Hant news via Google News RSS with UA header and HK/TW fallbacks."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    base = "https://news.google.com/rss/search"
    urls = [
        f"{base}?q={quote(query)}&hl=zh-HK&gl=HK&ceid=HK:zh-Hant",
        f"{base}?q={quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    ]
    for rss_url in urls:
        try:
            resp = requests.get(rss_url, timeout=15, headers=headers)
            resp.raise_for_status()
            items = _parse_google_rss(resp.text, limit=limit)
            if items:
                return items
        except Exception as e:
            print(f"Google RSS fetch failed: {rss_url} -> {e}")
    return []

@st.cache_data(ttl=300)
def _fetch_tc_business_headlines(limit: int = 6) -> list[dict]:
    """Fallback: zh-Hant BUSINESS headlines (ensures something appears even if search feed is empty)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    urls = [
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=zh-HK&gl=HK&ceid=HK:zh-Hant",
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            items = _parse_google_rss(resp.text, limit=limit)
            if items:
                return items
        except Exception as e:
            print(f"Business RSS fetch failed: {url} -> {e}")
    return []

def fetch_traditional_chinese_news(symbol: str, is_hk_stock: bool) -> list[dict]:
    """
    Sentiment-free TC news.
    Return list[dict] with schema used by UI:
    {title, url, source, published_at, description}
    """
    # Build query using existing HK name resolver
    name = get_company_name(symbol, is_hk_stock) or symbol
    if is_hk_stock and any('一' <= ch <= '龥' for ch in name):  # contains Chinese
        query = f"{name} {symbol}"
    else:
        query = name
    items = _fetch_tc_news_google(query, limit=8)
    if not items:
        items = _fetch_tc_business_headlines(limit=6)
    # Optionally keep top 5 most recent by published_at
    try:
        items = sorted(
            items,
            key=lambda a: pd.to_datetime(a.get("published_at")) if a.get("published_at") else pd.Timestamp.min,
            reverse=True,
        )[:5]
    except Exception:
        items = items[:5]
    return items
# ================================
# Recommendation logic (RSI + MACD only; sentiment removed)
# ================================
def combined_recommendation(rsi_val: float, macd_signal: str):
    score = 0
    rsi_text = "Neutral"
    macd_text = "Neutral"

    if rsi_val < 30:
        score += 1
        rsi_text = "📉 RSI < 30: Oversold (Buy)"
    elif rsi_val > 70:
        score -= 1
        rsi_text = "📈 RSI > 70: Overbought (Sell)"
    else:
        rsi_text = "📊 RSI in neutral range"

    if macd_signal == "buy":
        score += 1
        macd_text = "🟢 MACD crossover: Buy signal"
    elif macd_signal == "sell":
        score -= 1
        macd_text = "🔴 MACD crossover: Sell signal"
    else:
        macd_text = "⚪ No recent MACD crossover"

    if score >= 2:
        return "✅ Strong Buy Recommendation", rsi_text, macd_text, "success"
    elif score == 1:
        return "☑️ Moderate Buy Recommendation", rsi_text, macd_text, "info"
    elif score == 0:
        return "ℹ️ Hold Recommendation", rsi_text, macd_text, "warning"
    else:
        return "⚠️ Sell Recommendation", rsi_text, macd_text, "error"

# ================================
# UI
# ================================
st.title("📊 Stock Bot")

# Market selection
market_type = st.radio("Select Market:", ("Hong Kong", "US"), horizontal=True)
symbol = st.text_input(
    "Enter Stock Symbol",
    placeholder="e.g., 0001, 9988" if market_type == "Hong Kong" else "e.g., AAPL, MSFT",
).strip().upper()

if symbol:
    is_hk_stock = (market_type == "Hong Kong")

    # Fetch data - now getting both daily and intraday data
    daily_df, intraday_df = fetch_stock_data(symbol, is_hk_stock)
    if daily_df is None or daily_df.empty:
        st.error("Failed to fetch stock data. Please check the symbol and try again.")
        st.stop()

    # Use daily data for charts and indicators
    df = daily_df

    # Use intraday data for timestamp if available, otherwise fall back to daily data
    if intraday_df is not None and not intraday_df.empty:
        timestamp_df = intraday_df
    else:
        timestamp_df = daily_df

    # --- Latest timestamp & delay handling (robust, tz-safe) ---
    def _to_utc_aware(ts) -> pd.Timestamp:
        """
        Return a pandas Timestamp that is tz-aware in UTC.
        - If ts is naive -> localize to UTC
        - If ts has a timezone -> convert to UTC
        """
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is None and getattr(ts, "tz", None) is None:
            return ts.tz_localize("UTC")
        else:
            return ts.tz_convert("UTC")

    latest_raw_ts = timestamp_df.index[-1]
    latest_ts_utc = _to_utc_aware(latest_raw_ts)
    now_utc = pd.Timestamp.now(tz="UTC")
    data_delay_min = (now_utc - latest_ts_utc).total_seconds() / 60.0
    delay_note = "(Delayed)" if data_delay_min > 15 else ""

    # Build display string (show in HKT for readability)
    try:
        latest_update_str = latest_ts_utc.tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        latest_update_str = latest_ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Window for display (last up to 100 points)
    LOOKBACK_DAYS = min(100, len(df))
    idx_plot = df.index[-LOOKBACK_DAYS:]
    df_plot = df.loc[idx_plot]

    # Define series
    open_series = df["1. open"]
    high_series = df["2. high"]
    low_series = df["3. low"]
    close_series = df["4. close"]
    volume_series = df["5. volume"]

    display_symbol = f"{symbol}.HK" if is_hk_stock else symbol

    # Get company name (HK prioritizes Traditional Chinese)
    company_name = get_company_name(symbol, is_hk_stock) or display_symbol

    # Caption with precise timestamp and delay status
    st.caption(f"📅 Latest stock data for {display_symbol} ({company_name}): {latest_update_str} {delay_note}")

    # Compute indicators
    rsi = calculate_rsi(df)
    macd, signal, hist = calculate_macd(df)
    buy_signals, sell_signals = detect_macd_signals(macd, signal)
    stoch_k, stoch_d = stochastic_oscillator(df)
    atr14 = atr(df)

    # Overlays
    sma20_full = sma(close_series, 20)
    sma50_full = sma(close_series, 50)
    bb_up_series, bb_mid_series, bb_low_series = bollinger_bands(close_series, window=20, mult=2.0)

    sma20 = sma20_full.reindex(idx_plot)
    sma50 = sma50_full.reindex(idx_plot)
    bb_up = bb_up_series.reindex(idx_plot)
    bb_low = bb_low_series.reindex(idx_plot)

    # Latest values
    latest_close = float(close_series.iloc[-1])
    latest_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    latest_stoch_k = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else None
    latest_stoch_d = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else None
    latest_atr = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else None

    # Last-bar MACD signal
    latest_macd_signal = "None"
    if len(macd) >= 2 and len(signal) >= 2:
        if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            latest_macd_signal = "Buy"
        elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            latest_macd_signal = "Sell"

    # --- News (Traditional Chinese, NO sentiment) ---
    articles = fetch_traditional_chinese_news(symbol, is_hk_stock)

    # Recommendation banner    
    financials = fetch_financial_info(symbol, is_hk_stock)
    extra_score = 0
    rating = financials.get("analyst_rating")
    pe = financials.get("pe_ratio")
    if rating is not None:
        if rating <= 2:
            extra_score += 1
        elif rating >= 4:
            extra_score -= 1
    if pe is not None:
        if pe < 15:
            extra_score += 1
        elif pe > 30:
            extra_score -= 1

    recommendation, rsi_text, macd_text, banner_type = combined_recommendation(
        latest_rsi, latest_macd_signal
    )
    
    score_map = {
        "✅ Strong Buy Recommendation": 2,
        "☑️ Moderate Buy Recommendation": 1,
        "ℹ️ Hold Recommendation": 0,
        "⚠️ Sell Recommendation": -1
    }
    final_score = score_map.get(recommendation, 0) + extra_score
    if final_score >= 2:
        recommendation = "✅ Strong Buy Recommendation"
        banner_type = "success"
    elif final_score == 1:
        recommendation = "☑️ Moderate Buy Recommendation"
        banner_type = "info"
    elif final_score == 0:
        recommendation = "ℹ️ Hold Recommendation"
        banner_type = "warning"
    else:
        recommendation = "⚠️ Sell Recommendation"
        banner_type = "error"

    getattr(st, banner_type)(recommendation)

    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Last Close", f"${latest_close:,.2f}" if not is_hk_stock else f"HK${latest_close:,.2f}")

    with kpi2:
        st.metric("RSI (14)", f"{latest_rsi:.2f}")

    with kpi3:
        pe_val = financials.get("pe_ratio")
        st.metric("Trailing P/E", f"{pe_val:.2f}" if pe_val is not None else "N/A")

    with kpi4:
        rating_val = financials.get("analyst_rating")
        st.metric("Analyst Rating", f"{rating_val:.2f}" if rating_val is not None else "N/A")

    # Two-column layout: Indicators & News
    col_ind, col_news = st.columns([1, 1], gap="large")

    with col_ind:
        st.subheader("🎯 Indicators")

        # Indicators text
        st.markdown(f"""
        <ul>
        <li><span title="Relative Strength Index: Measures momentum. Below 30 is oversold, above 70 is overbought.">
        📉 RSI (14): {latest_rsi:.3f}</span></li>
        <li><span title="MACD: Moving Average Convergence Divergence. Indicates trend changes.">
        📊 MACD Signal: {latest_macd_signal}</span></li>
        <li><span title="Trailing P/E: Price divided by earnings over the last 12 months. Lower is cheaper.">
        📈 Trailing P/E Ratio: {financials.get('pe_ratio', 'N/A'):.3f}</span></li>
        <li><span title="Forward P/E: Price divided by projected earnings. Useful for growth expectations.">
        📈 Forward P/E Ratio: {financials.get('forward_pe', 'N/A'):.3f}</span></li>
        <li><span title="Price-to-Book: Compares market value to book value. Below 1 may indicate undervaluation.">
        📘 Price-to-Book Ratio: {financials.get('pb_ratio', 'N/A'):.3f}</span></li>
        <li><span title="Analyst Rating: Average recommendation. 1 = Strong Buy, 5 = Sell.">
        🧠 Analyst Rating: {financials.get('analyst_rating', 'N/A'):.3f}</span></li>
        </ul>
        """, unsafe_allow_html=True)

        # Snapshot
        if not pd.isna(sma20_full.iloc[-1]) and not pd.isna(sma50_full.iloc[-1]):
            if sma20_full.iloc[-1] > sma50_full.iloc[-1]:
                sma_bias = "Bullish (SMA20 > SMA50)"
            elif sma20_full.iloc[-1] < sma50_full.iloc[-1]:
                sma_bias = "Bearish (SMA20 < SMA50)"
            else:
                sma_bias = "Neutral"
        else:
            sma_bias = "Neutral"

        if latest_stoch_k is not None and latest_stoch_d is not None:
            stoch_text = (
                f"%K: {latest_stoch_k:.1f}, %D: {latest_stoch_d:.1f} — " +
                ("Overbought (>80)" if latest_stoch_k > 80 else "Oversold (<20)" if latest_stoch_k < 20 else "Neutral")
            )
        else:
            stoch_text = "N/A"

        atr_text = f"{latest_atr:.2f}" if latest_atr is not None else "N/A"

        st.markdown(
            f"- **SMA Bias:** {sma_bias}\n"
            f"- **Stochastic (14,3):** {stoch_text}\n"
            f"- **ATR (14):** {atr_text}"
        )
    
    with col_news:
        st.subheader("📰 Recent News")
        if articles:
            for a in articles:
                title = a.get("title", "No title") or "No title"
                raw_url = a.get("url")
                url = normalize_url(raw_url)

                # Headline clickable
                headline_html = f'<a href="{html.escape(url)}" target="_blank">{html.escape(title)}</a>'

                st.markdown(
                    f"""
                    {headline_html}
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No news articles available at the moment.")

    # ====== Charts ======
    st.subheader("📈 Trends")

    # Price + Volume (top figure with subplot)
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    # Candlesticks
    fig_price.add_trace(
        go.Candlestick(
            x=idx_plot,
            open=df_plot["1. open"], high=df_plot["2. high"],
            low=df_plot["3. low"], close=df_plot["4. close"],
            name="Price",
            increasing=dict(line=dict(color="#00B26F")),
            decreasing=dict(line=dict(color="#E45756")),
        ),
        row=1, col=1
    )
    # SMA 20/50
    fig_price.add_trace(
        go.Scatter(x=idx_plot, y=sma20, mode='lines', name='SMA 20', line=dict(color="#1f77b4", width=1.5)),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=idx_plot, y=sma50, mode='lines', name='SMA 50', line=dict(color="#ff7f0e", width=1.5)),
        row=1, col=1
    )
    # Bollinger bands
    fig_price.add_trace(
        go.Scatter(x=idx_plot, y=bb_up, mode='lines', name='BB Upper', line=dict(color="rgba(31,119,180,0.4)", width=1)),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(x=idx_plot, y=bb_low, mode='lines', name='BB Lower', line=dict(color="rgba(31,119,180,0.4)", width=1)),
        row=1, col=1
    )
    # Mark last MACD cross on price
    if buy_signals:
        last_buy = buy_signals[-1][0]
        if last_buy in idx_plot:
            price_at_buy = df.loc[last_buy, "4. close"]
            fig_price.add_trace(
                go.Scatter(
                    x=[last_buy], y=[price_at_buy], mode="markers",
                    marker=dict(symbol="triangle-up", color="#00B26F", size=12),
                    name="Buy Xover",
                ),
                row=1, col=1
            )
    if sell_signals:
        last_sell = sell_signals[-1][0]
        if last_sell in idx_plot:
            price_at_sell = df.loc[last_sell, "4. close"]
            fig_price.add_trace(
                go.Scatter(
                    x=[last_sell], y=[price_at_sell], mode="markers",
                    marker=dict(symbol="triangle-down", color="#E45756", size=12),
                    name="Sell Xover",
                ),
                row=1, col=1
            )
    # Volume colored by up/down
    up_mask = df_plot["4. close"] >= df_plot["1. open"]
    volume_colors = np.where(up_mask, "#C7F2D8", "#F7C9C6")
    fig_price.add_trace(
        go.Bar(x=idx_plot, y=df_plot["5. volume"], name="Volume", marker_color=volume_colors, opacity=0.8),
        row=2, col=1
    )
    fig_price.update_layout(
        title=f"{display_symbol} — Price with SMA(20/50) & Bollinger Bands",
        xaxis=dict(showgrid=False), yaxis=dict(title="Price"),
        xaxis2=dict(showgrid=False), yaxis2=dict(title="Volume"),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=110),
        height=680,
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # MACD (bottom figure)
    macd_idx = macd.index[-LOOKBACK_DAYS:]
    macd_plot = go.Figure()
    macd_plot.add_trace(
        go.Bar(x=macd_idx, y=hist.reindex(macd_idx), name="Histogram", marker_color="#9ecae1")
    )
    macd_plot.add_trace(
        go.Scatter(x=macd_idx, y=macd.reindex(macd_idx), mode='lines', name='MACD', line=dict(color="#2ca02c", width=1.5))
    )
    macd_plot.add_trace(
        go.Scatter(x=macd_idx, y=signal.reindex(macd_idx), mode='lines', name='Signal', line=dict(color="#d62728", width=1.5))
    )
    # Buy/Sell markers within window
    bxs = [(t, v) for (t, v) in buy_signals if t in macd_idx]
    sxs = [(t, v) for (t, v) in sell_signals if t in macd_idx]
    if bxs:
        macd_plot.add_trace(
            go.Scatter(x=[t for (t, v) in bxs], y=[v for (t, v) in bxs], mode='markers',
                       marker=dict(color='green', size=9), name='Buy Xover')
        )
    if sxs:
        macd_plot.add_trace(
            go.Scatter(x=[t for (t, v) in sxs], y=[v for (t, v) in sxs], mode='markers',
                       marker=dict(color='red', size=9), name='Sell Xover')
        )
    macd_plot.update_layout(
        title="MACD",
        xaxis_title="Date", yaxis_title="Value",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=110),
        height=420,
    )
    st.plotly_chart(macd_plot, use_container_width=True)

st.caption("⚠️ Educational use only. Not financial advice.")