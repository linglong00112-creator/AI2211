import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import pandas_ta as ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on a DataFrame with OHLCV columns."""
    if df.empty or len(df) < 50:
        return df

    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)
    # MACD
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd.get("MACD_12_26_9")
        df["MACD_signal"] = macd.get("MACDs_12_26_9")
        df["MACD_hist"] = macd.get("MACDh_12_26_9")
    # EMAs
    df["EMA_9"] = ta.ema(df["Close"], length=9)
    df["EMA_21"] = ta.ema(df["Close"], length=21)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    df["EMA_200"] = ta.ema(df["Close"], length=200)
    # ATR
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None:
        # pandas-ta >=0.4.x uses "BBU_20_2.0_2.0" column naming
        df["BB_upper"] = bb.get("BBU_20_2.0") or bb.get("BBU_20_2.0_2.0")
        df["BB_middle"] = bb.get("BBM_20_2.0") or bb.get("BBM_20_2.0_2.0")
        df["BB_lower"] = bb.get("BBL_20_2.0") or bb.get("BBL_20_2.0_2.0")
    # Volume
    df["Volume_SMA"] = ta.sma(df["Volume"], length=20)
    # ADX
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None:
        df["ADX"] = adx.get("ADX_14")
        df["DMP"] = adx.get("DMP_14")
        df["DMN"] = adx.get("DMN_14")
    # Stochastic
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    if stoch is not None:
        df["STOCH_K"] = stoch.get("STOCHk_14_3_3")
        df["STOCH_D"] = stoch.get("STOCHd_14_3_3")
    # Support / Resistance
    df["Resistance"] = df["High"].rolling(window=20).max()
    df["Support"] = df["Low"].rolling(window=20).min()

    return df


def get_indicator_signals(df: pd.DataFrame) -> Dict[str, dict]:
    """Generate individual buy/sell/neutral signals from each indicator."""
    signals = {}
    if df.empty or len(df) < 50:
        return {"error": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["Close"]

    # --- RSI ---
    rsi = last.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30:
            signals["RSI"] = {"signal": "BUY", "value": round(rsi, 1), "reason": f"Oversold (RSI={rsi:.1f})"}
        elif rsi > 70:
            signals["RSI"] = {"signal": "SELL", "value": round(rsi, 1), "reason": f"Overbought (RSI={rsi:.1f})"}
        else:
            signals["RSI"] = {"signal": "NEUTRAL", "value": round(rsi, 1), "reason": f"Neutral (RSI={rsi:.1f})"}

    # --- MACD ---
    macd = last.get("MACD")
    macd_signal = last.get("MACD_signal")
    prev_macd = prev.get("MACD")
    prev_macd_signal = prev.get("MACD_signal")
    if all(v is not None and not pd.isna(v) for v in [macd, macd_signal, prev_macd, prev_macd_signal]):
        if prev_macd < prev_macd_signal and macd > macd_signal:
            signals["MACD"] = {"signal": "BUY", "value": round(macd, 4), "reason": "Bullish crossover"}
        elif prev_macd > prev_macd_signal and macd < macd_signal:
            signals["MACD"] = {"signal": "SELL", "value": round(macd, 4), "reason": "Bearish crossover"}
        elif macd > macd_signal:
            signals["MACD"] = {"signal": "BUY", "value": round(macd, 4), "reason": "Histogram above zero"}
        else:
            signals["MACD"] = {"signal": "SELL", "value": round(macd, 4), "reason": "Histogram below zero"}

    # --- EMA Crossover ---
    ema9 = last.get("EMA_9")
    ema21 = last.get("EMA_21")
    ema50 = last.get("EMA_50")
    ema200 = last.get("EMA_200")
    if all(v is not None and not pd.isna(v) for v in [ema9, ema21, ema50]):
        if ema9 > ema21 > ema50:
            signals["EMA"] = {"signal": "BUY", "value": round(ema9, 2), "reason": "EMA 9 > 21 > 50 (Bullish trend)"}
        elif ema9 < ema21 < ema50:
            signals["EMA"] = {"signal": "SELL", "value": round(ema9, 2), "reason": "EMA 9 < 21 < 50 (Bearish trend)"}
        elif ema9 > ema21:
            signals["EMA"] = {"signal": "BUY", "value": round(ema9, 2), "reason": "EMA 9 above 21 (Short-term bullish)"}
        else:
            signals["EMA"] = {"signal": "SELL", "value": round(ema9, 2), "reason": "EMA 9 below 21 (Short-term bearish)"}

    # --- Bollinger Bands ---
    bb_lower = last.get("BB_lower")
    bb_upper = last.get("BB_upper")
    bb_middle = last.get("BB_middle")
    if all(v is not None and not pd.isna(v) for v in [bb_lower, bb_upper, bb_middle]):
        if close <= bb_lower:
            signals["BB"] = {"signal": "BUY", "value": round(close, 2), "reason": "Price at lower band (oversold)"}
        elif close >= bb_upper:
            signals["BB"] = {"signal": "SELL", "value": round(close, 2), "reason": "Price at upper band (overbought)"}
        elif close > bb_middle:
            signals["BB"] = {"signal": "BUY", "value": round(close, 2), "reason": "Above middle band"}
        else:
            signals["BB"] = {"signal": "SELL", "value": round(close, 2), "reason": "Below middle band"}

    # --- ADX ---
    adx = last.get("ADX")
    dmp = last.get("DMP")
    dmn = last.get("DMN")
    if all(v is not None and not pd.isna(v) for v in [adx, dmp, dmn]):
        if adx > 25:
            if dmp > dmn:
                signals["ADX"] = {"signal": "BUY", "value": round(adx, 1), "reason": f"Strong trend (ADX={adx:.1f}, +DI > -DI)"}
            else:
                signals["ADX"] = {"signal": "SELL", "value": round(adx, 1), "reason": f"Strong trend (ADX={adx:.1f}, -DI > +DI)"}
        else:
            signals["ADX"] = {"signal": "NEUTRAL", "value": round(adx, 1), "reason": f"Weak trend (ADX={adx:.1f}, range market)"}

    # --- Stochastic ---
    stoch_k = last.get("STOCH_K")
    stoch_d = last.get("STOCH_D")
    if all(v is not None and not pd.isna(v) for v in [stoch_k, stoch_d]):
        if stoch_k < 20 and stoch_d < 20:
            signals["STOCH"] = {"signal": "BUY", "value": round(stoch_k, 1), "reason": f"Oversold (K={stoch_k:.1f})"}
        elif stoch_k > 80 and stoch_d > 80:
            signals["STOCH"] = {"signal": "SELL", "value": round(stoch_k, 1), "reason": f"Overbought (K={stoch_k:.1f})"}
        else:
            signals["STOCH"] = {"signal": "NEUTRAL", "value": round(stoch_k, 1), "reason": f"Neutral (K={stoch_k:.1f})"}

    # --- Volume ---
    vol = last.get("Volume")
    vol_sma = last.get("Volume_SMA")
    if vol is not None and vol_sma is not None and not pd.isna(vol_sma) and vol_sma > 0:
        if vol > vol_sma * 1.5:
            signals["VOLUME"] = {"signal": "CONFIRM", "value": round(vol / vol_sma, 1), "reason": f"High volume ({vol/vol_sma:.1f}x avg)"}
        else:
            signals["VOLUME"] = {"signal": "NEUTRAL", "value": round(vol / vol_sma, 1), "reason": f"Normal volume"}

    # --- Price vs EMAs ---
    if ema200 is not None and not pd.isna(ema200):
        if close > ema200:
            signals["TREND_LONG"] = {"signal": "BUY", "value": round(close, 2), "reason": "Price above EMA 200 (Bullish macro)"}
        else:
            signals["TREND_LONG"] = {"signal": "SELL", "value": round(close, 2), "reason": "Price below EMA 200 (Bearish macro)"}

    return signals


def get_atr_levels(df: pd.DataFrame) -> Optional[dict]:
    """Extract ATR-based Stop Loss and Take Profit levels."""
    last = df.iloc[-1]
    atr = last.get("ATR")
    close = last["Close"]
    if atr is None or pd.isna(atr) or atr == 0:
        return None
    return {
        "ATR": round(atr, 4),
        "SL_1x": round(close - atr, 2),
        "TP_1x": round(close + atr, 2),
        "TP_1_5x": round(close + atr * 1.5, 2),
        "TP_2x": round(close + atr * 2, 2),
        "SL_1_5x": round(close - atr * 1.5, 2),
    }


def get_support_resistance(df: pd.DataFrame) -> dict:
    """Get recent support and resistance levels."""
    last = df.iloc[-1]
    return {
        "support": round(last.get("Support", last["Low"]), 2),
        "resistance": round(last.get("Resistance", last["High"]), 2),
        "close": round(last["Close"], 2),
    }


def compute_full_analysis(df: pd.DataFrame) -> dict:
    """Run full technical analysis and return structured results."""
    df = compute_all_indicators(df)
    signals = get_indicator_signals(df)
    atr = get_atr_levels(df)
    sr = get_support_resistance(df)
    return {
        "signals": signals,
        "atr": atr,
        "support_resistance": sr,
        "last_price": round(df.iloc[-1]["Close"], 4),
    }
