#!/usr/bin/env python3
"""
BlackMagicAI OMEGA — AI-Powered Telegram Trading Bot
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

from config import MARKETS, CATEGORIES, COMMANDS, TIMEFRAMES
from database import (
    SessionLocal, save_signal, get_track_record, update_signal_result,
    TradeJournal, AlertConfig, UserSettings
)
from market_data import fetch_ohlcv, get_current_price, get_live_ticker_text, get_multi_prices
from analysis import compute_full_analysis
from signals import generate_signal, format_signal_message, format_analysis_message
from ai_engine import (
    ai_analyze_market, ai_scan_markets, format_ai_signal_message,
    format_ai_scan_message, AI_AVAILABLE, PROVIDER as AI_PROVIDER,
    ai_confluence, ai_build_strategy, ai_sentiment, ai_detect_patterns,
    ai_psychology, ai_correlated_markets,
    format_confluence_message, format_strategy_message,
    format_sentiment_message, format_pattern_message,
    format_psychology_message, format_correlation_message,
    ai_commander_analyze, format_commander_message
)
from quant_engine import (
    monte_carlo_simulation, calculate_var, kelly_criterion,
    detect_volatility_regime, mean_reversion_test, calculate_ratios,
    full_quant_report, correlation_matrix,
    format_quant_report, format_var_report, format_mc_summary
)
from multi_llm import (
    multi_model_consensus, analyze_news_sentiment, macro_fundamental_analysis,
    format_consensus_message, format_news_sentiment_message, format_macro_message,
    get_available_models
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "@TradekhmerAI")

# -------------------- Keyboards --------------------

def main_keyboard():
    return ReplyKeyboardMarkup(
        [
            ["🔮 OMEGA Commander ⚡"],
            ["🧠 AI Analysis", "🔎 AI Scanner"],
            ["🔗 Confluence", "💭 Sentiment"],
            ["📐 Patterns", "📐 Strategy"],
            ["🧠 Psychology", "🔗 Correlation"],
            ["🏦 Quant Report", "🧠 Consensus"],
            ["📊 Markets", "💹 Live Price"],
            ["📋 Track Record", "📓 Journal"],
            ["⚖️ Risk Calc", "🔔 Alert"],
            ["📰 News", "🏛️ Macro"],
            ["❓ Help"]
        ],
        resize_keyboard=True
    )


def markets_inline_keyboard():
    buttons = []
    for cat, symbols in CATEGORIES.items():
        buttons.append([InlineKeyboardButton(f"📂 {cat}", callback_data=f"cat_{cat}")])
    buttons.append([InlineKeyboardButton("🔄 Refresh All Prices", callback_data="refresh_all")])
    return InlineKeyboardMarkup(buttons)


def symbol_list_keyboard(category: str):
    symbols = CATEGORIES.get(category, [])
    buttons = []
    for sym in symbols:
        m = MARKETS.get(sym)
        if m:
            buttons.append([InlineKeyboardButton(
                f"{m.emoji} {m.symbol} — {m.name}",
                callback_data=f"price_{sym}"
            )])
    buttons.append([InlineKeyboardButton("🔙 Back", callback_data="markets_menu")])
    return InlineKeyboardMarkup(buttons)


def signal_timeframe_keyboard():
    buttons = [
        [
            InlineKeyboardButton("⏱ 15m", callback_data="sig_tf_15m"),
            InlineKeyboardButton("⏱ 1H", callback_data="sig_tf_1h"),
        ],
        [
            InlineKeyboardButton("⏱ 4H", callback_data="sig_tf_4h"),
            InlineKeyboardButton("⏱ 1D", callback_data="sig_tf_1d"),
        ],
        [InlineKeyboardButton("🔙 Cancel", callback_data="cancel")],
    ]
    return InlineKeyboardMarkup(buttons)


def journal_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ បន្ថែម Add Trade", callback_data="journal_add")],
        [InlineKeyboardButton("📋 មើល View History", callback_data="journal_view")],
        [InlineKeyboardButton("🔙 Back", callback_data="cancel")],
    ])

# -------------------- Helper --------------------
def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        return None

# -------------------- Command Handlers --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome = f"""
🚀 *សួស្តី {user.first_name}! ស្វាគមន៍មកកាន់ BlackMagicAI OMEGA*

🔮 ខ្ញុំជា AI Trading Commander ជំនាន់ OMEGA!

🏦 *INSTITUTIONAL AI ENGINES*
├ 🎲 Monte Carlo Simulation (GBM)
├ ⚠️ Value at Risk (VaR + CVaR)
├ 💰 Kelly Criterion Position Sizing
├ 📈 Volatility Regime Detection
├ 🔄 Mean Reversion (Ornstein-Uhlenbeck)
├ 📐 Sharpe/Sortino/Calmar Ratios
├ 🧠 Multi-LLM Consensus (3+ AI models)
└ 📰 News Sentiment + Macro Analysis

⚡ *OMEGA Commander* — 8 AI Engines ក្នុងការវិភាគតែមួយ
├ 🔗 Multi-TF Confluence (15m/1H/4H/1D)
├ 💭 Market Sentiment Analysis
├ 📐 Chart Pattern Recognition
├ 📐 Personalized Strategy Builder
├ 🧠 Trading Psychology Coach
├ 🔗 Cross-Market Correlation
├ 📊 Full Technical Analysis (RSI/MACD/EMA/ADX/BB)
└ 🕐 Timing & Schedule Optimizer

📊 *មុខងារផ្សេងទៀត:*
│ 💹 Live Prices — Gold, Forex, Crypto, Indices
│ 📋 Track Record — តាមដាម Win Rate
│ 📓 Trade Journal — កំណត់ហេតុ
│ ⚖️ Risk Calculator — គណនា Risk/Reward
│ 🔔 Price Alerts — ជូនដំណឹងតម្លៃ

📌 ប្រើ /help សម្រាប់ពាក្យបញ្ជាទាំងអស់

⚠️ _For educational purposes only. Not financial advice._
"""
    await update.message.reply_text(welcome, parse_mode="Markdown", reply_markup=main_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ai_status = f"✅ AI Active ({AI_PROVIDER})" if AI_AVAILABLE else "⚠️ AI Not Configured (add API key to .env)"
    models = get_available_models()
    model_count = sum(len(v) for v in models.values()) + 1

    help_text = f"""*📖 ពាក្យបញ្ជា / COMMANDS*

🏦 *INSTITUTIONAL AI ({ai_status})*
/quant <symbol> — Monte Carlo, VaR, Kelly, Vol Regime, Mean Reversion
/consensus <symbol> — Multi-LLM Consensus ({model_count} AI models vote)
/news [symbol] — News Sentiment Analysis (AI)
/macro [symbol] — Macro-Fundamental Analysis (AI)

🔮 *OMEGA COMMANDER ({ai_status})*
/ai_complete <symbol> — 8 AI Engines វិភាគក្នុងពេលតែមួយ
/omega <symbol> — Alias for OMEGA Commander

🧠 *AI DEEP ANALYSIS*
/ai_signal <symbol> — AI វិភាគស៊ីជម្រៅដោយ LLM
/ai_scan — AI ស្កេនទីផ្សាររកឱកាសល្អបំផុត

🔗 *AI CONFLUENCE*
/ai_confluence <symbol> — វិភាគ 4 Timeframes (15m/1H/4H/1D)

💭 *AI SENTIMENT*
/ai_sentiment <symbol> — វិភាគមនោសញ្ចេតនាទីផ្សារ

📐 *AI PATTERNS*
/ai_pattern <symbol> — AI រក Chart Patterns ស្វ័យប្រវត្តិ

📐 *AI STRATEGY BUILDER*
/ai_strategy — AI បង្កើត Trading Strategy ផ្ទាល់ខ្លួន

🧠 *AI PSYCHOLOGY*
/ai_psychology [topic] — គ្រូបង្វឹកចិត្តសាស្ត្រ Trading

🔗 *AI CORRELATION*
/ai_correlate <symbol> — វិភាគទំនាក់ទំនងរវាងទីផ្សារ

🤖 *សញ្ញា & វិភាគ*
/signal <symbol> — សញ្ញា 8 Indicators
/analysis <symbol> — វិភាគបច្ចេកទេសពេញ
/price <symbol> — តម្លៃផ្សាយផ្ទាល់

📊 *ទីផ្សារ*
/markets — បញ្ជីទីផ្សារទាំងអស់
/scan — ស្កេនសញ្ញាគ្រប់ទីផ្សារ

📋 *Track Record & Journal*
/trackrecord — មើលកំណត់ត្រាឈ្នះ-ចាញ់
/journal — កំណត់ហេតុជួញដូរ

⚖️ *Risk & Alerts*
/risk <entry> <sl> <tp> — គណនា Risk/Reward
/alert <symbol> <above|below> <price> — ដំឡើង Alert

💡 ដាក់ GROQ_API_KEY ក្នុង .env ដើម្បីប្រើ AI
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")

# [ចំណាំ៖ កូដខាងក្រោមនេះគឺរក្សាទុក Command ផ្សេងៗដូចដើមទាំងអស់ (cmd_quant, cmd_markets, etc.) 
# គ្រាន់តែដកកូដណាដែលពាក់ព័ន្ធជាមួយ License ចេញតែប៉ុណ្ណោះ] 

async def cmd_quant(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not AI_AVAILABLE:
        await update.message.reply_text("⚠️ *AI មិនទាន់បានដំឡើងទេ!*\nដាក់ `GROQ_API_KEY` ក្នុង `.env` ហើយ restart bot ។", parse_mode="Markdown")
        return
    args = context.args
    if not args:
        buttons = []
        for cat, symbols in CATEGORIES.items():
            for sym in symbols:
                m = MARKETS.get(sym)
                if m:
                    buttons.append([InlineKeyboardButton(f"{m.emoji} {m.symbol}", callback_data=f"quant_{sym}")])
        buttons.append([InlineKeyboardButton("🔙 Cancel", callback_data="cancel")])
        await update.message.reply_text("🏦 *INSTITUTIONAL QUANT LAB*\nជ្រើសរើសទីផ្សារដើម្បីវិភាគ៖", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(buttons))
        return
    symbol = args[0].upper()
    if symbol not in MARKETS:
        await update.message.reply_text(f"❌ មិនស្គាល់ទីផ្សារ: {symbol}")
        return
    processing = await update.message.reply_text(f"🏦 *កំពុងដំណើរការ Quant Analysis សម្រាប់ {symbol}...*\nMonte Carlo • VaR • Kelly • Vol Regime • Mean Reversion ⏳", parse_mode="Markdown")
    try:
        df = fetch_ohlcv(symbol, interval="1h", period="30d")
        if df is None or df.empty or len(df) < 30:
            await processing.edit_text(f"❌ មិនអាចទាញទិន្នន័យ {symbol}")
            return
        prices = df["close"].tolist()
        report = full_quant_report(symbol, prices, position_value=10000.0)
        msg = format_quant_report(report)
        await processing.delete()
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await processing.edit_text(f"❌ មានបញ្ហាក្នុងការវិភាគ {symbol}\n`{str(e)[:200]}`", parse_mode="Markdown")

# (សូមបញ្ចូល Function ទាំងអស់ដែលសល់ពីកូដដើមរបស់អ្នក (cmd_consensus, cmd_news, cmd_macro, etc...) នៅទីនេះដដែល)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text == "🔮 OMEGA Commander ⚡":
        buttons = []
        for cat, symbols in CATEGORIES.items():
            row = []
            for sym in symbols:
                m = MARKETS.get(sym)
                if m:
                    row.append(InlineKeyboardButton(f"{m.emoji} {m.symbol}", callback_data=f"omega_{sym}"))
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)
        buttons.append([InlineKeyboardButton("🔙 Cancel", callback_data="cancel")])
        await update.message.reply_text(
            "╔══════════════════════╗\n║  🔮 *OMEGA COMMANDER*  ║\n╚══════════════════════╝\n\n*8 AI Engines — 1 Ultimate Analysis*\n\n✅ ជ្រើសរើសទីផ្សារដើម្បីចាប់ផ្តើម៖",
            parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(buttons))
    elif text == "📊 Markets":
        await update.message.reply_text("📊 *ទីផ្សារ / MARKETS*\nជ្រើសរើសប្រភេទទីផ្សារ៖", parse_mode="Markdown", reply_markup=markets_inline_keyboard())
    elif text == "❓ Help ជំនួយ" or text == "❓ Help":
        await cmd_help(update, context)
    # (ដាក់ if statements ផ្សេងទៀតដែលមិនជាប់ពាក់ព័ន្ធជាមួយ Buy License)
    else:
        upper = text.upper().strip()
        if upper in MARKETS:
            processing = await update.message.reply_text(f"🤖 កំពុងវិភាគ {upper}... ⏳")
            sig = generate_signal(upper, "1h")
            if sig:
                msg = format_signal_message(sig)
                await processing.delete()
                await update.message.reply_text(msg, parse_mode="Markdown")
            else:
                await processing.edit_text(f"❌ មិនអាចវិភាគ {upper}")
        else:
            await update.message.reply_text("សូមប្រើប៊ូតុងឬពាក្យបញ្ជា (/help សម្រាប់ជំនួយ)")


def main():
    if not BOT_TOKEN:
        print("❌ សូមដំឡើង TELEGRAM_BOT_TOKEN ក្នុង .env file ឬ ក្នុង Environment Variable របស់ Railway!")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    # (បញ្ចូល Handler ដទៃទៀតដែលសល់ដូចដើម... សូមកុំដាក់ Handler របស់ផ្នែក License)
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 BlackMagicAI Trading Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
