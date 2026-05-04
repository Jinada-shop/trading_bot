import os
import logging
import json
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import ccxt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERS = [int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()]

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# ========== 30 ТОРГОВЫХ ПАР ==========
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "LTC/USDT", "NEAR/USDT", "ATOM/USDT", "UNI/USDT",
    "OP/USDT", "ARB/USDT", "APT/USDT", "SUI/USDT", "INJ/USDT",
    "FET/USDT", "RNDR/USDT", "WLD/USDT", "JUP/USDT", "ENA/USDT",
    "PEPE/USDT", "WIF/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT"
]

MIN_CONFIDENCE = 0.60
CHECK_INTERVAL = 60
REPORT_INTERVAL = 600

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Подключение к Binance
if BINANCE_API_KEY and BINANCE_API_SECRET:
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    print(f"✅ Binance подключён | {len(SYMBOLS)} пар")
else:
    exchange = ccxt.binance({'enableRateLimit': True})
    print(f"⚙️ Binance: только сигналы | {len(SYMBOLS)} пар")

# ========== ДАННЫЕ ==========
user_balance = {}
user_auto = {}
user_trades = {}
user_messages = {}
signal_history = defaultdict(list)

def load_data():
    for f, d in [("balance.json", user_balance), ("auto.json", user_auto), ("trades.json", user_trades)]:
        if os.path.exists(f):
            with open(f, "r") as file:
                d.update(json.load(file))

def save_data():
    with open("balance.json", "w") as f: json.dump(user_balance, f)
    with open("auto.json", "w") as f: json.dump(user_auto, f)
    with open("trades.json", "w") as f: json.dump(user_trades, f)

def get_balance(uid): return user_balance.get(str(uid), 500)
def set_balance(uid, amt): user_balance[str(uid)] = max(0, amt); save_data()
def add_balance(uid, amt): set_balance(uid, get_balance(uid) + amt)
def sub_balance(uid, amt):
    if get_balance(uid) >= amt:
        set_balance(uid, get_balance(uid) - amt)
        return True
    return False

def get_auto(uid):
    uid = str(uid)
    if uid not in user_auto:
        user_auto[uid] = {
            'enabled': False, 'pos_size': 20, 'stop_loss': 5, 'take_profit': 10,
            'trades_today': 0, 'daily_loss': 0, 'last_date': datetime.now().strftime("%Y-%m-%d")
        }
    return user_auto[uid]

# ========== ПАНЕЛЬ ==========
def get_panel(uid):
    auto = get_auto(uid)
    balance = get_balance(uid)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💰 {balance:.0f} USDT", callback_data="show_balance"),
         InlineKeyboardButton(f"🔘 АВТО: {'✅' if auto['enabled'] else '❌'}", callback_data="toggle")],
        [InlineKeyboardButton("📈 СИГНАЛ ПО ВСЕМ", callback_data="all_signals")],
        [InlineKeyboardButton("🏆 ТОП 5 (ПОКУПКА)", callback_data="top_buy"),
         InlineKeyboardButton("🏆 ТОП 5 (ПРОДАЖА)", callback_data="top_sell")],
        [InlineKeyboardButton("🧠 ОБУЧИТЬ", callback_data="train"),
         InlineKeyboardButton("⚙️ НАСТРОЙКИ", callback_data="settings")],
        [InlineKeyboardButton("📊 СТАТИСТИКА", callback_data="stats"),
         InlineKeyboardButton("📖 ПОМОЩЬ", callback_data="help")],
    ])

def get_settings_panel(uid):
    auto = get_auto(uid)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💰 Бюджет: ${get_balance(uid):.0f}", callback_data="set_budget")],
        [InlineKeyboardButton(f"📊 Позиция: ${auto['pos_size']:.0f}", callback_data="set_pos")],
        [InlineKeyboardButton(f"🛡️ Стоп: {auto['stop_loss']}%", callback_data="set_sl"),
         InlineKeyboardButton(f"🎯 Профит: {auto['take_profit']}%", callback_data="set_tp")],
        [InlineKeyboardButton("◀️ НАЗАД", callback_data="main")],
    ])

async def safe_edit(chat_id, context, text, panel):
    try:
        if str(chat_id) in user_messages:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=user_messages[str(chat_id)])
            except:
                pass
        msg = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=panel, parse_mode='Markdown')
        user_messages[str(chat_id)] = msg.message_id
    except Exception as e:
        logger.error(f"Ошибка отправки: {e}")

# ========== АНАЛИЗАТОР ==========
class Analyzer:
    @staticmethod
    def rsi(prices, period=14):
        if len(prices) < period + 1: return 50
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        ag = np.mean(gain[-period:])
        al = np.mean(loss[-period:])
        if al == 0: return 100
        return 100 - (100 / (1 + ag / al))
    
    @staticmethod
    def extract_features(df):
        closes = df['close'].values
        features = []
        for i in range(50, len(df)):
            w = closes[i-50:i]
            r = Analyzer.rsi(w, 14)
            s7 = np.mean(w[-7:])
            s25 = np.mean(w[-25:])
            pc = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
            vol = np.std(w[-20:]) / np.mean(w[-20:]) if np.mean(w[-20:]) > 0 else 0.01
            trend = (s7 / s25 - 1) if s25 > 0 else 0
            features.append([r/100, trend, pc, vol])
        return np.array(features) if len(features) > 50 else None

# ========== МОДЕЛЬ ==========
class TradingModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.load()
    
    def train(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        results = {}
        total = len(symbols)
        for idx, sym in enumerate(symbols):
            try:
                logger.info(f"Обучение {idx+1}/{total}: {sym}")
                ohlcv = exchange.fetch_ohlcv(sym, "1h", limit=300)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                feat = Analyzer.extract_features(df)
                if feat is None: 
                    results[sym] = {"error": "Недостаточно данных"}
                    continue
                closes = df['close'].values
                X = feat[:-6]
                y = [1 if closes[i+6+50] > closes[i+50] else 0 for i in range(len(feat)-6)]
                y = np.array(y)
                X = X[:len(y)]
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
                model.fit(Xs, y)
                acc = accuracy_score(y, model.predict(Xs))
                self.models[sym] = model
                self.scalers[sym] = scaler
                results[sym] = {"accuracy": acc, "samples": len(X)}
                logger.info(f"✅ {sym}: {acc:.1%}")
            except Exception as e:
                results[sym] = {"error": str(e)}
        self.is_trained = True
        self.save()
        return results
    
    def predict(self, symbol):
        if not self.is_trained or symbol not in self.models:
            return None, 0.5, "⚪"
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            feat = Analyzer.extract_features(df)
            if feat is None or len(feat) == 0:
                return None, 0.5, "⚪"
            Xc = feat[-1:].reshape(1, -1)
            Xs = self.scalers[symbol].transform(Xc)
            proba = self.models[symbol].predict_proba(Xs)[0]
            pred = self.models[symbol].predict(Xs)[0]
            conf = max(proba)
            if conf > 0.75: strength = "🟢"
            elif conf > 0.65: strength = "🟡"
            else: strength = "⚪"
            return pred, conf, strength
        except:
            return None, 0.5, "⚪"
    
    def predict_all(self):
        results = []
        for sym in SYMBOLS:
            pred, conf, strength = self.predict(sym)
            if pred is not None and conf > MIN_CONFIDENCE:
                name = sym.replace("/USDT", "")
                results.append({
                    'symbol': name,
                    'action': '🟢 ПОКУПАТЬ' if pred == 1 else '🔴 ПРОДАВАТЬ',
                    'confidence': conf,
                    'strength': strength
                })
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def save(self):
        for sym, model in self.models.items():
            safe = sym.replace("/", "_")
            joblib.dump((model, self.scalers[sym]), f"model_{safe}.pkl")
    
    def load(self):
        for sym in SYMBOLS:
            safe = sym.replace("/", "_")
            if os.path.exists(f"model_{safe}.pkl"):
                try:
                    data = joblib.load(f"model_{safe}.pkl")
                    if isinstance(data, tuple) and len(data) == 2:
                        self.models[sym], self.scalers[sym] = data
                    self.is_trained = True
                except:
                    pass
        logger.info(f"Загружено моделей: {len(self.models)}/{len(SYMBOLS)}")

model = TradingModel()

# ========== СИГНАЛЫ ==========
async def send_signal(uid, context, symbol, pred, conf, price):
    name = symbol.replace("/USDT", "")
    if pred == 1:
        text = f"🚨 *{name}* — 🟢 **ПОКУПАЙ!**\n💰 ${price:,.0f}\n🎯 Уверенность: {conf:.0%}"
    else:
        text = f"🚨 *{name}* — 🔴 **ПРОДАВАЙ!**\n💰 ${price:,.0f}\n🎯 Уверенность: {conf:.0%}"
    await context.bot.send_message(chat_id=uid, text=text, parse_mode='Markdown')

async def check_markets(context):
    if not model.is_trained:
        return
    
    for uid in ALLOWED_USERS:
        best_signals = []
        for sym in SYMBOLS:
            try:
                pred, conf, strength = model.predict(sym)
                if pred is not None and conf > MIN_CONFIDENCE:
                    ticker = exchange.fetch_ticker(sym)
                    name = sym.replace("/USDT", "")
                    best_signals.append({
                        'symbol': name,
                        'pred': pred,
                        'conf': conf,
                        'price': ticker['last']
                    })
                    signal_history[name].append({
                        'timestamp': datetime.now(),
                        'pred': pred,
                        'conf': conf,
                        'price': ticker['last']
                    })
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.error(f"Ошибка {sym}: {e}")
        
        best_signals.sort(key=lambda x: x['conf'], reverse=True)
        for signal in best_signals[:3]:
            await send_signal(uid, context, signal['symbol'], signal['pred'], signal['conf'], signal['price'])
            await asyncio.sleep(1)

async def send_report(context):
    if not model.is_trained:
        return
    
    now = datetime.now()
    ten_min_ago = now - timedelta(minutes=10)
    
    pair_stats = []
    for symbol, history in signal_history.items():
        recent = [h for h in history if h['timestamp'] > ten_min_ago]
        if not recent:
            continue
        signal_count = len(recent)
        max_conf = max([h['conf'] for h in recent])
        avg_conf = sum([h['conf'] for h in recent]) / signal_count
        buy_count = sum(1 for h in recent if h['pred'] == 1)
        sell_count = signal_count - buy_count
        pair_stats.append({
            'symbol': symbol,
            'signal_count': signal_count,
            'max_conf': max_conf,
            'avg_conf': avg_conf,
            'buy_count': buy_count,
            'sell_count': sell_count,
        })
    
    report = f"📊 *ОТЧЁТ ЗА 10 МИНУТ* ({now.strftime('%H:%M')})\n\n"
    
    if pair_stats:
        pair_stats.sort(key=lambda x: x['signal_count'], reverse=True)
        top_active = pair_stats[:5]
        report += "🔥 *САМЫЕ АКТИВНЫЕ ПАРЫ:*\n"
        for i, p in enumerate(top_active):
            report += f"{i+1}. *{p['symbol']}* — {p['signal_count']} сигн"
            if p['buy_count'] > p['sell_count']:
                report += f" (🟢 покупка {p['buy_count']}/{p['signal_count']})\n"
            else:
                report += f" (🔴 продажа {p['sell_count']}/{p['signal_count']})\n"
        
        pair_stats.sort(key=lambda x: x['max_conf'], reverse=True)
        top_confident = pair_stats[:5]
        report += "\n🎯 *САМАЯ ВЫСОКАЯ УВЕРЕННОСТЬ:*\n"
        for i, p in enumerate(top_confident):
            emoji = "🟢" if p['buy_count'] > p['sell_count'] else "🔴"
            report += f"{i+1}. *{p['symbol']}* — {emoji} {p['max_conf']:.0%} (средняя {p['avg_conf']:.0%})\n"
    else:
        report += "⚠️ *ЗА 10 МИНУТ НЕТ СИГНАЛОВ*\n"
    
    report += "\n💰 *ТЕКУЩИЕ ЦЕНЫ:*\n"
    try:
        btc = exchange.fetch_ticker("BTC/USDT")
        eth = exchange.fetch_ticker("ETH/USDT")
        report += f"₿ BTC: ${btc['last']:,.0f}\n"
        report += f"⟠ ETH: ${eth['last']:,.0f}\n"
    except:
        pass
    
    for uid in ALLOWED_USERS:
        await context.bot.send_message(chat_id=uid, text=report, parse_mode='Markdown')
    
    for symbol in signal_history:
        signal_history[symbol] = [h for h in signal_history[symbol] if h['timestamp'] > now - timedelta(minutes=30)]

async def all_signals(uid, context):
    if not model.is_trained:
        await safe_edit(uid, context, "❌ Сначала обучите модель!", get_panel(uid))
        return
    all_sig = model.predict_all()
    if not all_sig:
        await safe_edit(uid, context, "📊 *НЕТ СИГНАЛОВ*", get_panel(uid))
        return
    text = "📊 *ВСЕ СИГНАЛЫ*\n\n"
    for sig in all_sig[:15]:
        text += f"{sig['strength']} {sig['symbol']}: {sig['action']} ({sig['confidence']:.0%})\n"
    await safe_edit(uid, context, text, get_panel(uid))

async def top_buy_signals(uid, context):
    if not model.is_trained:
        await safe_edit(uid, context, "❌ Сначала обучите модель!", get_panel(uid))
        return
    all_sig = model.predict_all()
    buy = [s for s in all_sig if "ПОКУПАТЬ" in s['action']]
    if not buy:
        await safe_edit(uid, context, "📊 *НЕТ СИГНАЛОВ*", get_panel(uid))
        return
    text = "🏆 *ТОП-5 ПОКУПКА*\n\n"
    for i, sig in enumerate(buy[:5]):
        text += f"{i+1}. *{sig['symbol']}*: {sig['confidence']:.0%}\n"
    await safe_edit(uid, context, text, get_panel(uid))

async def top_sell_signals(uid, context):
    if not model.is_trained:
        await safe_edit(uid, context, "❌ Сначала обучите модель!", get_panel(uid))
        return
    all_sig = model.predict_all()
    sell = [s for s in all_sig if "ПРОДАВАТЬ" in s['action']]
    if not sell:
        await safe_edit(uid, context, "📊 *НЕТ СИГНАЛОВ*", get_panel(uid))
        return
    text = "🏆 *ТОП-5 ПРОДАЖА*\n\n"
    for i, sig in enumerate(sell[:5]):
        text += f"{i+1}. *{sig['symbol']}*: {sig['confidence']:.0%}\n"
    await safe_edit(uid, context, text, get_panel(uid))

# ========== КОМАНДЫ ==========
async def start(update, context):
    uid = update.effective_user.id
    if uid not in ALLOWED_USERS:
        await update.message.reply_text("❌ Доступ запрещён")
        return
    if get_balance(uid) == 500:
        set_balance(uid, 500)
    await safe_edit(uid, context,
        f"🧠 *УМНЫЙ ТОРГОВЫЙ БОТ*\n\n"
        f"💰 Баланс: ${get_balance(uid):.0f}\n"
        f"📊 Моделей: {len(model.models)}/{len(SYMBOLS)}\n"
        f"⏱️ Сигналы: лучшие 3 каждую минуту\n"
        f"📈 Отчёт: каждые 10 минут\n\n"
        f"👇 Выберите действие:",
        get_panel(uid))

async def handler(update, context):
    query = update.callback_query
    try:
        await query.answer()
    except:
        return
    
    uid = update.effective_chat.id
    data = query.data
    
    if data == "main":
        await safe_edit(uid, context, f"🧠 *МЕНЮ*\n💰 Баланс: ${get_balance(uid):.0f}", get_panel(uid))
    elif data == "all_signals":
        await all_signals(uid, context)
    elif data == "top_buy":
        await top_buy_signals(uid, context)
    elif data == "top_sell":
        await top_sell_signals(uid, context)
    elif data == "toggle":
        auto = get_auto(uid)
        auto['enabled'] = not auto['enabled']
        save_data()
        await safe_edit(uid, context, f"✅ Автоторговля {'ВКЛ' if auto['enabled'] else 'ВЫКЛ'}", get_panel(uid))
    elif data == "train":
        await safe_edit(uid, context, "🧠 ОБУЧЕНИЕ... 3-5 минут", get_panel(uid))
        res = model.train(SYMBOLS)
        trained = len([r for r in res.values() if "accuracy" in r])
        await safe_edit(uid, context, f"✅ ОБУЧЕНО {trained}/{len(SYMBOLS)} ПАР!", get_panel(uid))
    elif data == "settings":
        await safe_edit(uid, context, "⚙️ *НАСТРОЙКИ*", get_settings_panel(uid))
    elif data == "set_budget":
        await safe_edit(uid, context, "💰 Введите бюджет (50-10000):", get_settings_panel(uid))
        context.user_data['set_budget'] = uid
    elif data == "set_pos":
        await safe_edit(uid, context, "📊 Размер позиции (10-500):", get_settings_panel(uid))
        context.user_data['set_pos'] = uid
    elif data == "set_sl":
        await safe_edit(uid, context, "🛡️ Стоп-лосс % (2-10):", get_settings_panel(uid))
        context.user_data['set_sl'] = uid
    elif data == "set_tp":
        await safe_edit(uid, context, "🎯 Тейк-профит % (4-20):", get_settings_panel(uid))
        context.user_data['set_tp'] = uid
    elif data == "stats":
        bal = get_balance(uid)
        auto = get_auto(uid)
        await safe_edit(uid, context,
            f"📊 *СТАТИСТИКА*\n💰 Баланс: ${bal:.0f}\n📈 Сделок сегодня: {auto['trades_today']}\n🛡️ Стоп: {auto['stop_loss']}%\n🎯 Профит: {auto['take_profit']}%",
            get_panel(uid))
    elif data == "show_balance":
        try:
            await query.answer(f"Баланс: ${get_balance(uid):.0f}")
        except:
            pass
    elif data == "help":
        await safe_edit(uid, context, "📖 *ИНСТРУКЦИЯ*\n\n1️⃣ ОБУЧИТЬ (1 раз)\n2️⃣ СИГНАЛ ПО ВСЕМ\n3️⃣ ТОП 5\n\n📡 Сигналы каждую минуту!", get_panel(uid))

async def text_input(update, context):
    uid = update.effective_user.id
    text = update.message.text.strip()
    
    if context.user_data.get('set_budget') == uid:
        try:
            v = float(text)
            if 50 <= v <= 10000:
                set_balance(uid, v)
                await update.message.reply_text(f"✅ Бюджет: ${v:.0f}")
            else:
                await update.message.reply_text("❌ От 50 до 10000")
        except:
            await update.message.reply_text("❌ Введите число")
        context.user_data.pop('set_budget', None)
        await safe_edit(uid, context, f"🧠 *МЕНЮ*\n💰 Баланс: ${get_balance(uid):.0f}", get_panel(uid))
    
    elif context.user_data.get('set_pos') == uid:
        try:
            v = float(text)
            if 10 <= v <= 500:
                get_auto(uid)['pos_size'] = v
                save_data()
                await update.message.reply_text(f"✅ Позиция: ${v:.0f}")
            else:
                await update.message.reply_text("❌ От 10 до 500")
        except:
            await update.message.reply_text("❌ Введите число")
        context.user_data.pop('set_pos', None)
        await safe_edit(uid, context, f"🧠 *МЕНЮ*\n💰 Баланс: ${get_balance(uid):.0f}", get_panel(uid))
    
    elif context.user_data.get('set_sl') == uid:
        try:
            v = float(text)
            if 2 <= v <= 10:
                get_auto(uid)['stop_loss'] = v
                save_data()
                await update.message.reply_text(f"✅ Стоп-лосс: {v}%")
            else:
                await update.message.reply_text("❌ От 2 до 10")
        except:
            await update.message.reply_text("❌ Введите число")
        context.user_data.pop('set_sl', None)
        await safe_edit(uid, context, f"🧠 *МЕНЮ*\n💰 Баланс: ${get_balance(uid):.0f}", get_panel(uid))
    
    elif context.user_data.get('set_tp') == uid:
        try:
            v = float(text)
            if 4 <= v <= 20:
                get_auto(uid)['take_profit'] = v
                save_data()
                await update.message.reply_text(f"✅ Тейк-профит: {v}%")
            else:
                await update.message.reply_text("❌ От 4 до 20")
        except:
            await update.message.reply_text("❌ Введите число")
        context.user_data.pop('set_tp', None)
        await safe_edit(uid, context, f"🧠 *МЕНЮ*\n💰 Баланс: ${get_balance(uid):.0f}", get_panel(uid))

def main():
    load_data()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input))
    
    if app.job_queue:
        app.job_queue.run_repeating(check_markets, interval=CHECK_INTERVAL, first=10)
        app.job_queue.run_repeating(send_report, interval=REPORT_INTERVAL, first=15)
    
    print("="*70)
    print(f"🧠 БОТ ЗАПУЩЕН | {len(SYMBOLS)} пар")
    print("="*70)
    app.run_polling()

if __name__ == "__main__":
    main()
