import os
import logging
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import ccxt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERS = [int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()]

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME = "1h"
MIN_CONFIDENCE = 0.6
CHECK_INTERVAL = 60

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

# ========== БАЛАНС ==========
user_balances = {}

def get_balance(user_id):
    return user_balances.get(str(user_id), 1000)

def set_balance(user_id, amount):
    user_balances[str(user_id)] = amount

def subtract_balance(user_id, amount):
    current = get_balance(user_id)
    if current >= amount:
        set_balance(user_id, current - amount)
        return True
    return False

def add_balance(user_id, amount):
    set_balance(user_id, get_balance(user_id) + amount)
    return get_balance(user_id)

# ========== АВТОТОРГОВЛЯ ==========
autotrade_settings = {}

def get_autotrade(user_id):
    uid = str(user_id)
    if uid not in autotrade_settings:
        autotrade_settings[uid] = {
            'enabled': False, 'mode': 'signal_only', 'max_trades_per_day': 5,
            'max_daily_loss': 10, 'cooldown_minutes': 30, 'trades_today': 0, 'daily_loss': 0
        }
    return autotrade_settings[uid]

# ========== РЕКОМЕНДАЦИИ ==========
def get_budget_recommendations(balance_usdt):
    if balance_usdt < 100:
        return {
            'level': '🟢 Начинающий', 'position_percent': 15, 'max_position_usdt': balance_usdt * 0.15,
            'stop_loss_percent': 7, 'take_profit_percent': 12, 'recommended_symbols': ['BTC/USDT'],
            'warning': '⚠️ Не рискуйте больше чем готовы потерять!'
        }
    elif balance_usdt < 500:
        return {
            'level': '🟡 Средний', 'position_percent': 8, 'max_position_usdt': balance_usdt * 0.08,
            'stop_loss_percent': 6, 'take_profit_percent': 12, 'recommended_symbols': ['BTC/USDT', 'ETH/USDT'],
            'warning': '✅ Не забывайте про стоп-лоссы'
        }
    elif balance_usdt < 2000:
        return {
            'level': '🟠 Продвинутый', 'position_percent': 5, 'max_position_usdt': balance_usdt * 0.05,
            'stop_loss_percent': 5, 'take_profit_percent': 10, 'recommended_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'warning': '📊 Используйте автоторговлю с осторожностью'
        }
    else:
        return {
            'level': '🔴 Профессиональный', 'position_percent': 3, 'max_position_usdt': balance_usdt * 0.03,
            'stop_loss_percent': 4, 'take_profit_percent': 8, 'recommended_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'],
            'warning': '🎯 Используйте трейлинг стоп'
        }

# ========== КЛАСС МОДЕЛИ ==========
class TradingModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.load()
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def get_features(self, closes):
        features = []
        for i in range(50, len(closes)):
            window = closes[i-50:i]
            rsi = self.calculate_rsi(window)
            sma7 = np.mean(window[-7:])
            sma25 = np.mean(window[-25:])
            price_change = (window[-1] - window[-2]) / window[-2] if len(window) > 1 else 0
            volatility = np.std(window[-20:]) / np.mean(window[-20:]) if np.mean(window[-20:]) > 0 else 0.01
            features.append([rsi, sma7/sma25 - 1, price_change, volatility, window[-1]])
        return features
    
    def prepare_data(self, symbol, limit=600):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            closes = [c[4] for c in ohlcv]
            features = self.get_features(closes)
            if len(features) < 100:
                return None, None
            X, y = [], []
            for i in range(len(features) - 6):
                X.append(features[i][:4])
                y.append(1 if features[i+6][4] > features[i][4] else 0)
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Ошибка {symbol}: {e}")
            return None, None
    
    def train(self, symbol=None):
        syms = [symbol] if symbol else SYMBOLS
        results = {}
        for sym in syms:
            X, y = self.prepare_data(sym)
            if X is None or len(X) < 50:
                results[sym] = {"error": "Недостаточно данных"}
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")
            model.fit(Xs, y)
            acc = np.mean(model.predict(Xs) == y)
            self.models[sym] = model
            self.scalers[sym] = scaler
            results[sym] = {"accuracy": acc, "samples": len(X)}
            logger.info(f"{sym}: {acc:.2%}")
        self.is_trained = True
        self.save()
        return results
    
    def predict(self, symbol):
        if not self.is_trained or symbol not in self.models:
            return None, 0.5, {}
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            closes = [c[4] for c in ohlcv]
            rsi = self.calculate_rsi(closes[-50:])
            sma7 = np.mean(closes[-7:])
            sma25 = np.mean(closes[-25:])
            price_change = (closes[-1] - closes[-2]) / closes[-2]
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) if np.mean(closes[-20:]) > 0 else 0.01
            features = np.array([[rsi, sma7/sma25 - 1, price_change, volatility]])
            Xs = self.scalers[symbol].transform(features)
            proba = self.models[symbol].predict_proba(Xs)[0]
            pred = self.models[symbol].predict(Xs)[0]
            levels = {'stop_loss': closes[-1] * 0.95, 'take_profit': closes[-1] * 1.10}
            risk = "🟢 Низкий" if rsi < 70 and rsi > 30 else "🟡 Средний" if rsi < 80 and rsi > 20 else "🔴 Высокий"
            return pred, max(proba), {'rsi': rsi, 'levels': levels, 'risk': risk}
        except Exception as e:
            logger.error(f"Ошибка {symbol}: {e}")
            return None, 0.5, {}
    
    def save(self):
        for sym, model in self.models.items():
            safe = sym.replace("/", "_")
            joblib.dump(model, f"model_{safe}.pkl")
            joblib.dump(self.scalers[sym], f"scaler_{safe}.pkl")
    
    def load(self):
        for sym in SYMBOLS:
            safe = sym.replace("/", "_")
            if os.path.exists(f"model_{safe}.pkl"):
                self.models[sym] = joblib.load(f"model_{safe}.pkl")
                self.scalers[sym] = joblib.load(f"scaler_{safe}.pkl")
                self.is_trained = True
                logger.info(f"Загружена {sym}")

model = TradingModel()

# ========== КЛАВИАТУРЫ ==========
def get_main_keyboard(user_id):
    balance = get_balance(user_id)
    auto = get_autotrade(user_id)
    auto_status = "✅ Вкл" if auto['enabled'] else "❌ Выкл"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💰 Баланс: ${balance:.2f}", callback_data="balance")],
        [InlineKeyboardButton("📊 Рекомендации", callback_data="budget")],
        [InlineKeyboardButton("📈 BTC", callback_data="signal_BTC/USDT"), InlineKeyboardButton("📈 ETH", callback_data="signal_ETH/USDT")],
        [InlineKeyboardButton("📈 SOL", callback_data="signal_SOL/USDT"), InlineKeyboardButton("📈 BNB", callback_data="signal_BNB/USDT")],
        [InlineKeyboardButton("🧠 Обучить", callback_data="train")],
        [InlineKeyboardButton(f"🤖 Автоторговля: {auto_status}", callback_data="autotrade")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"), InlineKeyboardButton("💬 Советы", callback_data="tips")],
        [InlineKeyboardButton("➕ +100", callback_data="deposit_100"), InlineKeyboardButton("➖ -100", callback_data="withdraw_100")],
    ])

def get_autotrade_keyboard(user_id):
    at = get_autotrade(user_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{'✅' if at['enabled'] else '❌'} Вкл/Выкл", callback_data="at_toggle")],
        [InlineKeyboardButton(f"📊 Лимит сделок: {at['max_trades_per_day']}", callback_data="at_limit_trades")],
        [InlineKeyboardButton(f"📉 Лимит убытка: {at['max_daily_loss']}%", callback_data="at_limit_loss")],
        [InlineKeyboardButton("◀️ Назад", callback_data="back")],
    ])

# ========== ОТПРАВКА СИГНАЛОВ ==========
async def send_signal_to_user(context, user_id, symbol, prediction, confidence, info, price):
    balance = get_balance(user_id)
    rec = get_budget_recommendations(balance)
    
    if prediction == 1 and confidence > MIN_CONFIDENCE:
        text = f"🚨 *{symbol} — ПОКУПКА!*\n💰 ${price:,.0f}\n🎯 Уверенность: {confidence:.1%}\n🛑 Стоп: ${info.get('levels', {}).get('stop_loss', 0):,.0f}\n🎯 Профит: ${info.get('levels', {}).get('take_profit', 0):,.0f}\n💡 Входите ${rec['max_position_usdt']:.2f}"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🟢 Купить", callback_data=f"buy_{symbol}_{price}")]])
    elif prediction == 0 and confidence > MIN_CONFIDENCE:
        text = f"🚨 *{symbol} — ПРОДАЖА!*\n💰 ${price:,.0f}\n🎯 Уверенность: {confidence:.1%}"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔴 Продать", callback_data=f"sell_{symbol}_{price}")]])
    else:
        text = f"⏸️ *{symbol} — НЕТ СИГНАЛА*\n💰 ${price:,.0f}"
        kb = None
    await context.bot.send_message(chat_id=user_id, text=text, reply_markup=kb, parse_mode='Markdown')

# ========== ИСПОЛНЕНИЕ ==========
async def execute_buy(context, user_id, symbol, price):
    rec = get_budget_recommendations(get_balance(user_id))
    if subtract_balance(user_id, rec['max_position_usdt']):
        await context.bot.send_message(chat_id=user_id, text=f"✅ КУПЛЕНО!\n💰 -${rec['max_position_usdt']:.2f}\n💰 Новый баланс: ${get_balance(user_id):.2f}", parse_mode='Markdown')
    else:
        await context.bot.send_message(chat_id=user_id, text="❌ Недостаточно средств!")

async def execute_sell(context, user_id, symbol, price):
    await context.bot.send_message(chat_id=user_id, text=f"✅ ПРОДАНО!\n💰 Баланс: ${get_balance(user_id):.2f}", parse_mode='Markdown')

# ========== ФОНОВАЯ ПРОВЕРКА ==========
async def periodic_check(context: ContextTypes.DEFAULT_TYPE):
    if not model.is_trained:
        return
    for sym in SYMBOLS:
        try:
            ticker = exchange.fetch_ticker(sym)
            pred, conf, info = model.predict(sym)
            for uid in ALLOWED_USERS:
                await send_signal_to_user(context, uid, sym, pred, conf, info, ticker['last'])
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Ошибка {sym}: {e}")

# ========== ОБРАБОТЧИКИ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ALLOWED_USERS:
        await update.message.reply_text("❌ Доступ запрещён")
        return
    if str(uid) not in user_balances:
        user_balances[str(uid)] = 1000
    await update.message.reply_text("🤖 *ТОРГОВЫЙ БОТ*", reply_markup=get_main_keyboard(uid), parse_mode='Markdown')

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = update.effective_user.id
    if uid not in ALLOWED_USERS:
        await query.edit_message_text("❌ Нет доступа")
        return
    data = query.data
    
    if data.startswith("buy_"):
        parts = data.split("_")
        symbol = parts[1] + "/" + parts[2]
        price = float(parts[3])
        await execute_buy(context, uid, symbol, price)
        await start(update, context)
    elif data.startswith("sell_"):
        parts = data.split("_")
        symbol = parts[1] + "/" + parts[2]
        price = float(parts[3])
        await execute_sell(context, uid, symbol, price)
        await start(update, context)
    elif data.startswith("signal_"):
        sym = data.split("_")[1].replace("-", "/")
        if not model.is_trained:
            await query.edit_message_text("❌ Сначала обучите модель!")
            return
        ticker = exchange.fetch_ticker(sym)
        pred, conf, info = model.predict(sym)
        await send_signal_to_user(context, uid, sym, pred, conf, info, ticker['last'])
        await start(update, context)
    elif data == "budget":
        balance = get_balance(uid)
        rec = get_budget_recommendations(balance)
        text = f"📊 *БЮДЖЕТ*\n💰 ${balance:.2f}\n📈 {rec['level']}\n💡 Входите ${rec['max_position_usdt']:.2f}\n🛑 Стоп: -{rec['stop_loss_percent']}%\n🎯 Профит: +{rec['take_profit_percent']}%"
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    elif data == "train":
        await query.edit_message_text("🧠 Обучение... 2-3 минуты")
        res = model.train()
        text = "✅ Обучено!\n" + "\n".join([f"{sym}: {r.get('accuracy', r.get('error'))}" for sym, r in res.items()])
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    elif data == "autotrade":
        await query.edit_message_text("⚙️ *АВТОТОРГОВЛЯ*", reply_markup=get_autotrade_keyboard(uid), parse_mode='Markdown')
    elif data == "at_toggle":
        at = get_autotrade(uid)
        at['enabled'] = not at['enabled']
        await query.edit_message_text(f"✅ Автоторговля {'вкл' if at['enabled'] else 'выкл'}")
        await button_handler(update, context)
    elif data == "at_limit_trades":
        await query.edit_message_text("💰 Введите лимит сделок (1-20):")
        context.user_data['waiting_trades_limit'] = uid
    elif data == "at_limit_loss":
        await query.edit_message_text("📉 Введите лимит убытка % (1-50):")
        context.user_data['waiting_loss_limit'] = uid
    elif data == "deposit_100":
        add_balance(uid, 100)
        await query.edit_message_text(f"✅ Пополнено! Баланс: ${get_balance(uid):.2f}")
        await start(update, context)
    elif data == "withdraw_100":
        if subtract_balance(uid, 100):
            await query.edit_message_text(f"✅ Выведено! Баланс: ${get_balance(uid):.2f}")
        else:
            await query.edit_message_text("❌ Недостаточно средств!")
        await start(update, context)
    elif data == "balance":
        await query.edit_message_text(f"💰 Баланс: ${get_balance(uid):.2f}", parse_mode='Markdown')
        await start(update, context)
    elif data == "stats":
        at = get_autotrade(uid)
        await query.edit_message_text(f"📊 Статистика\n🎯 Модель: {'✅' if model.is_trained else '❌'}\n💰 ${get_balance(uid):.2f}\n🤖 Автоторговля: {'✅' if at['enabled'] else '❌'}", parse_mode='Markdown')
        await start(update, context)
    elif data == "tips":
        await query.edit_message_text("📖 *СОВЕТЫ*\n• Входите 2-5% от депозита\n• Всегда стоп-лосс\n• Переобучайте модель раз в сутки", parse_mode='Markdown')
        await start(update, context)
    elif data == "back":
        await start(update, context)

async def handle_message(update, context):
    uid = update.effective_user.id
    if context.user_data.get('waiting_trades_limit') == uid:
        try:
            val = int(update.message.text)
            if 1 <= val <= 20:
                get_autotrade(uid)['max_trades_per_day'] = val
                await update.message.reply_text(f"✅ Лимит: {val}")
            else:
                await update.message.reply_text("❌ 1-20")
        except:
            await update.message.reply_text("❌ Число")
        del context.user_data['waiting_trades_limit']
    elif context.user_data.get('waiting_loss_limit') == uid:
        try:
            val = float(update.message.text)
            if 1 <= val <= 50:
                get_autotrade(uid)['max_daily_loss'] = val
                await update.message.reply_text(f"✅ Лимит: {val}%")
            else:
                await update.message.reply_text("❌ 1-50")
        except:
            await update.message.reply_text("❌ Число")
        del context.user_data['waiting_loss_limit']

# ========== ЗАПУСК ==========
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.job_queue.run_repeating(periodic_check, interval=CHECK_INTERVAL, first=10)
    print("🤖 БОТ ЗАПУЩЕН")
    app.run_polling()

if __name__ == "__main__":
    main()
