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

# 10 самых популярных монет
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "LINK/USDT", "LTC/USDT"
]

TIMEFRAME = "1h"
MIN_CONFIDENCE = 0.6
CHECK_INTERVAL = 60

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

# ========== БАЛАНС ==========
user_balances = {}

def get_balance(user_id):
    return user_balances.get(str(user_id), 10000)

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
            'enabled': False,
            'max_position_percent': 5,
            'max_trades_per_day': 10,
            'max_daily_loss': 10,
            'signals': {}
        }
    return autotrade_settings[uid]

# ========== РЕКОМЕНДАЦИИ ==========
def get_budget_recommendations(balance_usdt):
    if balance_usdt < 100:
        return {'position': 15, 'stop': 7, 'take': 12, 'risk': 'Высокий'}
    elif balance_usdt < 500:
        return {'position': 10, 'stop': 6, 'take': 12, 'risk': 'Средний'}
    elif balance_usdt < 2000:
        return {'position': 5, 'stop': 5, 'take': 10, 'risk': 'Низкий'}
    else:
        return {'position': 3, 'stop': 4, 'take': 8, 'risk': 'Минимальный'}

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
            features.append([rsi, sma7/sma25 - 1, price_change, volatility])
        return features
    
    def prepare_data(self, symbol, limit=500):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            closes = [c[4] for c in ohlcv]
            features = self.get_features(closes)
            if len(features) < 100:
                return None, None
            X = []
            y = []
            for i in range(len(features) - 6):
                X.append(features[i])
                y.append(1 if closes[i+56] > closes[i+50] else 0)
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
            return None, 0.5, 50
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            closes = [c[4] for c in ohlcv]
            features = self.get_features(closes)
            if len(features) < 1:
                return None, 0.5, 50
            latest = features[-1]
            Xs = self.scalers[symbol].transform([latest])
            proba = self.models[symbol].predict_proba(Xs)[0]
            pred = self.models[symbol].predict(Xs)[0]
            rsi = self.calculate_rsi(closes)
            return pred, max(proba), rsi
        except Exception as e:
            logger.error(f"Ошибка {symbol}: {e}")
            return None, 0.5, 50
    
    def predict_all(self):
        """Предсказать для всех монет и вернуть отсортированные по силе сигнала"""
        results = []
        for sym in SYMBOLS:
            pred, conf, rsi = self.predict(sym)
            if pred is not None:
                results.append({
                    'symbol': sym,
                    'signal': pred,
                    'confidence': conf,
                    'rsi': rsi,
                    'strength': conf if pred == 1 else -conf
                })
            await asyncio.sleep(0.5)  # Пауза между запросами
        # Сортировка по силе сигнала (от лучшего к худшему)
        results.sort(key=lambda x: x['strength'], reverse=True)
        return results
    
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
    auto_status = "✅" if auto['enabled'] else "❌"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"💰 {balance:.0f} USDT", callback_data="balance")],
        [InlineKeyboardButton("📊 ТОП сигналов", callback_data="top_signals")],
        [InlineKeyboardButton("🧠 Обучить все", callback_data="train_all")],
        [InlineKeyboardButton(f"🤖 Авто: {auto_status}", callback_data="auto_menu")],
        [InlineKeyboardButton("📈 BTC", callback_data="signal_BTC/USDT"), InlineKeyboardButton("📈 ETH", callback_data="signal_ETH/USDT")],
        [InlineKeyboardButton("📈 SOL", callback_data="signal_SOL/USDT"), InlineKeyboardButton("📈 BNB", callback_data="signal_BNB/USDT")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"), InlineKeyboardButton("💡 Советы", callback_data="tips")],
        [InlineKeyboardButton("➕ +1000", callback_data="deposit"), InlineKeyboardButton("➖ -1000", callback_data="withdraw")],
    ])

def get_auto_keyboard(user_id):
    at = get_autotrade(user_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{'✅' if at['enabled'] else '❌'} Вкл/Выкл", callback_data="auto_toggle")],
        [InlineKeyboardButton(f"💰 Макс время: {at['max_position_percent']}%", callback_data="auto_percent")],
        [InlineKeyboardButton(f"📊 Лимит сделок: {at['max_trades_per_day']}", callback_data="auto_trades")],
        [InlineKeyboardButton("◀️ Назад", callback_data="back")],
    ])

# ========== ФУНКЦИИ ==========
async def top_signals_message(context, user_id):
    """Формирует сообщение с топ-сигналами"""
    if not model.is_trained:
        await context.bot.send_message(chat_id=user_id, text="❌ Сначала обучите модели! Нажмите «Обучить все»")
        return
    
    await context.bot.send_message(chat_id=user_id, text="🔄 Анализирую 10 монет... ⏳")
    results = await model.predict_all()
    
    buy_signals = [r for r in results if r['signal'] == 1 and r['confidence'] > MIN_CONFIDENCE]
    sell_signals = [r for r in results if r['signal'] == 0 and r['confidence'] > MIN_CONFIDENCE]
    wait_signals = [r for r in results if r['confidence'] <= MIN_CONFIDENCE]
    
    text = "🔥 *ТОП СИГНАЛОВ*\n\n"
    
    if buy_signals:
        text += "*🟢 К ПОКУПКЕ:*\n"
        for r in buy_signals[:5]:
            text += f"• {r['symbol']} — ув.{r['confidence']:.0%} | RSI {r['rsi']:.0f}\n"
    else:
        text += "*🟢 ПОКУПКА:* нет сигналов\n"
    
    if sell_signals:
        text += f"\n*🔴 К ПРОДАЖЕ:*\n"
        for r in sell_signals[:5]:
            text += f"• {r['symbol']} — ув.{r['confidence']:.0%} | RSI {r['rsi']:.0f}\n"
    else:
        text += f"\n*🔴 ПРОДАЖА:* нет сигналов\n"
    
    rec = get_budget_recommendations(get_balance(user_id))
    text += f"\n💡 *Рекомендация:* входите {rec['position']}% депозита, стоп {rec['stop']}%"
    
    await context.bot.send_message(chat_id=user_id, text=text, parse_mode='Markdown')

async def send_signal_to_user(context, user_id, symbol, prediction, confidence, rsi, price):
    balance = get_balance(user_id)
    rec = get_budget_recommendations(balance)
    
    if prediction == 1 and confidence > MIN_CONFIDENCE:
        text = (f"🚨 *{symbol} — ПОКУПКА!*\n💰 ${price:,.0f}\n🎯 Уверенность: {confidence:.0%}\n"
                f"📈 RSI: {rsi:.0f}\n💡 Входите ${get_balance(user_id) * rec['position'] / 100:.0f} USDT")
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🟢 Купить", callback_data=f"buy_{symbol}_{price}")]])
    elif prediction == 0 and confidence > MIN_CONFIDENCE:
        text = (f"🚨 *{symbol} — ПРОДАЖА!*\n💰 ${price:,.0f}\n🎯 Уверенность: {confidence:.0%}\n📉 RSI: {rsi:.0f}")
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔴 Продать", callback_data=f"sell_{symbol}_{price}")]])
    else:
        text = f"⏸️ *{symbol} — сигнала нет*\n💰 ${price:,.0f}\n📊 RSI: {rsi:.0f}"
        kb = None
    
    await context.bot.send_message(chat_id=user_id, text=text, reply_markup=kb, parse_mode='Markdown')

# ========== ОБРАБОТЧИКИ ==========
async def background_task(app):
    """Фоновый цикл"""
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        if model.is_trained:
            for sym in SYMBOLS:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    pred, conf, rsi = model.predict(sym)
                    for uid in ALLOWED_USERS:
                        await send_signal_to_user(app, uid, sym, pred, conf, rsi, ticker['last'])
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Ошибка {sym}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ALLOWED_USERS:
        await update.message.reply_text("❌ Доступ запрещён")
        return
    if str(uid) not in user_balances:
        user_balances[str(uid)] = 10000
    await update.message.reply_text(
        "🤖 *CRYPTO TRADING BOT*\n\n"
        f"📊 Монет: {len(SYMBOLS)}\n"
        f"💰 Баланс: {get_balance(uid):.0f} USDT\n"
        "✅ Автосигналы раз в минуту\n"
        "👇 Выберите действие:",
        reply_markup=get_main_keyboard(uid),
        parse_mode='Markdown'
    )

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
        rec = get_budget_recommendations(get_balance(uid))
        amount = get_balance(uid) * rec['position'] / 100
        if subtract_balance(uid, amount):
            await query.edit_message_text(f"✅ КУПЛЕНО {symbol}\n💰 -{amount:.0f} USDT\n💡 Новый баланс: {get_balance(uid):.0f} USDT")
        else:
            await query.edit_message_text("❌ Недостаточно средств!")
        await start(update, context)
    
    elif data.startswith("sell_"):
        parts = data.split("_")
        symbol = parts[1] + "/" + parts[2]
        await query.edit_message_text(f"✅ ПРОДАНО {symbol}\n💰 Баланс: {get_balance(uid):.0f} USDT")
        await start(update, context)
    
    elif data.startswith("signal_"):
        sym = data.split("_")[1].replace("-", "/")
        if not model.is_trained:
            await query.edit_message_text("❌ Сначала обучите модель!")
            await start(update, context)
            return
        ticker = exchange.fetch_ticker(sym)
        pred, conf, rsi = model.predict(sym)
        await send_signal_to_user(context, uid, sym, pred, conf, rsi, ticker['last'])
        await start(update, context)
    
    elif data == "top_signals":
        await top_signals_message(context, uid)
        await start(update, context)
    
    elif data == "train_all":
        await query.edit_message_text("🧠 Обучение 10 моделей... 3-4 минуты ⏳")
        res = model.train()
        text = "✅ *ОБУЧЕНО!*\n\n"
        for sym, r in res.items():
            if "error" in r:
                text += f"❌ {sym}: {r['error']}\n"
            else:
                text += f"✅ {sym}: {r['accuracy']:.0%} ({r['samples']})\n"
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "auto_menu":
        await query.edit_message_text("⚙️ *АВТОТОРГОВЛЯ*", reply_markup=get_auto_keyboard(uid), parse_mode='Markdown')
    
    elif data == "auto_toggle":
        at = get_autotrade(uid)
        at['enabled'] = not at['enabled']
        await query.edit_message_text(f"✅ Автоторговля {'включена' if at['enabled'] else 'выключена'}")
        await button_handler(update, context)
    
    elif data == "auto_percent":
        at = get_autotrade(uid)
        at['max_position_percent'] = 5 if at['max_position_percent'] >= 10 else at['max_position_percent'] + 1
        await query.edit_message_text(f"✅ Макс. позиция: {at['max_position_percent']}%")
        await button_handler(update, context)
    
    elif data == "auto_trades":
        await query.edit_message_text("💰 Введите лимит сделок в день (1-30):")
        context.user_data['waiting_trades'] = uid
    
    elif data == "deposit":
        add_balance(uid, 1000)
        await query.edit_message_text(f"✅ Пополнено 1000 USDT\n💰 Новый баланс: {get_balance(uid):.0f} USDT")
        await start(update, context)
    
    elif data == "withdraw":
        if subtract_balance(uid, 1000):
            await query.edit_message_text(f"✅ Выведено 1000 USDT\n💰 Новый баланс: {get_balance(uid):.0f} USDT")
        else:
            await query.edit_message_text("❌ Недостаточно средств!")
        await start(update, context)
    
    elif data == "balance":
        await query.edit_message_text(f"💰 *БАЛАНС*\n💵 {get_balance(uid):.0f} USDT", parse_mode='Markdown')
        await start(update, context)
    
    elif data == "stats":
        trained = sum(1 for s in SYMBOLS if s in model.models)
        at = get_autotrade(uid)
        text = (f"📊 *СТАТИСТИКА*\n\n"
                f"🎯 Модели: {trained}/{len(SYMBOLS)}\n"
                f"💰 Баланс: {get_balance(uid):.0f} USDT\n"
                f"🤖 Автоторговля: {'✅' if at['enabled'] else '❌'}\n"
                f"📊 Лимит сделок: {at['max_trades_per_day']}/день")
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "tips":
        tips = (
            "📖 *СОВЕТЫ ПО ТРЕЙДИНГУ*\n\n"
            "1️⃣ *Риск-менеджмент*\n"
            "• Входите 3-15% от депозита\n"
            "• Всегда ставьте стоп-лосс\n\n"
            "2️⃣ *Сигналы*\n"
            "• 🟢 ПОКУПАТЬ → входите\n"
            "• 🔴 ПРОДАВАТЬ → фиксируйте\n"
            "• ⏸️ ЖДАТЬ → не торгуйте\n\n"
            "3️⃣ *Рекомендации*\n"
            "• Обучайте модели 1-2 раза в день\n"
            "• Не торгуйте на эмоциях\n"
            "⚠️ Криптовалюты — высокий риск!"
        )
        await query.edit_message_text(tips, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "back":
        await start(update, context)

async def handle_message(update, context):
    uid = update.effective_user.id
    if context.user_data.get('waiting_trades') == uid:
        try:
            val = int(update.message.text)
            if 1 <= val <= 30:
                get_autotrade(uid)['max_trades_per_day'] = val
                await update.message.reply_text(f"✅ Лимит сделок: {val} в день")
            else:
                await update.message.reply_text("❌ Введите число от 1 до 30")
        except:
            await update.message.reply_text("❌ Введите число")
        del context.user_data['waiting_trades']

# ========== ЗАПУСК ==========
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(background_task(app))
    
    print("="*50)
    print("🤖 БОТ ЗАПУЩЕН (Render)")
    print(f"📊 Монет: {len(SYMBOLS)}")
    print(f"💰 Стартовый баланс: 10000 USDT")
    print("⏱️ Автосигналы каждую минуту")
    print("="*50)
    app.run_polling()

if __name__ == "__main__":
    main()
