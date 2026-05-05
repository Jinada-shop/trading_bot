import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

# ========== БАЛАНС ПОЛЬЗОВАТЕЛЯ ==========
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

# ========== АВТОТОРГОВЛЯ ==========
def get_default_autotrade():
    return {
        'enabled': False,
        'mode': 'signal_only',
        'max_trades_per_day': 5,
        'max_daily_loss': 10,
        'cooldown_minutes': 30,
        'position': None,
        'entry_price': 0,
        'trades_today': 0,
        'daily_loss': 0,
        'last_trade_time': None
    }

autotrade_settings = {}

def get_autotrade(user_id):
    uid = str(user_id)
    if uid not in autotrade_settings:
        autotrade_settings[uid] = get_default_autotrade()
    return autotrade_settings[uid]

# ========== РЕКОМЕНДАЦИИ ПО БЮДЖЕТУ ==========
def get_budget_recommendations(balance_usdt):
    if balance_usdt < 100:
        return {
            'level': '🟢 Начинающий',
            'advice': 'Торгуйте аккуратно',
            'position_percent': 15,
            'max_position_usdt': balance_usdt * 0.15,
            'stop_loss_percent': 7,
            'take_profit_percent': 12,
            'recommended_symbols': ['BTC/USDT'],
            'warning': '⚠️ Не рискуйте больше чем готовы потерять!'
        }
    elif balance_usdt < 500:
        return {
            'level': '🟡 Средний',
            'advice': 'Можно торговать активнее',
            'position_percent': 8,
            'max_position_usdt': balance_usdt * 0.08,
            'stop_loss_percent': 6,
            'take_profit_percent': 12,
            'recommended_symbols': ['BTC/USDT', 'ETH/USDT'],
            'warning': '✅ Не забывайте про стоп-лоссы'
        }
    elif balance_usdt < 2000:
        return {
            'level': '🟠 Продвинутый',
            'advice': 'Диверсифицируйте портфель',
            'position_percent': 5,
            'max_position_usdt': balance_usdt * 0.05,
            'stop_loss_percent': 5,
            'take_profit_percent': 10,
            'recommended_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'warning': '📊 Используйте автоторговлю с осторожностью'
        }
    else:
        return {
            'level': '🔴 Профессиональный',
            'advice': 'Можете использовать сложные стратегии',
            'position_percent': 3,
            'max_position_usdt': balance_usdt * 0.03,
            'stop_loss_percent': 4,
            'take_profit_percent': 8,
            'recommended_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'],
            'warning': '🎯 Используйте трейлинг стоп'
        }

def get_signal_strength(prediction, confidence):
    if prediction is None:
        return "Недостаточно данных"
    if prediction == 1 and confidence > 0.7:
        return "🟢 СИЛЬНЫЙ СИГНАЛ"
    elif prediction == 1 and confidence > 0.6:
        return "🟡 СРЕДНИЙ СИГНАЛ"
    elif prediction == 0 and confidence > 0.6:
        return "🔴 СИГНАЛ НА ЗАКРЫТИЕ"
    else:
        return "⚪ НЕТ СИГНАЛА"

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
            levels = {
                'stop_loss': closes[-1] * 0.95,
                'take_profit': closes[-1] * 1.10,
                'support': np.min(closes[-20:]),
                'resistance': np.max(closes[-20:])
            }
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
        [InlineKeyboardButton("📊 Рекомендации по бюджету", callback_data="budget")],
        [InlineKeyboardButton("📈 Сигнал (BTC)", callback_data="signal_BTC/USDT"),
         InlineKeyboardButton("📈 Сигнал (ETH)", callback_data="signal_ETH/USDT")],
        [InlineKeyboardButton("📈 Сигнал (SOL)", callback_data="signal_SOL/USDT"),
         InlineKeyboardButton("📈 Сигнал (BNB)", callback_data="signal_BNB/USDT")],
        [InlineKeyboardButton("🧠 Обучить модель", callback_data="train")],
        [InlineKeyboardButton(f"🤖 Автоторговля: {auto_status}", callback_data="autotrade")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"),
         InlineKeyboardButton("💬 Советы", callback_data="tips")],
    ])

# ========== ОТПРАВКА СИГНАЛОВ ==========
async def send_signal_to_user(context, user_id, symbol, prediction, confidence, info, price):
    balance = get_balance(user_id)
    budget_rec = get_budget_recommendations(balance)
    strength = get_signal_strength(prediction, confidence)
    
    if prediction == 1 and confidence > MIN_CONFIDENCE:
        text = (
            f"🚨 *{symbol} — СИГНАЛ НА ПОКУПКУ!*\n\n"
            f"🟢 *КУПИТЬ*\n"
            f"💰 Цена: ${price:,.0f}\n"
            f"🎯 Уверенность: {confidence:.1%}\n"
            f"📊 RSI: {info.get('rsi', 50):.1f}\n"
            f"⚠️ Риск: {info.get('risk', 'Н/Д')}\n\n"
            f"*Уровни:*\n"
            f"🛑 Стоп-лосс: ${info.get('levels', {}).get('stop_loss', 0):,.0f} (-{budget_rec['stop_loss_percent']}%)\n"
            f"🎯 Тейк-профит: ${info.get('levels', {}).get('take_profit', 0):,.0f} (+{budget_rec['take_profit_percent']}%)\n\n"
            f"*💰 По вашему бюджету (${balance:.2f}):*\n"
            f"• Рекомендованная позиция: **${budget_rec['max_position_usdt']:.2f}**\n"
            f"• {strength}\n\n"
            f"💡 {budget_rec['advice']}"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🟢 Купить", callback_data=f"buy_{symbol}_{price}")]])
    elif prediction == 0 and confidence > MIN_CONFIDENCE:
        text = (
            f"🚨 *{symbol} — СИГНАЛ НА ПРОДАЖУ!*\n\n"
            f"🔴 *ПРОДАТЬ*\n"
            f"💰 Цена: ${price:,.0f}\n"
            f"🎯 Уверенность: {confidence:.1%}\n"
            f"📊 RSI: {info.get('rsi', 50):.1f}\n\n"
            f"{strength}"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔴 Продать", callback_data=f"sell_{symbol}_{price}")]])
    else:
        text = (
            f"⏸️ *{symbol} — НЕТ СИГНАЛА*\n\n"
            f"💰 Цена: ${price:,.0f}\n"
            f"📊 RSI: {info.get('rsi', 50):.1f}\n\n"
            f"{strength}\n\n"
            f"💡 Продолжайте наблюдать"
        )
        kb = None
    
    await context.bot.send_message(chat_id=user_id, text=text, reply_markup=kb, parse_mode='Markdown')

async def execute_buy(context, user_id, symbol, price):
    balance = get_balance(user_id)
    rec = get_budget_recommendations(balance)
    amount = rec['max_position_usdt']
    
    if subtract_balance(user_id, amount):
        await context.bot.send_message(
            chat_id=user_id,
            text=f"✅ *ПОКУПКА ВЫПОЛНЕНА!*\n\n"
                 f"📊 {symbol}\n"
                 f"💰 Сумма: ${amount:.2f}\n"
                 f"📈 Цена: ${price:,.0f}\n"
                 f"🛑 Стоп-лосс: -{rec['stop_loss_percent']}%\n"
                 f"🎯 Тейк-профит: +{rec['take_profit_percent']}%\n\n"
                 f"💡 Новый баланс: ${get_balance(user_id):.2f}",
            parse_mode='Markdown'
        )
    else:
        await context.bot.send_message(chat_id=user_id, text="❌ Недостаточно средств на балансе!")

async def execute_sell(context, user_id, symbol, price):
    await context.bot.send_message(
        chat_id=user_id,
        text=f"✅ *ПРОДАЖА ВЫПОЛНЕНА!*\n\n"
             f"📊 {symbol}\n"
             f"📈 Цена: ${price:,.0f}\n\n"
             f"💰 Баланс: ${get_balance(user_id):.2f}",
        parse_mode='Markdown'
    )

# ========== ФОНОВАЯ ПРОВЕРКА ==========
async def periodic_check(application):
    while True:
        await asyncio.sleep(60)
        if model.is_trained:
            for sym in SYMBOLS:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    pred, conf, info = model.predict(sym)
                    for uid in ALLOWED_USERS:
                        await send_signal_to_user(application, uid, sym, pred, conf, info, ticker['last'])
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Ошибка {sym}: {e}")

# ========== ОБРАБОТЧИКИ КОМАНД ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ALLOWED_USERS:
        await update.message.reply_text("❌ Доступ запрещён")
        return
    
    if str(uid) not in user_balances:
        user_balances[str(uid)] = 1000
    
    await update.message.reply_text(
        "🤖 *ТОРГОВЫЙ БОТ*\n\n"
        "✅ Автосигналы раз в минуту\n"
        "✅ 4 монеты: BTC, ETH, SOL, BNB\n"
        "✅ Рекомендации по вашему бюджету\n"
        "✅ Стоп-лосс и тейк-профит\n"
        "✅ Автоторговля с защитой\n\n"
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
        await query.edit_message_text("🟢 Исполняю покупку...")
        await execute_buy(context, uid, symbol, price)
        await start(update, context)
    
    elif data.startswith("sell_"):
        parts = data.split("_")
        symbol = parts[1] + "/" + parts[2]
        price = float(parts[3])
        await query.edit_message_text("🔴 Исполняю продажу...")
        await execute_sell(context, uid, symbol, price)
        await start(update, context)
    
    elif data.startswith("signal_"):
        sym = data.split("_")[1].replace("-", "/")
        if not model.is_trained:
            await query.edit_message_text("❌ Сначала обучите модель (кнопка «Обучить»)")
            return
        ticker = exchange.fetch_ticker(sym)
        pred, conf, info = model.predict(sym)
        await send_signal_to_user(context, uid, sym, pred, conf, info, ticker['last'])
        await start(update, context)
    
    elif data == "budget":
        balance = get_balance(uid)
        rec = get_budget_recommendations(balance)
        text = (
            f"📊 *РЕКОМЕНДАЦИИ ПО БЮДЖЕТУ*\n\n"
            f"💰 Ваш баланс: **${balance:.2f}**\n"
            f"📈 Уровень: {rec['level']}\n\n"
            f"*Стратегия:* {rec['advice']}\n\n"
            f"*Размер позиции:* {rec['position_percent']}% = **${rec['max_position_usdt']:.2f}**\n"
            f"*Стоп-лосс:* -{rec['stop_loss_percent']}%\n"
            f"*Тейк-профит:* +{rec['take_profit_percent']}%\n"
            f"*Рекомендуемые монеты:* {', '.join(rec['recommended_symbols'])}\n\n"
            f"{rec['warning']}"
        )
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "train":
        await query.edit_message_text("🧠 Обучение моделей... 2-3 минуты")
        res = model.train()
        text = "✅ *Модели обучены!*\n\n"
        for sym, r in res.items():
            if "error" in r:
                text += f"❌ {sym}: {r['error']}\n"
            else:
                text += f"✅ {sym}: точность {r['accuracy']:.2%}\n"
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "autotrade":
        await query.edit_message_text("⚙️ *АВТОТОРГОВЛЯ*\nНастройки будут добавлены позже", parse_mode='Markdown')
        await start(update, context)
    
    elif data == "balance":
        await query.edit_message_text(f"💰 *БАЛАНС*\n💵 USDT: ${get_balance(uid):.2f}", parse_mode='Markdown')
        await start(update, context)
    
    elif data == "stats":
        at = get_autotrade(uid)
        text = (
            f"📊 *СТАТИСТИКА*\n\n"
            f"🎯 Модель: {'✅ Обучена' if model.is_trained else '❌ Нет'}\n"
            f"💰 Баланс: ${get_balance(uid):.2f}\n"
            f"🤖 Автоторговля: {'✅ Вкл' if at['enabled'] else '❌ Выкл'}\n"
            f"📊 Сделок сегодня: {at['trades_today']}/{at['max_trades_per_day']}\n"
            f"📉 Убыток сегодня: {at['daily_loss']:.1f}%"
        )
        await query.edit_message_text(text, parse_mode='Markdown')
        await start(update, context)
    
    elif data == "tips":
        tips = (
            "📖 *СОВЕТЫ ПО ТОРГОВЛЕ*\n\n"
            "1️⃣ *Управление рисками*\n"
            "• Входите 2-5% от депозита\n"
            "• Всегда ставьте стоп-лосс\n\n"
            "2️⃣ *Сигналы*\n"
            "• 🟢 ПОКУПАТЬ → входите\n"
            "• 🔴 ПРОДАВАТЬ → фиксируйте\n"
            "• ⏸️ ЖДАТЬ → ничего не делайте\n\n"
            "3️⃣ *Ошибки новичков*\n"
            "• Не усредняйте убытки\n"
            "• Не торгуйте на эмоциях\n"
            "• Переобучайте модель раз в сутки\n\n"
            "⚠️ Торговля криптовалютами — высокий риск!"
        )
        await query.edit_message_text(tips, parse_mode='Markdown')
        await start(update, context)

async def handle_message(update, Update, context):
    # Простой обработчик текстовых сообщений
    pass

# ========== FLASK ДЛЯ WEBHOOK ==========
webhook_app = Flask(__name__)
telegram_app = None

@webhook_app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    try:
        update = Update.de_json(request.get_json(force=True), telegram_app.bot)
        asyncio.run_coroutine_threadsafe(telegram_app.process_update(update), loop)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error"}), 500

@webhook_app.route('/')
def health():
    return jsonify({"status": "alive", "message": "Bot is running"}), 200

# ========== ЗАПУСК ==========
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def main():
    global telegram_app
    
    # Создаём приложение Telegram
    telegram_app = Application.builder().token(TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CallbackQueryHandler(button_handler))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем фоновую проверку
    asyncio.run_coroutine_threadsafe(periodic_check(telegram_app), loop)
    
    # Запускаем Flask
    port = int(os.environ.get('PORT', 10000))
    print("="*50)
    print("🤖 ТОРГОВЫЙ БОТ ЗАПУЩЕН НА RENDER")
    print(f"🌐 Webhook URL: https://your-app.onrender.com/{TOKEN}")
    print("✅ Health check доступен по /")
    print("="*50)
    
    webhook_app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
