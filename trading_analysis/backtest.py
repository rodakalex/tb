# trading_analysis/backtest.py
import ast
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

def plot_trades(df, trades, symbol="SYMBOL"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Candles"
    ))

    for trade in trades:
        if len(trade) == 5:
            trade_type, ts, price, pnl, _ = trade
        elif len(trade) == 3:
            trade_type, ts, price = trade

        if "OPEN" in trade_type:
            symbol_shape = "triangle-down" if "SHORT" in trade_type else "triangle-up"
            color = "orange"
        elif "TP" in trade_type:
            symbol_shape = "star"
            color = "lime"
        elif "SL" in trade_type:
            symbol_shape = "x"
            color = "red"
        elif "CLOSE" in trade_type:
            symbol_shape = "circle"
            color = "white"
        else:
            symbol_shape = "diamond"
            color = "blue"

        fig.add_trace(go.Scatter(
            x=[ts],
            y=[price],
            mode="markers",
            marker=dict(symbol=symbol_shape, size=12, color=color),
            name=trade_type
        ))

    fig.update_layout(
        title=f"Сделки по {symbol}",
        xaxis_title="Время",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700
    )

    fig.write_html(f"charts/trades_{symbol}.html")
    print(f"График сохранён: charts/trades_{symbol}.html")


def _process_take_profit(row, position_type, entry_price, position, balance, tp_pct=0.05, fee=0.00055):
    """Обрабатывает срабатывание тейк-профита с учётом комиссии"""
    indicators = {
        "EMA_9": row.get("EMA_9"),
        "EMA_21": row.get("EMA_21"),
        "EMA_200": row.get("EMA_200"),
        "RSI": row.get("RSI"),
        "MACD": row.get("MACD"),
        "Signal": row.get("Signal"),
        "Volume": row.get("volume"),
        "Volume_SMA_20": row.get("Volume_SMA_20"),
        "ATR": row.get("atr"),
    }

    if position_type == "long":
        tp_price = entry_price * (1 + tp_pct)
        if row["high"] >= tp_price:
            exit_price = tp_price
            fee = position * exit_price * fee
            pnl = (exit_price - entry_price) * position - fee
            return ("TP CLOSE LONG", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    elif position_type == "short":
        tp_price = entry_price * (1 - tp_pct)
        if row["low"] <= tp_price:
            exit_price = tp_price
            fee = position * exit_price * fee
            pnl = (entry_price - exit_price) * position - fee
            return ("TP CLOSE SHORT", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    return None, balance, position_type, position

def _process_stop_loss(row, position_type, entry_price, position, balance, sl_pct=0.02, fee=0.00055):
    """Обрабатывает срабатывание стоп-лосса с учётом комиссии"""
    indicators = {
        "EMA_9": row.get("EMA_9"),
        "EMA_21": row.get("EMA_21"),
        "EMA_200": row.get("EMA_200"),
        "RSI": row.get("RSI"),
        "MACD": row.get("MACD"),
        "Signal": row.get("Signal"),
        "Volume": row.get("volume"),
        "Volume_SMA_20": row.get("Volume_SMA_20"),
        "ATR": row.get("atr"),
    }

    sl_pct = row['ATR'] * 1.2 / entry_price

    if position_type == "long":
        sl_price = entry_price * (1 - sl_pct)
        if row["low"] <= sl_price:
            exit_price = sl_price if row["open"] > sl_price else row["low"]
            pnl = (exit_price - entry_price) * position
            commission = exit_price * position * fee
            pnl -= commission
            return ("SL CLOSE LONG", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    elif position_type == "short":
        sl_price = entry_price * (1 + sl_pct)
        if row["high"] >= sl_price:
            exit_price = sl_price if row["open"] < sl_price else row["high"]
            pnl = (entry_price - exit_price) * position
            commission = exit_price * position * fee
            pnl -= commission
            return ("SL CLOSE SHORT", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    return None, balance, position_type, position

def _open_position(row, balance, leverage, position_size, fee):
    entry_price = row["close"]
    position_value = position_size * entry_price
    actual_fee = position_value * fee
    balance -= actual_fee

    indicators = {
        "EMA_9": row.get("EMA_9"),
        "EMA_21": row.get("EMA_21"),
        "EMA_200": row.get("EMA_200"),
        "RSI": row.get("RSI"),
        "MACD": row.get("MACD"),
        "Signal": row.get("Signal"),
        "Volume": row.get("volume"),
        "Volume_SMA_20": row.get("Volume_SMA_20"),
        "ATR": row.get("atr"),
    }

    if row.get("long_entry"):
        return ("OPEN LONG", row.name, entry_price, 0, indicators), balance, "long", position_size
    elif row.get("short_entry"):
        return ("OPEN SHORT", row.name, entry_price, 0, indicators), balance, "short", position_size

    return None, balance, None, 0

def _close_position_by_signal(row, position_type, entry_price, position, balance, fee=0.00055):
    """Закрывает позицию по сигналу выхода (с учётом комиссии)"""
    indicators = {
        "EMA_9": row.get("EMA_9"),
        "EMA_21": row.get("EMA_21"),
        "EMA_200": row.get("EMA_200"),
        "RSI": row.get("RSI"),
        "MACD": row.get("MACD"),
        "Signal": row.get("Signal"),
        "Volume": row.get("volume"),
        "Volume_SMA_20": row.get("Volume_SMA_20"),
        "ATR": row.get("atr"),
    }

    exit_price = row["close"]

    if position_type == "long" and row.get("long_exit"):
        pnl = (exit_price - entry_price) * position
        commission = position * exit_price * fee
        pnl -= commission
        return ("CLOSE LONG", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    elif position_type == "short" and row.get("short_exit"):
        pnl = (entry_price - exit_price) * position
        commission = position * exit_price * fee
        pnl -= commission
        return ("CLOSE SHORT", row.name, exit_price, pnl, indicators), balance + pnl, None, 0

    return None, balance, position_type, position

def analyze_backtest(trades, final_balance, initial_balance):
    closed_trades = [
        t for t in trades 
        if isinstance(t, tuple)
        and len(t) == 5
        and isinstance(t[3], (int, float))
        and abs(t[3]) > 1e-8  # исключает "OPEN" с PnL=0.0
    ]
    total_trades = len(closed_trades)

    if total_trades == 0:
        return {
            "winrate": 0.0,
            "pnl": final_balance - initial_balance,
            "avg_trade": 0.0,
            "total_trades": 0,
            "loss": 9999,
            "final_balance": final_balance
        }

    wins = [t for t in closed_trades if t[3] > 0]
    total_pnl = sum(t[3] for t in closed_trades)
    avg_trade = total_pnl / total_trades
    winrate = len(wins) / total_trades

    return {
        "winrate": winrate,
        "pnl": total_pnl,
        "avg_trade": avg_trade,
        "total_trades": total_trades,
        "loss": -winrate,
        "final_balance": final_balance
    }


def _print_backtest_results(trades, final_balance, symbol, result):

    print(f"\n>>> Результат бэктеста по {symbol} <<<")
    print(f"Финальный баланс: ${final_balance:.2f}")
    print(f"Сделок всего: {result['total_trades']}")
    print(f"Winrate: {result['winrate'] * 100:.2f}%")
    print(f"Средняя сделка: ${result['avg_trade']:.2f}")
    print(f"Общий PnL: ${result['pnl']:.2f}")
    
    if result["total_trades"] > 0:
        losses = [t for t in trades if len(t) == 5 and t[3] <= 0]
        avg_loss = sum(t[3] for t in losses) / len(losses) if losses else 0
        max_loss = min(t[3] for t in losses) if losses else 0
        print(f"Средний убыток: ${avg_loss:.2f}")
        print(f"Максимальный убыток: ${max_loss:.2f}")
    else:
        print("Сделок для анализа нет.")


def _prepare_tp_sl(row, entry_price, fallback_tp, fallback_sl, dynamic):
    if dynamic and row["ATR"] > 0 and entry_price > 0:
        return {
            'tp': row["ATR"] * 2 / entry_price,
            'sl': row["ATR"] * 1.2 / entry_price
        }
    return {'tp': fallback_tp, 'sl': fallback_sl}

def _handle_active_position(state, row, tp_sl):
    for func in [_process_take_profit, _process_stop_loss]:
        if func == _process_take_profit:
            trade, new_balance, new_type, new_pos = func(
                row, state['position_type'], state['entry_price'],
                state['position'], state['balance'], 
                tp_pct=tp_sl['tp'], fee=state['fee']
            )
        else:
            trade, new_balance, new_type, new_pos = func(
                row, state['position_type'], state['entry_price'],
                state['position'], state['balance'], 
                sl_pct=tp_sl['sl'], fee=state['fee']
            )

        if trade:
            state.update({
                'balance': new_balance,
                'position_type': new_type,
                'position': new_pos,
                'entry_price': 0
            })
            state['trades'].append(trade)
            return True
    return False


def _try_open_position(state, row):
    price = row['close']
    balance = state['balance']
    risk_pct = state.get('risk_pct', 0.01)
    leverage = state['leverage']
    sl_pct = state.get('sl_pct', 0.02)  # минимальная защита
    fee = state['fee']

    # 💡 Расчёт позиции через риск
    risk_amount = balance * risk_pct
    position_value = (risk_amount * leverage) / sl_pct
    position_size = position_value / price

    # Вызов с готовым объёмом позиции
    trade, new_balance, pos_type, pos = _open_position(
        row,
        balance,
        leverage,
        position_size,
        fee
    )
    if trade:
        state.update({
            'balance': new_balance,
            'position_type': pos_type,
            'position': pos,
            'entry_price': price
        })
        state['trades'].append(trade)
        return True
    return False


def _maybe_force_close(state, row):
    trade, new_balance, pos_type, pos = _close_position_by_signal(
        row, state['position_type'], state['entry_price'], state['position'], state['balance'], state['fee']
    )
    if trade:
        state.update({
            'balance': new_balance,
            'position_type': pos_type,
            'position': pos
        })
        state['trades'].append(trade)

def _finalize_position(state, df):
    if state['position_type'] and state['position'] > 0:
        final_price = df["close"].iloc[-1]
        pnl = (final_price - state['entry_price']) * state['position'] \
              if state['position_type'] == 'long' \
              else (state['entry_price'] - final_price) * state['position']
        
        commission = final_price * state['position'] * state['fee']
        pnl -= commission
        
        state['balance'] += pnl
        action = "AUTO SELL" if state['position_type'] == 'long' else "AUTO COVER"

        indicators = {
            "EMA_9": df["EMA_9"].iloc[-1],
            "EMA_21": df["EMA_21"].iloc[-1],
            "EMA_200": df["EMA_200"].iloc[-1],
            "RSI": df["RSI"].iloc[-1],
            "MACD": df["MACD"].iloc[-1],
            "Signal": df["Signal"].iloc[-1],
            "Volume": df["volume"].iloc[-1],
            "Volume_SMA_20": df["Volume_SMA_20"].iloc[-1],
            "ATR": df["ATR"].iloc[-1],
        }

        state['trades'].append((action, df.index[-1], final_price, pnl, indicators))

        state.update({
            'position_type': None,
            'position': 0,
            'entry_price': 0
        })

def _save_to_csv(trades, symbol: str, output_dir: str = "csv"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for trade in trades:
        if len(trade) == 5:
            ttype, ts, price, pnl, indicators = trade
        else:
            ttype, ts, price = trade
            pnl, indicators = None, {}

        if isinstance(indicators, str):
            try:
                indicators = ast.literal_eval(indicators)
            except:
                indicators = {}

        if isinstance(indicators, dict):
            row = {
                "type": ttype,
                "timestamp": ts,
                "price": price,
                "pnl": pnl,
                **{k: indicators.get(k, None) for k in [
                    "EMA_9", "EMA_21", "EMA_200", "RSI", "MACD", "Signal",
                    "Volume", "Volume_SMA_20", "ATR"]}
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/trades_{symbol}.csv", index=False)

def _save_and_report(df, state, symbol, initial_balance, result):
    _print_backtest_results(state["trades"], state["balance"], symbol, result)
    plot_trades(df, state["trades"], symbol)
    _save_to_csv(state["trades"], symbol)

def run_backtest(df, symbol=None, leverage=1.0, initial_balance=1000.0,
                 tp_pct=0.025, sl_pct=0.0175, use_dynamic_tp_sl=True,
                 report=True, initial_state=None, finalize=True, risk_pct=0.05):
    fee = 0.00055
    if initial_state:
        state = initial_state.copy()
    else:
        state = {
            'balance': initial_balance,
            'position': 0,
            'entry_price': 0,
            'position_type': None,
            'trades': [],
            'leverage': leverage,
            'fee': fee,
        }
    state["risk_pct"] = risk_pct
    state["sl_pct"] = sl_pct

    for index, row in df.iterrows():
        tp_sl = _prepare_tp_sl(row, state['entry_price'], tp_pct, sl_pct, use_dynamic_tp_sl)

        if state['position_type']:
            if _handle_active_position(state, row, tp_sl):
                continue

        if state['position_type'] is None:
            if _try_open_position(state, row):
                continue

        if state['position_type']:
            _maybe_force_close(state, row)

    if finalize:
        _finalize_position(state, df)
    
    
    result = analyze_backtest(state["trades"], state["balance"], initial_balance)
    if report:
        _save_and_report(df, state, symbol, initial_balance, result)

    return result, state


def plot_equity_curve(trades, initial_balance):
    balance = initial_balance
    equity = []
    timestamps = []

    for trade in trades:
        if len(trade) == 4:
            _, ts, _, pnl = trade
            balance += pnl
            equity.append(balance)
            timestamps.append(ts)

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, equity, label="Equity Curve", color="green")
    plt.title("Balance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    
    equity_df = pd.DataFrame({
        "timestamp": timestamps,
        "balance": equity
    })
    equity_df.to_csv("equity_log.csv", index=False)
    