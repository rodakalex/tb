# trading_analysis/backtest.py
import ast
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

verbose = False

from trading_analysis.risk import calculate_position_size

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


def _process_take_profit(index, price_high, price_low, position_type, entry_price, position, balance, tp_pct, fee):
    """Возвращает сделку по TP, если достигнута целевая цена"""
    if position_type == "long":
        tp_price = entry_price * (1 + tp_pct)
        if price_high >= tp_price:
            pnl = (tp_price - entry_price) * position - (position * tp_price * fee)
            return ("TP CLOSE LONG", index, tp_price, pnl, {}), balance + pnl, None, 0

    elif position_type == "short":
        tp_price = entry_price * (1 - tp_pct)
        if price_low <= tp_price:
            pnl = (entry_price - tp_price) * position - (position * tp_price * fee)
            return ("TP CLOSE SHORT", index, tp_price, pnl, {}), balance + pnl, None, 0

    return None, balance, position_type, position



def _process_stop_loss(index, price_open, price_high, price_low,
                       position_type, entry_price, position, balance, sl_pct, fee=0.00055):
    """Проверяет срабатывание стоп-лосса, без индикаторов"""
    if position_type == "long":
        sl_price = entry_price * (1 - sl_pct)
        if price_low <= sl_price:
            exit_price = price_open if price_open < sl_price else sl_price
            pnl = (exit_price - entry_price) * position
            pnl -= exit_price * position * fee
            return ("SL CLOSE LONG", index, exit_price, pnl, {}), balance + pnl, None, 0

    elif position_type == "short":
        sl_price = entry_price * (1 + sl_pct)
        if price_high >= sl_price:
            exit_price = price_open if price_open > sl_price else sl_price
            pnl = (entry_price - exit_price) * position
            pnl -= exit_price * position * fee
            return ("SL CLOSE SHORT", index, exit_price, pnl, {}), balance + pnl, None, 0

    return None, balance, position_type, position

def _open_position(row, balance, leverage, position_size, fee):
    entry_price = row["close"]
    position_value = position_size * entry_price
    actual_fee = position_value * fee
    balance -= actual_fee

    if row.get("long_entry"):
        return ("OPEN LONG", row.name, entry_price, 0, {}), balance, "long", position_size
    elif row.get("short_entry"):
        return ("OPEN SHORT", row.name, entry_price, 0, {}), balance, "short", position_size

    return None, balance, None, 0


def _close_position_by_signal(row, position_type, entry_price, position, balance, fee=0.00055):
    exit_price = row["close"]

    if position_type == "long" and int(row.get("long_exit", 0)) == 1:
        pnl = (exit_price - entry_price) * position
        commission = position * exit_price * fee
        pnl -= commission
        return ("CLOSE LONG", row.name, exit_price, pnl, {}), balance + pnl, None, 0

    elif position_type == "short" and row.get("short_exit", 0) == 1:
        pnl = (entry_price - exit_price) * position
        commission = position * exit_price * fee
        pnl -= commission
        return ("CLOSE SHORT", row.name, exit_price, pnl, {}), balance + pnl, None, 0

    return None, balance, position_type, position

def max_streak(trades, kind="win"):
    streak = 0
    max_streak = 0
    for t in trades:
        if len(t) != 5:
            continue
        pnl = t[3]
        if (kind == "win" and pnl > 0) or (kind == "loss" and pnl <= 0):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

def analyze_backtest(trades, final_balance, initial_balance):
    # Отбираем только сделки на закрытие с корректной структурой и PnL
    if verbose:
        print(trades)
    
    closed_trades = [
        t for t in trades
        if isinstance(t, tuple)
        and len(t) >= 4
        and isinstance(t[3], (int, float))
        and t[0] in (
            'CLOSE LONG', 'CLOSE SHORT',
            'TP CLOSE LONG', 'TP CLOSE SHORT',
            'SL CLOSE LONG', 'SL CLOSE SHORT',
            'AUTO SELL', 'AUTO COVER' 
        )
    ]

    total_trades = len(closed_trades)

    if total_trades == 0:
        pnl = final_balance - initial_balance
        return {
            "winrate": 0.0,
            "pnl": pnl,
            "avg_trade": 0.0,
            "total_trades": 0,
            "lossrate": 1.0,
            "final_balance": final_balance,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "trades": [],
        }

    wins = [t for t in closed_trades if t[3] > 0]
    losses = [t for t in closed_trades if t[3] < 0]
    total_pnl = sum(t[3] for t in closed_trades)
    avg_trade = total_pnl / total_trades
    winrate = len(wins) / total_trades
    lossrate = len(losses) / total_trades

    return {
        "winrate": winrate,
        "pnl": total_pnl,
        "avg_trade": avg_trade,
        "total_trades": total_trades,
        "lossrate": lossrate,
        "final_balance": final_balance,
        "max_win_streak": max_streak(closed_trades, "win"),
        "max_loss_streak": max_streak(closed_trades, "loss"),
        "trades": trades,
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

def _handle_active_position(state, row, tp_sl):
    if verbose:
        print("⚙️ Обработка активной позиции", state)
        print("📈 Цены: high =", row["high"], "low =", row["low"])
    
    price_high = row["high"]
    price_low = row["low"]
    index = row.name

    checkers = [
        lambda: _process_take_profit(
            index, price_high, price_low,
            state['position_type'], state['entry_price'],
            state['position'], state['balance'],
            tp_sl['tp'], state['fee']
        ),
        lambda: _process_stop_loss(
            index,
            row["open"], row["high"], row["low"],
            state['position_type'], state['entry_price'],
            state['position'], state['balance'], tp_sl['sl'], state['fee']
        ),
    ]

    for checker in checkers:
        trade, new_balance, new_type, new_pos = checker()
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
    sl_pct = state.get('sl_pct', 0.02)
    fee = state['fee']
    position_size = calculate_position_size(balance, risk_pct, leverage, sl_pct, price)

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
        if verbose:
            print("🚪 Форсируем закрытие в конце:", state)
        final_price = df["close"].iloc[-1]

        if state['position_type'] == 'long':
            pnl = (final_price - state['entry_price']) * state['position']
            action = "AUTO SELL"
        else:
            pnl = (state['entry_price'] - final_price) * state['position']
            action = "AUTO COVER"

        commission = final_price * state['position'] * state['fee']
        pnl -= commission

        state['balance'] += pnl

        # Без индикаторов — просто пустой словарь
        state['trades'].append((action, df.index[-1], final_price, pnl, {}))

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
                 tp_pct=0.025, sl_pct=0.0175,
                 report=True, open_position=None, finalize=True, risk_pct=0.05, fee = 0.00055):
    state = {
        'balance': initial_balance,
        'position': 0,
        'entry_price': 0,
        'position_type': None,
        'trades': [],
        'leverage': leverage,
        'fee': fee,
        'risk_pct': risk_pct,
        'sl_pct': sl_pct
    }
    if open_position:
        state.update({
            'position': open_position.get("position", 0),
            'entry_price': open_position.get("entry_price", 0),
            'position_type': open_position.get("position_type", None),
            'trades': open_position.get("trades", [])  # важно!
        })

    state["risk_pct"] = risk_pct
    state["sl_pct"] = sl_pct
    if verbose:
        print("🔁 Восстановление позиции:", state)

    for index, row in df.iterrows():
        tp_sl = {'tp': tp_pct, 'sl': sl_pct}

        if state['position_type']:
            if _handle_active_position(state, row, tp_sl):
                continue

            _maybe_force_close(state, row)
        else:
            _try_open_position(state, row)

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
   