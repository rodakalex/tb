import logging

logger = logging.getLogger(__name__)

def calculate_position_size(
    balance: float,
    risk_pct: float,
    leverage: float,
    sl_pct: float,
    price: float,
) -> float:
    """
    Расчёт размера позиции по фиксированному риску.

    :param balance: Общий баланс аккаунта.
    :param risk_pct: Доля баланса, которую готов рискнуть (например, 0.01 = 1%).
    :param leverage: Плечо.
    :param sl_pct: Расстояние до стопа в процентах от цены (0.01 = 1%).
    :param price: Текущая цена инструмента.
    :return: Размер позиции в единицах инструмента.
    """
    if not all(isinstance(v, (int, float)) for v in [balance, risk_pct, leverage, sl_pct, price]):
        return 0.0

    if price <= 0 or sl_pct <= 0 or risk_pct <= 0 or leverage <= 0:
        return 0.0

    risk_amount = balance * risk_pct
    position_value = (risk_amount * leverage) / sl_pct
    position_size = position_value / price

    return position_size

def calculate_sl_pct_from_atr(atr, entry_price, multiplier=1.2):
    """
    Расчёт процентного стоп-лосса на основе ATR и цены входа.
    """
    if atr <= 0 or entry_price <= 0:
        return 0.02  # fallback SL
    return atr * multiplier / entry_price


def generate_dynamic_tp_sl(atr, entry_price, tp_mult=2.0, sl_mult=1.2):
    """
    Возвращает tp/sl как проценты от цены на основе ATR.
    """
    if atr <= 0 or entry_price <= 0:
        return {"tp": 0.05, "sl": 0.02}
    return {
        "tp": atr * tp_mult / entry_price,
        "sl": atr * sl_mult / entry_price
    }


def calculate_inverse_balance_risk(current_balance, initial_balance, base=0.05, max_risk=0.20, min_risk=0.01):
    """
    Уменьшает риск по мере роста капитала. При 2x росте — риск в 2 раза меньше.
    Это строгий консервативный режим: чем больше баланс, тем меньше риск.

    :param current_balance: текущий баланс
    :param initial_balance: начальный баланс
    :param base: риск при начальном балансе
    :param max_risk: потолок (на случай, если баланс резко упал)
    :param min_risk: минимум риска при большом росте
    """
    if current_balance <= 0:
        return max_risk

    ratio = initial_balance / current_balance
    raw_risk = base * ratio
    risk = max(min_risk, min(raw_risk, max_risk))

    logger.debug(
        f"Risk vs capital | balance: {current_balance:.2f}, initial: {initial_balance:.2f}, "
        f"raw: {raw_risk:.4f}, clipped: {risk:.4f}"
    )
    return risk
