import hashlib
import json
import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Возвращает MD5-хэш содержимого DataFrame.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def hash_dict(d: dict) -> str:
    """
    Возвращает MD5-хэш словаря.
    """
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def hash_params(params: dict) -> str:
    """
    Хэширует параметры стратегии. Подходит для использования в кэшах.
    """
    return hash_dict(params)
