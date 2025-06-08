import pandas as pd
import json
from trading_analysis.db import SessionLocal
from trading_analysis.models import ModelRun

def load_model_runs_df() -> pd.DataFrame:
    """Загружает все запуски модели из БД и нормализует параметры."""
    session = SessionLocal()
    runs = session.query(ModelRun).order_by(ModelRun.date.asc()).all()
    session.close()

    records = []
    for r in runs:
        try:
            params = json.loads(r.params_json) if r.params_json else {}
            if not isinstance(params, dict):
                params = {}
        except json.JSONDecodeError:
            params = {}

        base = {
            "date": r.date,
            "pnl": r.pnl,
            "winrate": r.winrate,
            "retrained": r.retrained
        }

        # Объединяем, только если params — словарь
        full_record = base | params  # Python 3.9+
        records.append(full_record)

    df = pd.DataFrame(records)

    if df.empty:
        print("❌ Нет записей model_runs в базе данных.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df.sort_index()
