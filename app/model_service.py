import json
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

MODEL_PATH = 'model/catboost_model.cbm'
COLUMN_MAP_RU = {
    'Возраст': 'age',
    'Работа': 'job',
    'Семейное положение': 'marital',
    'Образование': 'education',
    'Дефолт': 'default',
    'Среднегодовой баланс, в евро': 'balance',
    'Наличие жилищного кредита': 'housing',
    'Наличие потребительского кредита': 'loan',
    'Тип контактной связи ': 'contact',
    'Последний контакт, месяц': 'month',
    'Длительность контакта, секунд': 'duration',
    'Количество контактов, выполненных во время этой кампании и для данного клиента (числовое, включая последний контакт)': 'campaign',
    'Количество дней, прошедших с момента последнего контакта с клиентом из предыдущей кампании (число, -1 означает, что с клиентом ранее не связывались)': 'pdays',
    'Количество контактов, выполненных до этой кампании и для этого клиента': 'previous',
    'Результат предыдущей маркетинговой кампании': 'poutcome',
    'Оформил ли клиент срочный депозит': 'y',
}

YES_NO_MAP = {'да': 1, 'нет': 0, 'yes': 1, 'no': 0}

cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

FEATURE_ORDER = ['age', 'job', 'marital', 'education', 'default', 'balance',
                 'housing', 'loan', 'contact', 'month', 'duration', 'campaign',
                 'pdays', 'previous', 'poutcome']

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f'Model not found at {MODEL_PATH}. Run train_model.py first.')
        _model = CatBoostClassifier()
        _model.load_model(MODEL_PATH)
    return _model


def predict_proba(client_data: dict) -> float:
    model = get_model()
    df = pd.DataFrame([client_data])
    df = df[FEATURE_ORDER]
    for col in ['default', 'housing', 'loan']:
        if col in df.columns:
            df[col] = df[col].map(lambda x: YES_NO_MAP.get(str(x).lower(), 0))
    proba = model.predict_proba(df)[0, 1]
    return float(proba)


def predict(client_data: dict, threshold: float = 0.5) -> dict:
    proba = predict_proba(client_data)
    pred = 1 if proba >= threshold else 0
    label = '✅ Оформит депозит' if pred == 1 else '❌ Не оформит'
    return {
        'probability': round(proba, 4),
        'prediction': pred,
        'prediction_label': label,
    }


def predict_batch(clients: list, threshold: float = 0.5) -> dict:
    results = []
    for c in clients:
        results.append(predict(c, threshold))
    probabilities = [r['probability'] for r in results]
    summary = {
        'total': len(results),
        'predicted_positive': sum(1 for r in results if r['prediction'] == 1),
        'predicted_negative': sum(1 for r in results if r['prediction'] == 0),
        'avg_probability': round(float(np.mean(probabilities)), 4),
        'max_probability': round(float(np.max(probabilities)), 4),
        'min_probability': round(float(np.min(probabilities)), 4),
    }
    return {'results': results, 'summary': summary}


def format_client_data(data: dict) -> dict:
    return {k: v for k, v in data.items()}
