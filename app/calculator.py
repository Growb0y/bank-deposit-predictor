import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .model_service import get_model, FEATURE_ORDER, YES_NO_MAP


def calculate_strategy(cost_per_call: float, profit_per_deposit: float, budget: float = None, df: pd.DataFrame = None) -> dict:
    model = get_model()

    if df is None:
        df = pd.read_excel('данные.xlsx', engine='openpyxl')
        col_map = {
            'Возраст': 'age', 'Работа': 'job', 'Семейное положение': 'marital',
            'Образование': 'education', 'Дефолт': 'default',
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
        df = df.rename(columns=col_map)
        df['y'] = df['y'].map({'да': 1, 'нет': 0})
    else:
        # df already has 'probability' column from prediction
        pass

    if 'probability' not in df.columns:
        X = df[FEATURE_ORDER].copy()
        for col in ['default', 'housing', 'loan']:
            X[col] = X[col].map(lambda x: YES_NO_MAP.get(str(x).lower(), 0))
        probabilities = model.predict_proba(X)[:, 1]
        df['probability'] = probabilities

    # Compute expected value for each client
    df['expected_value'] = df['probability'] * profit_per_deposit - cost_per_call
    df['expected_profit_if_positive'] = df['probability'] * profit_per_deposit - cost_per_call

    # Sort by expected value descending
    df_sorted = df.sort_values('expected_value', ascending=False).reset_index(drop=True)
    df_sorted['cumulative_cost'] = (np.arange(len(df_sorted)) + 1) * cost_per_call
    df_sorted['cumulative_expected_revenue'] = df_sorted['probability'].cumsum() * profit_per_deposit
    df_sorted['cumulative_expected_profit'] = df_sorted['cumulative_expected_revenue'] - df_sorted['cumulative_cost']

    # Find optimal threshold
    thresholds = np.linspace(0.01, 0.99, 200)
    best_profit = -np.inf
    best_threshold = 0.5
    best_n_calls = 0

    for thresh in thresholds:
        mask = df['probability'] >= thresh
        n_calls = mask.sum()
        if n_calls == 0:
            continue
        total_cost = n_calls * cost_per_call
        total_revenue = df.loc[mask, 'probability'].sum() * profit_per_deposit
        total_profit = total_revenue - total_cost

        if budget is not None and total_cost > budget:
            continue

        if total_profit > best_profit:
            best_profit = total_profit
            best_threshold = thresh
            best_n_calls = n_calls

    # Apply optimal threshold
    mask = df['probability'] >= best_threshold
    clients_to_call = mask.sum()
    if clients_to_call == 0:
        return {
            'optimal_threshold': 0.5,
            'total_clients': len(df),
            'clients_to_call': 0,
            'expected_deposits': 0,
            'expected_cost': 0,
            'expected_revenue': 0,
            'expected_profit': 0,
            'roi': 0,
            'conversion_rate': 0,
            'recommendations': ['Нет клиентов для обзвона с заданными параметрами.']
        }

    selected = df.loc[mask]
    expected_deposits = selected['probability'].sum()
    expected_cost = clients_to_call * cost_per_call
    expected_revenue = expected_deposits * profit_per_deposit
    expected_profit = expected_revenue - expected_cost
    roi = expected_profit / expected_cost if expected_cost > 0 else 0

    # If budget constraint
    budget_used = None
    if budget is not None:
        df_sorted_filtered = df_sorted[df_sorted['cumulative_cost'] <= budget]
        budget_n = len(df_sorted_filtered)
        if budget_n > 0:
            budget_selected = df_sorted_filtered
            budget_expected_deposits = budget_selected['probability'].sum()
            budget_expected_revenue = budget_expected_deposits * profit_per_deposit
            budget_expected_cost = budget_n * cost_per_call
            budget_expected_profit = budget_expected_revenue - budget_expected_cost
            budget_used = budget_expected_cost

            if budget_expected_profit > expected_profit:
                clients_to_call = budget_n
                expected_deposits = budget_expected_deposits
                expected_cost = budget_expected_cost
                expected_revenue = budget_expected_revenue
                expected_profit = budget_expected_profit
                best_threshold = None

    conversion_rate = expected_deposits / clients_to_call if clients_to_call > 0 else 0

    # Generate recommendations
    recommendations = []
    if clients_to_call > 0:
        recommendations.append(
            f'Рекомендуется обзвонить {clients_to_call} клиентов из {len(df)} ({clients_to_call/len(df)*100:.1f}%).'
        )
        recommendations.append(
            f'Ожидаемое количество оформленных депозитов: {expected_deposits:.0f} '
            f'(конверсия {conversion_rate:.1%}).'
        )
        recommendations.append(
            f'Ожидаемая прибыль: {expected_profit:.0f} евро '
            f'(выручка: {expected_revenue:.0f}, затраты: {expected_cost:.0f}).'
        )
        recommendations.append(f'ROI: {roi:.1%}.')

        if best_threshold is not None:
            recommendations.append(
                f'Оптимальный порог вероятности: {best_threshold:.2f}. '
                f'Клиенты с вероятностью >= {best_threshold:.2f} будут обзвонены.'
            )

        # Profile the best clients
        top_clients = selected.nlargest(5, 'expected_value')
        top_jobs = top_clients['job'].value_counts().index.tolist()[:3]
        recommendations.append(
            f'Наиболее перспективные клиенты: профессии {", ".join(top_jobs)}.'
        )

        # Compare with random calling
        if 'y' in df.columns:
            random_profit = df['y'].mean() * len(df) * profit_per_deposit - len(df) * cost_per_call
            if expected_profit > random_profit:
                recommendations.append(
                    f'Модель даёт на {expected_profit - random_profit:.0f} евро больше, чем случайный обзвон всех клиентов.'
                )
            else:
                recommendations.append(
                    f'Рандомный обзвон всех даёт {random_profit:.0f} евро.'
                )
    else:
        recommendations.append(
            'Ни один клиент не является прибыльным при заданных параметрах. '
            'Увеличьте прибыль с депозита или уменьшите стоимость звонка.'
        )

    # Profit curve data (sample roughly for chart)
    step = max(1, len(df_sorted) // 50)
    profit_curve_x = (np.arange(0, len(df_sorted), step) + 1).tolist()
    profit_curve_y = df_sorted['cumulative_expected_profit'].iloc[::step].tolist()
    if profit_curve_x[-1] != len(df_sorted):
        profit_curve_x.append(len(df_sorted))
        profit_curve_y.append(float(df_sorted['cumulative_expected_profit'].iloc[-1]))
    optimal_idx = min(clients_to_call, len(df_sorted)) - 1
    optimal_profit = float(df_sorted['cumulative_expected_profit'].iloc[optimal_idx])

    return {
        'optimal_threshold': float(best_threshold) if best_threshold is not None else None,
        'total_clients': int(len(df)),
        'clients_to_call': int(clients_to_call),
        'expected_deposits': float(round(expected_deposits, 1)),
        'expected_cost': float(round(expected_cost, 1)),
        'expected_revenue': float(round(expected_revenue, 1)),
        'expected_profit': float(round(expected_profit, 1)),
        'roi': float(round(roi, 4)),
        'conversion_rate': float(round(conversion_rate, 4)),
        'budget_used': float(round(budget_used, 1)) if budget_used is not None else None,
        'recommendations': recommendations,
        'profit_curve_x': profit_curve_x,
        'profit_curve_y': profit_curve_y,
        'optimal_n_calls': int(clients_to_call),
        'optimal_profit': round(optimal_profit, 1),
    }
