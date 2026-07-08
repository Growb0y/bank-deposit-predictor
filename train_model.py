import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

CHART_DIR = Path('static/charts')
CHART_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_MAP = {
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

TARGET_MAP = {'да': 1, 'нет': 0}
YES_NO_MAP = {'да': 1, 'нет': 0}

cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

FEATURE_RU = {
    'age':'Возраст','job':'Работа','marital':'Семейное положение','education':'Образование',
    'default':'Дефолт','balance':'Среднегодовой баланс, евро','housing':'Наличие жилищного кредита',
    'loan':'Наличие потребительского кредита','contact':'Тип контактной связи',
    'month':'Месяц последнего контакта','duration':'Длительность разговора, сек',
    'campaign':'Количество контактов','pdays':'Дней после контакта',
    'previous':'Контактов до кампании','poutcome':'Результат прошлой кампании','y':'Оформил депозит',
}
MONTH_RU = {'jan':'янв','feb':'фев','mar':'мар','apr':'апр','may':'май','jun':'июн',
            'jul':'июл','aug':'авг','sep':'сен','oct':'окт','nov':'ноя','dec':'дек'}


def load_data():
    df = pd.read_excel('данные.xlsx', engine='openpyxl')
    df = df.rename(columns=COLUMN_MAP)
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['y'] = df['y'].map(TARGET_MAP)
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map(YES_NO_MAP)
    return df


def save_plot(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path)


def plot_target_distribution(df):
    # y=0 = не оформил, y=1 = оформил
    n_not = int((df['y'] == 0).sum())
    n_sub = int((df['y'] == 1).sum())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#ff6b6b', '#51cf66']
    labels = ['Не оформил', 'Оформил']
    ax1.bar(labels, [n_not, n_sub], color=colors)
    for i, v in enumerate([n_not, n_sub]):
        ax1.text(i, v + 200, str(v), ha='center', fontweight='bold')
    ax1.set_title('Распределение целевой переменной')
    ax1.set_ylabel('Количество клиентов')
    wedges, texts, autotexts = ax2.pie(
        [n_not, n_sub], labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0, 0.05))
    ax2.set_title('Доля оформивших депозит')
    plt.tight_layout()
    return save_plot(fig, 'target_distribution.png')


def plot_hist_with_edges(ax, data0, data1, bins, xlabel, title):
    n0, _, patches0 = ax.hist(data0, bins=bins, alpha=0.25, color='#ff6b6b', label='Не оформил', align='left')
    for p in patches0:
        p.set_edgecolor('#ff6b6b')
        p.set_linewidth(0.8)
    n1, _, patches1 = ax.hist(data1, bins=bins, alpha=0.25, color='#51cf66', label='Оформил', align='left')
    for p in patches1:
        p.set_edgecolor('#51cf66')
        p.set_linewidth(0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Количество клиентов')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(bins[::max(1, len(bins)//12)])
    ax.xaxis.set_minor_locator(plt.MultipleLocator((bins[1]-bins[0])*2))
    ax.set_xlim(bins[0], bins[-1])


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    age_min = max(18, df['age'].min())
    bins = list(range(age_min, 95, 3))
    plot_hist_with_edges(ax, df[df['y'] == 0]['age'], df[df['y'] == 1]['age'],
                         bins, 'Возраст', 'Распределение возраста')
    plt.tight_layout()
    return save_plot(fig, 'age_distribution.png')


def plot_categorical_by_target(df, col, title, filename, figsize=(8, 4)):
    ct = pd.crosstab(df[col], df['y'], normalize='index')
    ct = ct.sort_values(1, ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#ff6b6b', '#51cf66'])
    ax.set_title(title)
    ax.set_xlabel(FEATURE_RU.get(col, col))
    ax.set_ylabel('Доля')
    ax.legend(['Не оформил', 'Оформил'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # Map 0/1 to нет/да on binary features
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    if set(tick_labels) <= {'0', '1'}:
        ax.set_xticklabels(['нет' if t == '0' else 'да' for t in tick_labels], rotation=45, ha='right')
    for i in range(len(ct)):
        total = ct.iloc[i].sum()
        if total > 0:
            ax.text(i, ct.iloc[i, 0] / 2, f'{ct.iloc[i, 0]:.0%}', ha='center', fontsize=8)
            ax.text(i, ct.iloc[i, 0] + ct.iloc[i, 1] / 2, f'{ct.iloc[i, 1]:.0%}', ha='center', fontsize=8)
    plt.tight_layout()
    return save_plot(fig, filename)


def plot_balance_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    df0 = df[df['y'] == 0]['balance']
    df1 = df[df['y'] == 1]['balance']
    bins = np.linspace(-1000, 20000, 42)
    plot_hist_with_edges(ax, df0, df1, bins,
                         'Среднегодовой баланс, евро', 'Распределение баланса')
    ax.set_xlim(-1500, 21000)
    ax.set_xticks(range(-1000, 21001, 3000))
    plt.tight_layout()
    return save_plot(fig, 'balance_distribution.png')


def plot_duration_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    df0 = df[df['y'] == 0]['duration']
    df1 = df[df['y'] == 1]['duration']
    bins = np.linspace(0, 2000, 40)
    plot_hist_with_edges(ax, df0, df1, bins,
                         'Длительность контакта, секунд', 'Распределение длительности звонка')
    ax.set_xlim(-50, 2050)
    ax.set_xticks(range(0, 2001, 250))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    plt.tight_layout()
    return save_plot(fig, 'duration_distribution.png')


def plot_correlation_heatmap(df):
    corr = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'y']].corr()
    corr = corr.rename(index=FEATURE_RU, columns=FEATURE_RU)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Корреляционная матрица числовых признаков')
    plt.tight_layout()
    return save_plot(fig, 'correlation_heatmap.png')


def plot_monthly_trend(df):
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df_m = df.copy()
    df_m['month'] = pd.Categorical(df_m['month'], categories=month_order, ordered=True)
    ct = df_m.groupby('month', observed=True)['y'].agg(['count', 'mean'])
    ct.index = [MONTH_RU.get(m, m) for m in ct.index]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(ct.index, ct['count'], color='#adb5bd', alpha=0.6, label='Всего контактов')
    ax1.set_ylabel('Количество контактов')
    ax2 = ax1.twinx()
    ax2.plot(ct.index, ct['mean'] * 100, color='#51cf66', marker='o', linewidth=2, label='Конверсия, %')
    ax2.set_ylabel('Конверсия, %')
    ax2.set_ylim(0, max(ct['mean'] * 100) * 1.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('Количество контактов и конверсия по месяцам')
    plt.tight_layout()
    return save_plot(fig, 'monthly_trend.png')


def plot_contact_analysis(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df['contact'], df['y'], normalize='index')
    bars = ct.plot(kind='bar', stacked=True, ax=ax, color=['#ff6b6b', '#51cf66'], legend=False)
    ax.set_title('Конверсия по типу контакта')
    ax.set_ylabel('Доля')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel('')
    ax.legend(['Не оформил', 'Оформил'])
    for i in range(len(ct)):
        ax.text(i, ct.iloc[i, 0] / 2, f'{ct.iloc[i, 0]:.0%}', ha='center', fontsize=8)
        ax.text(i, ct.iloc[i, 0] + ct.iloc[i, 1] / 2, f'{ct.iloc[i, 1]:.0%}', ha='center', fontsize=8)
    plt.tight_layout()
    return save_plot(fig, 'contact_analysis.png')


def plot_campaign_analysis(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    df0 = df[df['y'] == 0]['campaign']
    df1 = df[df['y'] == 1]['campaign']
    bins = range(1, 31)
    plot_hist_with_edges(ax, df0, df1, bins,
                         'Количество звонков', 'Распределение количества звонков')
    plt.tight_layout()
    return save_plot(fig, 'campaign_analysis.png')


def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#339af0', linewidth=2, label=f'ROC-кривая (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#339af0')
    ax.set_xlabel('Доля ложноположительных')
    ax.set_ylabel('Доля истинноположительных')
    ax.set_title('ROC-кривая')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = save_plot(fig, 'roc_curve.png')
    return path, auc


def plot_feature_importance(model, feature_names):
    importance = model.get_feature_importance()
    idx = np.argsort(importance)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), importance[idx], color='#339af0')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEATURE_RU.get(feature_names[i], feature_names[i]) for i in idx])
    ax.set_xlabel('Важность')
    ax.set_title('Топ-15 важных признаков')
    ax.invert_yaxis()
    plt.tight_layout()
    return save_plot(fig, 'feature_importance.png')


def generate_analysis_report(df, y_test, y_pred, y_pred_proba, model, feature_names, metrics):
    report = {
        'dataset': {
            'total_clients': int(len(df)),
            'subscribed': int(df['y'].sum()),
            'not_subscribed': int((1 - df['y']).sum()),
            'conversion_rate': float(df['y'].mean()),
        },
        'metrics': metrics,
        'client_profile': {
            'age': {
                'mean': float(df['age'].mean()),
                'std': float(df['age'].std()),
                'min': int(df['age'].min()),
                'max': int(df['age'].max()),
            },
            'balance': {
                'mean': float(df['balance'].mean()),
                'median': float(df['balance'].median()),
            },
            'duration': {
                'mean': float(df['duration'].mean()),
                'median': float(df['duration'].median()),
            },
        },
    }

    # Top segments by conversion rate
    print('\n=== АНАЛИЗ СЕГМЕНТОВ ===')
    segments = []

    for col in ['job', 'marital', 'education', 'contact', 'poutcome', 'month']:
        grp = df.groupby(col)['y'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        top = grp.head(3)
        segments.append({
            'feature': col,
            'top_segments': [
                {'value': idx, 'count': int(row['count']), 'conversion': float(row['mean'])}
                for idx, row in top.iterrows()
            ]
        })
        print(f'\n{col.upper()}:')
        print(grp.to_string())

    # Cross-segment analysis: best clients
    print('\n=== ЛУЧШИЕ КЛИЕНТЫ (по poutcome=success, длит. разговора > медианы) ===')
    median_dur = df['duration'].median()
    best = df[(df['poutcome'] == 'успех') & (df['duration'] > median_dur)]
    print(f'Всего таких клиентов: {len(best)}')
    print(f'Конверсия в сегменте: {best["y"].mean():.2%}')
    print(f'Средний баланс: {best["balance"].mean():.0f}')

    # Worst clients
    print('\n=== ХУДШИЕ КЛИЕНТЫ (никогда не было контакта, короткий разговор) ===')
    worst = df[(df['pdays'] == -1) & (df['duration'] < df['duration'].median() / 2)]
    print(f'Всего: {len(worst)}')
    print(f'Конверсия: {worst["y"].mean():.2%}')

    with open('analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report, segments


def generate_recommendations(df, model, feature_names, segments):
    recommendations = []

    # Overall conversion rate
    conv = df['y'].mean()
    recommendations.append({
        'title': 'Общая конверсия',
        'text': f'Текущая конверсия в оформление депозита составляет {conv:.1%}. '
                f'Это означает, что из 100 звонков в среднем {conv*100:.0f} клиентов оформляют депозит. '
                f'Основная цель модели — повысить эффективность обзвона за счёт приоритизации клиентов с высокой вероятностью.',
        'type': 'info'
    })

    # Best month
    month_conv = df.groupby('month')['y'].mean()
    best_month = month_conv.idxmax()
    recommendations.append({
        'title': 'Оптимальное время для кампании',
        'text': f'Наивысшая конверсия наблюдается в месяце {best_month.upper()}. '
                f'Рекомендуется планировать основные обзвоны на этот период. '
                f'Наихудшие результаты — в месяцах с минимальной конверсией.',
        'type': 'strategy'
    })

    # Contact type
    contact_conv = df.groupby('contact')['y'].mean()
    best_contact = contact_conv.idxmax()
    recommendations.append({
        'title': 'Тип контакта',
        'text': f'Клиенты, с которыми связывались через "{best_contact}", '
                f'оформляют депозит чаще. Рекомендуется отдавать предпочтение этому каналу связи.',
        'type': 'strategy'
    })

    # Poutcome
    poutcome_conv = df.groupby('poutcome')['y'].mean()
    best_poutcome = poutcome_conv.idxmax()
    worst_poutcome = poutcome_conv.idxmin()
    recommendations.append({
        'title': 'История предыдущих кампаний',
        'text': f'Клиенты с результатом "{best_poutcome}" имеют наибольшую конверсию ({poutcome_conv.max():.1%}). '
                f'Клиенты с результатом "{worst_poutcome}" — наименьшую ({poutcome_conv.min():.1%}). '
                f'Рекомендуется фокусироваться на клиентах без истории предыдущих кампаний '
                f'(конверсия {poutcome_conv.get("неизвестно", 0):.1%}), '
                f'так как они наиболее отзывчивы на предложение.',
        'type': 'strategy'
    })

    # Duration analysis
    median_dur = df['duration'].median()
    recommendations.append({
        'title': 'Длительность разговора',
        'text': f'Медианная длительность разговора с оформившими — {df[df["y"]==1]["duration"].median():.0f} сек, '
                f'с неоформившими — {df[df["y"]==0]["duration"].median():.0f} сек. '
                f'Рекомендуется уделять больше времени заинтересованным клиентам, '
                f'но не затягивать разговор с незаинтересованными (короткий звонок — признак отказа).',
        'type': 'insight'
    })

    # Job type
    job_conv = df.groupby('job')['y'].mean().sort_values(ascending=False)
    top_jobs = job_conv.head(3)
    recommendations.append({
        'title': 'Профессиональные группы',
        'text': f'Наиболее перспективные профессии: {", ".join(top_jobs.index)} '
                f'(конверсия {top_jobs.mean():.1%}). '
                f'Рекомендуется нацелиться на эти группы при холодных звонках.',
        'type': 'strategy'
    })

    # Education
    edu_conv = df.groupby('education')['y'].mean().sort_values(ascending=False)
    recommendations.append({
        'title': 'Уровень образования',
        'text': f'Клиенты с образованием "{edu_conv.index[0]}" имеют наивысшую конверсию ({edu_conv.iloc[0]:.1%}). '
                f'Это может быть связано с более высокой финансовой грамотностью.',
        'type': 'insight'
    })

    # Balance
    high_balance = df[df['balance'] > df['balance'].median()]
    low_balance = df[df['balance'] <= df['balance'].median()]
    recommendations.append({
        'title': 'Финансовое положение',
        'text': f'Клиенты с балансом выше медианы ({df["balance"].median():.0f} евро) имеют конверсию '
                f'{high_balance["y"].mean():.1%} против {low_balance["y"].mean():.1%} у клиентов с низким балансом. '
                f'Рекомендуется фокусироваться на клиентах со средним и высоким балансом.',
        'type': 'strategy'
    })

    # Campaign count
    campaign_0 = df[df['campaign'] <= 2]
    campaign_many = df[df['campaign'] > 5]
    direction = 'растёт' if campaign_many["y"].mean() > campaign_0["y"].mean() else 'падает'
    recommendations.append({
        'title': 'Количество звонков',
        'text': f'Клиенты, которым сделали 1-2 звонка, оформляют депозит в {campaign_0["y"].mean():.1%} случаев. '
                f'После 5+ звонков конверсия {direction} до {campaign_many["y"].mean():.1%}. '
                f'Рекомендуется анализировать причины повторных звонков для повышения эффективности.',
        'type': 'insight'
    })

    with open('recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)

    return recommendations


def train():
    print('=== Загрузка данных ===')
    df = load_data()
    print(f'Загружено {len(df)} записей, {df.shape[1]} признаков')
    print(f'Целевая переменная: 0={df["y"].value_counts()[0]}, 1={df["y"].value_counts()[1]}')
    print(f'Конверсия: {df["y"].mean():.2%}')

    print('\n=== Генерация визуализаций ===')
    charts = {}
    charts['target_distribution'] = plot_target_distribution(df)
    charts['age_distribution'] = plot_age_distribution(df)
    charts['balance_distribution'] = plot_balance_distribution(df)
    charts['duration_distribution'] = plot_duration_distribution(df)
    charts['correlation_heatmap'] = plot_correlation_heatmap(df)
    charts['monthly_trend'] = plot_monthly_trend(df)
    charts['contact_analysis'] = plot_contact_analysis(df)
    charts['campaign_analysis'] = plot_campaign_analysis(df)

    for col, title, fname in [
        ('job', 'Распределение профессий', 'job_conversion.png'),
        ('marital', 'Семейное положение', 'marital_conversion.png'),
        ('education', 'Образование', 'education_conversion.png'),
        ('housing', 'Наличие жилищного кредита', 'housing_conversion.png'),
        ('loan', 'Наличие потребительского кредита', 'loan_conversion.png'),
    ]:
        charts[fname.replace('.png', '')] = plot_categorical_by_target(df, col, title, fname)

    print('Визуализации сохранены в static/charts/')

    print('\n=== Подготовка данных для модели ===')
    feature_cols = [c for c in df.columns if c != 'y']
    X = df[feature_cols].copy()
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')

    # Determine which features are categorical for CatBoost
    cat_indices = [i for i, col in enumerate(feature_cols) if col in cat_features]
    print(f'Categorical features: {[feature_cols[i] for i in cat_indices]}')

    print('\n=== Обучение CatBoost ===')
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        cat_features=cat_indices,
        eval_metric='AUC',
        random_seed=42,
        verbose=50,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)

    print('\n=== Оценка модели ===')
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC: {auc:.4f}')

    metrics = classification_report(y_test, y_pred, output_dict=True, target_names=['Не оформил', 'Оформил'])
    print(classification_report(y_test, y_pred, target_names=['Не оформил', 'Оформил']))

    # ROC curve
    roc_path, auc_val = plot_roc_curve(y_test, y_pred_proba)
    charts['roc_curve'] = roc_path
    print(f'ROC-AUC: {auc_val:.4f}')

    charts['feature_importance'] = plot_feature_importance(model, feature_cols)

    print('\n=== Сохранение модели ===')
    model.save_model('model/catboost_model.cbm')
    print('Модель сохранена: model/catboost_model.cbm')

    # Generate analysis report
    report, segments = generate_analysis_report(df, y_test, y_pred, y_pred_proba, model, feature_cols, metrics)
    print(f'Аналитический отчёт сохранён в analysis_results.json')

    # Generate recommendations
    recommendations = generate_recommendations(df, model, feature_cols, segments)
    print(f'Рекомендации сохранены в recommendations.json')

    # Save charts manifest
    charts_manifest = {k: str(Path(v).relative_to('.')) for k, v in charts.items()}
    with open('charts_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(charts_manifest, f, ensure_ascii=False, indent=2)

    print('\n=== ГОТОВО ===')
    print(f'Модель: model/catboost_model.cbm')
    print(f'Чарты: {len(charts)} шт.')
    print(f'AUC: {auc:.4f}')


if __name__ == '__main__':
    train()
