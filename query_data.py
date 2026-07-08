import pandas as pd

df = pd.read_excel('данные.xlsx', engine='openpyxl')

poutcome_col = 'Результат предыдущей маркетинговой кампании'
target_col = 'Оформил ли клиент срочный депозит'

success = df[df[poutcome_col] == 'успех']
print(f'Всего с результатом "успех": {len(success)}')
print(f'Из них оформили депозит: {(success[target_col]=="да").sum()}')
print(f'Из них НЕ оформили: {(success[target_col]=="нет").sum()}')
conv = (success[target_col]=="да").mean() * 100
print(f'Конверсия в сегменте: {conv:.1f}%')
