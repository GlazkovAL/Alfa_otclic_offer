

Решение задачи бинарной классификации: нужно предсказать вероятность того, что клиент согласится на кредитный оффер. Основная метрика - ROC-AUC.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
python src/alfa_credit_offer_solution.py --train data/train_apps.csv --test data/test_apps.csv --output outputs
```


## Идея решения

1. Используется временная валидация:

```text
train_part: decision_day < 2025-04-01
valid_part: decision_day >= 2025-04-01
```

Такой сплит ближе к реальному тесту, потому что `test_apps.csv` лежит в будущем относительно большей части train.

2. Строятся признаки:

- календарные признаки из `decision_day`;
- missing flags для пропусков;
- признаки по ставке относительно `cb_rate`;
- признаки по лимитам;
- признаки финансовой активности за 30/90 дней;
- комбинация `db_group_last` и `fl_adminarea`;
- аккуратные proxy-признаки `front_id` для LightGBM/XGBoost.

3. Обучаются три разных семейства бустинга:

- CatBoost;
- LightGBM;
- XGBoost.

4. На temporal validation подбираются веса blend.

Лучшая локальная комбинация в эксперименте:

```text
CatBoost:  0.186
LightGBM:  0.334
XGBoost:   0.479
```

5. Финальные модели обучаются на всём подходящем train и смешиваются найденными весами.
