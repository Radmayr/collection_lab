"""RTK-пайплайн на collection_lab: данные → сплит → отбор → обучение → Optuna → отчёт.

Запуск: python examples/rtk_pipeline.py [путь к df_agg.csv]
"""

import sys
import time

import pandas as pd

import collection_lab as cl

DATA = sys.argv[1] if len(sys.argv) > 1 else "D:/ML_lib/df_agg.csv"
TARGET, DATE = "target", "rtk_send_date"
t0 = time.time()


def step(msg):
    print(f"\n=== [{time.time() - t0:6.1f}s] {msg}")


# 1. Данные и типы
step("Загрузка")
df = pd.read_csv(DATA, low_memory=False)
df[DATE] = pd.to_datetime(df[DATE], format="ISO8601").dt.normalize()
features = [c for c in df.columns
            if not c.endswith(("_dt", "_dttm", "_rk", "_date")) and "target" not in c.lower()
            and c not in ["sample", "contract_number", DATE, "financial_account_subtype_cd",
                          "parent_financial_account_subtype_cd"]]
num_cols, cat_cols = cl.data.split_feature_types(df, features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col=TARGET)
print(df.shape, f"признаков: {len(features)} (кат.: {len(cat_cols)})")

# 2. Сплит без утечек: test — OOT, отбор только на train
split = cl.data.time_split(df, TARGET, DATE, oot_from="2024-03-01", val_size=0.2)
print(split.summary(DATE))

# 3. Отбор признаков
step("Отбор признаков")
pipe = cl.selection.SelectionPipeline([
    cl.selection.QualityFilter(num_missing_threshold=0.999, num_zero_threshold=0.999,
                               num_zero_nan_threshold=0.999, cat_unique_max=300,
                               cat_missing_threshold=0.999),
    cl.selection.CorrelationFilter(threshold=0.8, strategy="hybrid", n_splits=5),
    cl.selection.CumulativeImportance(threshold=0.9),
    cl.selection.RFE(tol=0.01, n_splits=3, min_features=5, params={"n_estimators": 200},
                     verbose=False),
]).fit(split.train, TARGET, features, cat_features=cat_cols)
print(pipe.summary())
selected = pipe.selected_
cats = [c for c in cat_cols if c in selected]
print("Отобрано:", selected)

# 4. Forward-оценка отобранного набора
step("Incremental forward")
fwd = cl.selection.incremental_feature_eval(split.train, split.train[TARGET], selected,
                                            cat_features=cats, n_splits=5, verbose=False)
print(fwd.table[["n_features", "feature_changed", "auc_mean", "auc_signal_ratio"]])

# 5. Базовая модель и Optuna (выбор по val)
step("Обучение и Optuna")
base_params = {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 3, "num_leaves": 8}
base = cl.modeling.train_model(split, selected, cat_features=cats, params=base_params)
print("base:", {k: round(v, 4) for k, v in base.scores_.items()})
tune = cl.modeling.tune_hyperparams(split, selected, cat_features=cats, n_trials=30,
                                    fixed_params={"n_estimators": 1000}, verbose=False)
print("tune best:", round(tune.info["best_val"], 4), "test:", round(tune.info["best_test"], 4))
final = cl.modeling.train_model(split, selected, cat_features=cats,
                                params=tune.info["best_params"], early_stopping_rounds=None)
print("final:", {k: round(v, 4) for k, v in final.scores_.items()})

# 6. Отчёт, стабильность, эксперимент
step("Отчёт")
report = cl.validation.model_report(final, split, date_col=DATE, n_buckets=20)
print(report.metrics.round(4))
print(report.calibration)
figs = cl.validation.plot_stab(df["age"], df[TARGET], df[DATE], 5, feature_nm="age",
                               period="M", return_plotly_fig=True)

step("Сохранение эксперимента")
exp = cl.tracking.Experiment("RTK_model", root="experiments")
exp.log_params(tune.info["best_params"], name="model")
exp.log_metrics({f"auc_{k}": v for k, v in final.scores_.items()})
exp.save_features(selected, cat_features=cats, target=TARGET)
exp.save_model(final)
exp.save_pipeline(pipe)   # таблицы и графики: локально и в ClearML (если clearml=True)
exp.save_report(report)
print(exp)
exp.close()
step("Готово")
