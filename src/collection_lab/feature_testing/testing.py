"""Тестирование новых признаков и доменов данных относительно базовой модели.

Вопрос один: «что даст модели добавление этих признаков?». Ответ — прирост метрик с
доверительным интервалом, в целом и по сегментам.

Что передать в ``base``:

- список признаков — база и «база + кандидаты» обучаются с одинаковыми параметрами;
- обученную модель (адаптер библиотеки, ``.pkl`` эксперимента, папку экспорта, нативный
  LightGBM / CatBoost) — признаки и гиперпараметры берутся из неё;
- имя колонки со скором готовой модели — модель не трогается, кандидаты дообучаются поверх скора.

Способ сравнения ``how``:

- ``"retrain"`` — переобучить с кандидатами: «какой станет модель»;
- ``"on_top"`` — поверх скора базы обучается LightGBM только на кандидатах: «есть ли в кандидатах
  то, чего нет в модели». Если скор на train посчитан моделью, обученной на этих же строках,
  прирост по CV занижен (модель «помнит» train) — ориентир в этом режиме даёт test.

**Без утечки.** Вердикт ставится по кросс-валидации на train; val нужен для ранней остановки,
test только показывается.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import expit
from scipy.stats import t as student_t
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from collection_lab.config import LGBM_PARAMS, RANDOM_STATE, merge_params
from collection_lab.core.cv import Folds, make_folds
from collection_lab.core.models import (
    BaseModel,
    FeaturePreparer,
    _lgbm_has_eval_xy,
    make_model,
)
from collection_lab.core.results import Result, _jsonable, _maybe_send
from collection_lab.data.split import DataSplit
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.calibration import prob_to_logit
from collection_lab.metrics.classification import Metric, get_metric
from collection_lab.metrics.stability import psi_by_period
from collection_lab.plotting.theme import NEGATIVE, POSITIVE, color, style
from collection_lab.selection.univariate import direct_auc
from collection_lab.utils.io import write_csv

DEFAULT_METRICS = ("auc", "gini", "ks", "logloss", "pr_auc", "brier", "lift@10%")
ALL = "ALL"  # строка «все кандидаты вместе»
VERDICT_COLORS = {"better": POSITIVE, "worse": NEGATIVE, "same": "#7f7f7f"}

# Быстрая модель для предварительного отсева, когда кандидатов больше max_exact.
SCREEN_PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 8,
                 "min_child_samples": 50, "verbosity": -1, "n_jobs": -1,
                 "random_state": RANDOM_STATE}
SCREEN_STOPPING = 20
TOP_STOPPING = 50      # ранняя остановка в режиме on_top
CORR_SAMPLE = 100_000  # строк train для корреляций и однофакторного AUC
MIN_SEGMENT = 30       # сегменты меньше этого размера в разрезах не показываются

_BOOSTER_KEYS = ("learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
                 "min_sum_hessian_in_leaf", "bagging_fraction", "bagging_freq",
                 "feature_fraction", "lambda_l1", "lambda_l2", "min_gain_to_split", "max_bin",
                 "scale_pos_weight")
_SKIP_PARAMS = {"importance_type", "silent", "class_weight", "objective", "verbose",
                "logging_level", "train_dir", "loss_function", "eval_metric", "use_best_model",
                "od_type", "od_wait", "early_stopping_rounds"}


# --- база ---------------------------------------------------------------------------------

@dataclass
class _Base:
    features: list[str]              # пусто, если есть только скор
    cat_features: list[str] | None
    template: BaseModel | None       # необученный адаптер с параметрами базы
    scorer: Any                      # имя колонки скора | обученный адаптер | None
    label: str


def _native_spec(obj) -> tuple[str, list[str], list[str] | None, dict[str, Any]]:
    """Тип модели, признаки, категориальные и параметры нативной модели LightGBM / CatBoost."""
    module = type(obj).__module__
    if module.startswith("lightgbm"):
        if hasattr(obj, "get_params"):  # sklearn-API
            params = {k: v for k, v in obj.get_params().items()
                      if v is not None and k not in _SKIP_PARAMS}
            return "lgbm", list(obj.feature_name_), None, params
        params = {k: obj.params[k] for k in _BOOSTER_KEYS if k in (obj.params or {})}
        return "lgbm", list(obj.feature_name()), None, params
    if module.startswith("catboost"):
        names = list(obj.feature_names_)
        cats = [names[i] for i in obj.get_cat_feature_indices()]
        params = {k: v for k, v in (obj.get_params() or {}).items() if k not in _SKIP_PARAMS}
        return "catboost", names, cats, params
    raise TypeError(
        f"base: не знаю, что делать с {type(obj).__name__}. Передайте список признаков, модель "
        "(collection_lab, LightGBM, CatBoost), путь к ней или имя колонки со скором.")


def _load_path(path: Path):
    """Модель с диска: ``.pkl`` эксперимента, папка экспорта, ``model.txt`` или ``.cbm``."""
    if path.is_dir():
        spec = json.loads((path / "preprocessing.json").read_text(encoding="utf-8"))
        kind, _, _, params = _native_spec(_load_path(path / spec["model_file"]))
        return kind, list(spec["features"]), list(spec["cat_features"]), params
    if path.suffix == ".cbm":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    if path.suffix in (".pkl", ".joblib"):
        import joblib

        return joblib.load(path)
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def _resolve_base(base, params: dict[str, Any] | None, columns: Iterable[str]) -> _Base:
    if isinstance(base, str) and base in set(columns):
        return _Base([], None, None, base, f"скор «{base}»")
    if isinstance(base, (str, Path)):
        path = Path(base)
        if not path.exists():
            raise ValueError(f"base={base!r}: в таблице нет такой колонки и нет такого файла.")
        base = _load_path(path)
    if isinstance(base, BaseModel):
        base._check_fitted()
        return _Base(list(base.features_), list(base.cat_features_), base.clone(params), base,
                     f"модель {base.name}, {len(base.features_)} признаков")
    if isinstance(base, (list, pd.Index)):
        return _Base(list(base), None, make_model("lgbm", params), None,
                     f"{len(base)} признаков")
    kind, features, cats, native = base if isinstance(base, tuple) else _native_spec(base)
    return _Base(features, cats, make_model(kind, merge_params(native, params)), None,
                 f"модель {kind}, {len(features)} признаков")


# --- контекст расчёта ---------------------------------------------------------------------

@dataclass
class _Context:
    parts: dict[str, pd.DataFrame]
    target: str
    last: str                       # часть для итоговой оценки: test, а без него — val
    how: str
    base: _Base
    cats: list[str]
    metrics: list[Metric]
    segment: str | None
    date_col: str | None
    cv_df: pd.DataFrame             # train (или его подвыборка) для кросс-валидации
    folds: Folds
    n_splits: int
    top_params: dict[str, Any]
    alpha: float
    n_boot: int
    random_state: int
    logit: dict[str, np.ndarray] = field(default_factory=dict)   # on_top: логит базы по частям
    cv_logit: np.ndarray | None = None
    base_run: dict[str, Any] = field(default_factory=dict)
    corr_base: pd.DataFrame | None = None
    corr_rows: np.ndarray | None = None

    @property
    def primary(self) -> Metric:
        return self.metrics[0]

    def y(self, part: str) -> np.ndarray:
        return self.parts[part][self.target].to_numpy()


def _fit_top(X_tr, y_tr, X_va, y_va, logit_tr, logit_va, cats, params, stopping):
    """LightGBM поверх готового скора (логит базы — стартовое значение)."""
    import lightgbm as lgb

    prep = FeaturePreparer(cats).fit(X_tr)
    est = lgb.LGBMClassifier(**{**params, "objective": "binary"})
    Xp_va = prep.transform(X_va)
    eval_kw = ({"eval_X": (Xp_va,), "eval_y": (y_va,)} if _lgbm_has_eval_xy()
               else {"eval_set": [(Xp_va, y_va)]})
    est.fit(prep.transform(X_tr), y_tr, init_score=logit_tr, eval_init_score=[logit_va],
            categorical_feature=cats or "auto",
            callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping, verbose=False)],
            **eval_kw)

    def predict(X, logit):
        return expit(logit + est.predict(prep.transform(X), raw_score=True))

    return est, predict


def _fit_predict(ctx: _Context, feats: list[str], tr: pd.DataFrame, va: pd.DataFrame,
                 apply: dict[str, pd.DataFrame], logit_tr=None, logit_va=None, logit_apply=None):
    """Обучает на ``tr`` (ранняя остановка по ``va``), возвращает прогноз на ``va``, прогнозы на
    ``apply`` и важности признаков."""
    y_tr, y_va = tr[ctx.target].to_numpy(), va[ctx.target].to_numpy()
    cats = [c for c in ctx.cats if c in feats]
    if ctx.how == "retrain":
        model = ctx.base.template.clone()
        model.fit(tr[feats], y_tr, eval_set=(va[feats], y_va), cat_features=cats)
        return (model.predict(va), {k: model.predict(d) for k, d in apply.items()},
                model.feature_importance("gain"))
    est, predict = _fit_top(tr[feats], y_tr, va[feats], y_va, logit_tr, logit_va, cats,
                            ctx.top_params, TOP_STOPPING)
    importance = pd.Series(est.booster_.feature_importance("gain"), index=feats, name="gain")
    return (predict(va[feats], logit_va),
            {k: predict(d[feats], logit_apply[k]) for k, d in apply.items()},
            importance.sort_values(ascending=False))


def _scores(ctx: _Context, y, p) -> dict[str, float]:
    return {m.name: m(y, p) for m in ctx.metrics}


def _run(ctx: _Context, feats: list[str]) -> dict[str, Any]:
    """Кросс-валидация на train и обучение на train → прогноз на val / test."""
    y_cv = ctx.cv_df[ctx.target].to_numpy()
    oof = np.full(len(ctx.cv_df), np.nan)
    fold_scores = []
    on_top = ctx.how == "on_top"
    for j, (tr_i, va_i) in enumerate(ctx.folds):
        logits = (ctx.cv_logit[tr_i], ctx.cv_logit[va_i]) if on_top else (None, None)
        pred, _, _ = _fit_predict(ctx, feats, ctx.cv_df.iloc[tr_i], ctx.cv_df.iloc[va_i], {},
                                  *logits)
        fold_scores.append(_scores(ctx, y_cv[va_i], pred))
        if j < ctx.n_splits:  # первый повтор покрывает всю выборку
            oof[va_i] = pred
    others = {p: d for p, d in ctx.parts.items() if p not in ("train", "val")}
    logits = ((ctx.logit["train"], ctx.logit["val"], {p: ctx.logit[p] for p in others})
              if on_top else ())
    p_val, preds, importance = _fit_predict(ctx, feats, ctx.parts["train"], ctx.parts["val"],
                                            others, *logits)
    preds["val"] = p_val
    return {"fold_scores": fold_scores, "oof": oof, "preds": preds, "importance": importance}


def _base_run(ctx: _Context) -> dict[str, Any]:
    if ctx.how == "retrain":
        return _run(ctx, ctx.base.features)
    score = expit(ctx.cv_logit)
    y_cv = ctx.cv_df[ctx.target].to_numpy()
    return {"fold_scores": [_scores(ctx, y_cv[va], score[va]) for _, va in ctx.folds],
            "oof": score, "importance": None,
            "preds": {p: expit(v) for p, v in ctx.logit.items() if p != "train"}}


def _base_scores(ctx: _Context, scorer) -> dict[str, np.ndarray]:
    """Скор базы по частям для режима on_top."""
    out = {}
    for part, df in ctx.parts.items():
        if isinstance(scorer, str):
            p = pd.to_numeric(df[scorer], errors="coerce").to_numpy(dtype="float64")
        else:
            p = np.asarray(scorer.predict(df), dtype="float64")
        if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
            raise ValueError(
                f"Скор базы в части {part!r} должен быть вероятностью в [0, 1] без пропусков.")
        out[part] = p
    return out


def _setup(split: DataSplit, base, columns: list[str], *, how, segment, date_col, metrics,
           cat_features, params, n_splits, n_repeats, max_rows, n_boot, alpha,
           random_state) -> _Context:
    parts = dict(split.items())
    train = parts["train"]
    resolved = _resolve_base(base, params, train.columns)
    if how is None:
        how = "retrain" if resolved.features else "on_top"
    if how not in ("retrain", "on_top"):
        raise ValueError("how: 'retrain' или 'on_top'")
    if how == "retrain" and not resolved.features:
        raise ValueError("Для how='retrain' нужны признаки базы: по колонке скора модель не "
                         "переобучить. Передайте список признаков или саму модель.")
    if n_splits < 2:
        raise ValueError("n_splits должен быть не меньше 2.")
    used = resolved.features + columns
    missing = [c for c in dict.fromkeys(used) if c not in train.columns]
    if missing:
        raise KeyError(f"В таблице нет колонок: {missing}")
    overlap = sorted(set(columns) & set(resolved.features))
    if overlap:
        raise ValueError(f"Кандидаты уже есть в базе: {overlap}")
    if segment is not None and segment not in train.columns:
        raise KeyError(f"В таблице нет колонки сегмента {segment!r}")

    if cat_features is None:
        cats = list(resolved.cat_features or [])
        cats += detect_categorical(train[[c for c in dict.fromkeys(used) if c not in cats]])
    else:
        cats = list(cat_features)

    y = train[split.target].to_numpy()
    rows = np.arange(len(train))
    if max_rows and len(train) > max_rows:
        rows, _ = train_test_split(rows, train_size=max_rows, stratify=y,
                                   random_state=random_state)
        rows = np.sort(rows)
    folds = [f for r in range(n_repeats)
             for f in make_folds(y[rows], n_splits, random_state=random_state + r)]

    ctx = _Context(
        parts=parts, target=split.target, last="test" if "test" in parts else "val", how=how,
        base=resolved, cats=cats, metrics=[get_metric(m) for m in metrics], segment=segment,
        date_col=date_col, cv_df=train.iloc[rows], folds=folds, n_splits=n_splits,
        top_params=merge_params(LGBM_PARAMS, params), alpha=alpha, n_boot=n_boot,
        random_state=random_state)

    if how == "on_top":
        scorer = resolved.scorer
        if scorer is None:  # список признаков или нативная модель: скор считает модель на train
            scorer = resolved.template.clone().fit(
                train[resolved.features], y, eval_set=split.xy("val", resolved.features),
                cat_features=[c for c in cats if c in resolved.features])
        scores = _base_scores(ctx, scorer)
        ctx.logit = {p: prob_to_logit(s) for p, s in scores.items()}
        ctx.cv_logit = ctx.logit["train"][rows]
    ctx.base_run = _base_run(ctx)

    # корреляции кандидатов считаются с числовыми признаками базы (или с её скором)
    ctx.corr_rows = rows if len(rows) <= CORR_SAMPLE else np.random.default_rng(
        random_state).choice(rows, CORR_SAMPLE, replace=False)
    sample = train.iloc[ctx.corr_rows]
    num_base = [f for f in resolved.features if f not in cats]
    if num_base:
        ctx.corr_base = sample[num_base].apply(pd.to_numeric, errors="coerce").reset_index(
            drop=True)
    elif how == "on_top":
        ctx.corr_base = pd.DataFrame({"base_score": expit(ctx.logit["train"][ctx.corr_rows])})
    return ctx


# --- оценка одного набора кандидатов ------------------------------------------------------

def _cv_interval(deltas: np.ndarray, n_train: int, n_valid: int, alpha: float):
    """Интервал средней дельты по фолдам с поправкой на пересечение обучающих частей
    (corrected resampled t-test, Nadeau & Bengio)."""
    d = deltas[~np.isnan(deltas)]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(d.mean())
    se = np.sqrt((1 / len(d) + n_valid / n_train) * d.var(ddof=1))
    half = student_t.ppf(1 - alpha / 2, len(d) - 1) * se
    return mean, mean - half, mean + half


def _bootstrap_interval(ctx: _Context, y, base, new):
    """Парный бутстрап дельты основной метрики на отложенной части."""
    if not ctx.n_boot:
        return float("nan"), float("nan")
    rng = np.random.default_rng(ctx.random_state)
    metric, n = ctx.primary, len(y)
    deltas = []
    for _ in range(ctx.n_boot):
        i = rng.integers(0, n, n)
        deltas.append(metric.delta(metric(y[i], new[i]), metric(y[i], base[i])))
    lo, hi = np.nanquantile(deltas, [ctx.alpha / 2, 1 - ctx.alpha / 2])
    return float(lo), float(hi)


def _covered(df: pd.DataFrame, extra: list[str]) -> np.ndarray:
    return df[extra].notna().any(axis=1).to_numpy()


def _cheap_stats(ctx: _Context, extra: list[str]) -> dict[str, Any]:
    """Покрытие, период заполнения, однофакторный AUC, корреляция с базой, PSI — без моделей."""
    total = sum(len(d) for d in ctx.parts.values())
    covered = {p: _covered(d, extra) for p, d in ctx.parts.items()}
    row: dict[str, Any] = {"coverage": sum(c.sum() for c in covered.values()) / total}
    if ctx.date_col is not None:
        dates = pd.concat([pd.to_datetime(d.loc[covered[p], ctx.date_col])
                           for p, d in ctx.parts.items()])
        row["filled_from"], row["filled_to"] = dates.min(), dates.max()

    sample = ctx.parts["train"].iloc[ctx.corr_rows]
    numeric = [f for f in extra if f not in ctx.cats]
    row["uni_auc"] = (direct_auc(pd.to_numeric(sample[extra[0]], errors="coerce"),
                                 sample[ctx.target].to_numpy())
                      if len(extra) == 1 and numeric else np.nan)
    row["max_corr_base"], row["corr_with"] = np.nan, None
    if numeric and ctx.corr_base is not None:
        with np.errstate(invalid="ignore", divide="ignore"):  # константы дают corr = NaN
            corr = pd.concat({
                f: ctx.corr_base.corrwith(pd.Series(pd.to_numeric(sample[f], errors="coerce")
                                                    .to_numpy(dtype="float64")))
                for f in numeric}).abs()
        if corr.notna().any():
            row["max_corr_base"], row["corr_with"] = float(corr.max()), corr.idxmax()[1]
    if ctx.date_col is not None and ctx.last != "val":
        train, last = ctx.parts["train"], ctx.parts[ctx.last]
        row["psi_max"] = max(
            psi_by_period(last, f, ctx.date_col, reference=train[f],
                          categorical=True if f in ctx.cats else None)["psi"].max()
            for f in extra)
    return row


def _evaluate(ctx: _Context, name: str, kind: str, feats: list[str], extra: list[str]):
    """Сравнивает модель на ``feats`` с базой. ``extra`` — сами кандидаты (для покрытия)."""
    run, base = _run(ctx, feats), ctx.base_run
    primary, last = ctx.primary, ctx.last
    n_valid = len(ctx.folds[0][1])
    n_train = len(ctx.cv_df) - n_valid

    metric_rows, cv = [], {}
    for m in ctx.metrics:
        b = np.array([s[m.name] for s in base["fold_scores"]])
        n = np.array([s[m.name] for s in run["fold_scores"]])
        deltas = np.array([m.delta(x, y) for x, y in zip(n, b, strict=True)])
        mean, lo, hi = _cv_interval(deltas, n_train, n_valid, ctx.alpha)
        cv[m.name] = (deltas, mean, lo, hi, float(np.nanmean(b)))
        metric_rows.append({"name": name, "part": "cv", "metric": m.name,
                            "base": float(np.nanmean(b)), "new": float(np.nanmean(n)),
                            "delta": mean, "ci_low": lo, "ci_high": hi})
    deltas, mean, lo, hi, base_cv = cv[primary.name]
    row: dict[str, Any] = {
        "name": name, "kind": kind, "n_features": len(extra), **_cheap_stats(ctx, extra),
        f"base_{primary.name}_cv": base_cv, "delta_cv": mean, "cv_ci_low": lo, "cv_ci_high": hi,
        "folds_better": float(np.nanmean(deltas > 0)),
    }

    for part in dict.fromkeys(["val", last]):
        y, p_base, p_new = ctx.y(part), base["preds"][part], run["preds"][part]
        sb, sn = _scores(ctx, y, p_base), _scores(ctx, y, p_new)
        ci = _bootstrap_interval(ctx, y, p_base, p_new) if part == last else (np.nan, np.nan)
        for m in ctx.metrics:
            is_primary = m.name == primary.name
            metric_rows.append({
                "name": name, "part": part, "metric": m.name, "base": sb[m.name],
                "new": sn[m.name], "delta": m.delta(sn[m.name], sb[m.name]),
                "ci_low": ci[0] if is_primary else np.nan,
                "ci_high": ci[1] if is_primary else np.nan})
        row[f"delta_{part}"] = primary.delta(sn[primary.name], sb[primary.name])
        if part == last:
            row[f"{last}_ci_low"], row[f"{last}_ci_high"] = ci
            mask = _covered(ctx.parts[part], extra)
            row[f"delta_{last}_covered"] = (
                primary.delta(primary(y[mask], p_new[mask]), primary(y[mask], p_base[mask]))
                if 1 < mask.sum() < len(mask) else row[f"delta_{part}"])

    segment_rows = []
    if ctx.segment is not None:
        sources = {"cv": (ctx.cv_df, base["oof"], run["oof"]),
                   last: (ctx.parts[last], base["preds"][last], run["preds"][last])}
        for part, (df, p_base, p_new) in sources.items():
            seg = df[ctx.segment].astype(object).where(df[ctx.segment].notna(), "NaN").to_numpy()
            y = df[ctx.target].to_numpy()
            for value in sorted(pd.unique(seg), key=str):
                mask = seg == value
                if mask.sum() < MIN_SEGMENT:
                    continue
                for m in ctx.metrics:
                    b, n = m(y[mask], p_base[mask]), m(y[mask], p_new[mask])
                    segment_rows.append({"name": name, "part": part, "segment": value,
                                         "n": int(mask.sum()), "metric": m.name, "base": b,
                                         "new": n, "delta": m.delta(n, b)})
        on_last = [r for r in segment_rows
                   if r["part"] == last and r["metric"] == primary.name
                   and not np.isnan(r["delta"])]
        if on_last:
            worst = min(on_last, key=lambda r: r["delta"])
            row["worst_segment"], row["worst_segment_delta"] = worst["segment"], worst["delta"]

    row["verdict"] = "better" if lo > 0 else "worse" if hi < 0 else "same"
    return row, metric_rows, segment_rows, run["importance"]


def _screen(ctx: _Context, candidates: list[str]) -> pd.Series:
    """Быстрый отсев: маленькая модель на одном кандидате поверх скора базы, дельта на val."""
    y_cv = ctx.cv_df[ctx.target].to_numpy()
    val = ctx.parts["val"]
    y_val, primary = ctx.y("val"), ctx.primary
    logit_tr = prob_to_logit(ctx.base_run["oof"])  # для retrain — out-of-fold прогноз базы
    p_base = ctx.base_run["preds"]["val"]
    logit_va = prob_to_logit(p_base)
    base_value = primary(y_val, p_base)
    out = {}
    for f in tqdm(candidates, desc="быстрый отсев", leave=False):
        cats = [f] if f in ctx.cats else []
        _, predict = _fit_top(ctx.cv_df[[f]], y_cv, val[[f]], y_val, logit_tr, logit_va, cats,
                              SCREEN_PARAMS, SCREEN_STOPPING)
        out[f] = primary.delta(primary(y_val, predict(val[[f]], logit_va)), base_value)
    return pd.Series(out, name="screen_delta").sort_values(ascending=False)


# --- результат ----------------------------------------------------------------------------

def _plot_replacement(result: Result) -> go.Figure:
    t = result.table.sort_values("delta_replace")
    labels = t["candidate"] + " вместо " + t["replaces"]
    fig = go.Figure()
    fig.add_bar(y=labels, x=t["delta_add"], orientation="h", name="добавить к базе",
                marker_color=color(0))
    fig.add_bar(y=labels, x=t["delta_replace"], orientation="h", name="заменить",
                marker_color=color(1),
                error_x={"type": "data", "symmetric": False,
                         "array": t["replace_ci_high"] - t["delta_replace"],
                         "arrayminus": t["delta_replace"] - t["replace_ci_low"]})
    fig.add_vline(x=0, line_color="black", line_width=1)
    style(fig, f"Добавить или заменить: прирост {result.info['metric']} по CV на train",
          height=max(350, 60 * len(t) + 160),
          xaxis_title=f"Прирост {result.info['metric']} (> 0 — лучше базы)")
    return fig.update_layout(barmode="group", hovermode="closest", showlegend=True,
                             margin={"l": 260})


@dataclass
class FeatureTest:
    """Результат :func:`test_features` / :func:`test_domains`.

    Все дельты — «насколько стало лучше»: ``> 0`` — кандидаты улучшили метрику (для logloss и
    brier знак уже перевёрнут).

    Attributes
    ----------
    summary : pd.DataFrame
        Строка на кандидата (и ``ALL`` — все вместе) или на домен: покрытие, однофакторный AUC,
        корреляция с базой, прирост основной метрики по CV на train с интервалом, прирост на
        val и test, худший сегмент, ``verdict`` (``better`` / ``same`` / ``worse`` — по
        интервалу CV; ``screened out`` — не прошёл быстрый отсев).
    metrics : pd.DataFrame
        Все метрики: ``name, part (cv / val / test), metric, base, new, delta, ci_low, ci_high``.
    segments : pd.DataFrame | None
        То же по сегментам: ``name, part, segment, n, metric, base, new, delta`` (сегменты
        меньше 30 строк пропускаются).
    features : pd.DataFrame | None
        Для доменов — вклад признаков внутри домена.
    info : dict
        Параметры запуска.
    """

    name: str
    summary: pd.DataFrame
    metrics: pd.DataFrame
    segments: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    info: dict[str, Any] = field(default_factory=dict)
    _ctx: _Context | None = field(default=None, repr=False)

    def _auto(self, fig: go.Figure, suffix: str) -> go.Figure:
        from collection_lab.tracking.experiment import auto_figure

        auto_figure(fig, f"{self.name}_{suffix}")
        return fig

    def plot(self) -> go.Figure:
        """Прирост основной метрики по каждому кандидату с доверительными интервалами:
        CV на train и отложенная часть. Цвет точки CV — вердикт."""
        last, metric = self.info["last"], self.info["metric"]
        s = self.summary.dropna(subset=["delta_cv"]).sort_values("delta_cv")
        fig = go.Figure()
        fig.add_scatter(
            y=s["name"], x=s["delta_cv"], mode="markers", name="CV на train",
            marker={"size": 11, "color": s["verdict"].map(VERDICT_COLORS)},
            error_x={"type": "data", "symmetric": False, "color": "#444",
                     "array": s["cv_ci_high"] - s["delta_cv"],
                     "arrayminus": s["delta_cv"] - s["cv_ci_low"]})
        fig.add_scatter(
            y=s["name"], x=s[f"delta_{last}"], mode="markers", name=last,
            marker={"size": 9, "symbol": "diamond-open", "color": "#444"},
            error_x={"type": "data", "symmetric": False, "color": "#aaa",
                     "array": s[f"{last}_ci_high"] - s[f"delta_{last}"],
                     "arrayminus": s[f"delta_{last}"] - s[f"{last}_ci_low"]})
        fig.add_vline(x=0, line_color="black", line_width=1)
        style(fig, f"Прирост {metric} относительно базы ({self.info['base']}, {self.info['how']})",
              height=max(350, 45 * len(s) + 160),
              xaxis_title=f"Прирост {metric} (> 0 — лучше базы), интервал "
                          f"{1 - self.info['alpha']:.0%}")
        fig.update_layout(hovermode="closest", scattermode="group", showlegend=True,
                          margin={"l": 220})
        return self._auto(fig, "delta")

    def plot_segments(self, metric: str | None = None, part: str | None = None) -> go.Figure:
        """Прирост метрики по сегментам: строки — кандидаты, колонки — сегменты.

        ``part`` — ``"cv"`` или отложенная часть (по умолчанию она), ``metric`` — по умолчанию
        основная метрика."""
        if self.segments is None:
            raise ValueError("Сегменты не заданы: передайте segment='колонка'.")
        metric, part = metric or self.info["metric"], part or self.info["last"]
        t = self.segments[(self.segments["metric"] == metric) & (self.segments["part"] == part)]
        if t.empty:
            raise ValueError(f"Нет данных для metric={metric!r}, part={part!r}.")
        order = self.summary.dropna(subset=["delta_cv"]).sort_values("delta_cv")["name"]
        z = t.pivot_table(index="name", columns="segment", values="delta").reindex(order)
        sizes = t.drop_duplicates("segment").set_index("segment")["n"]
        fig = go.Figure(go.Heatmap(
            z=z.to_numpy(), x=[f"{c}<br>n = {sizes[c]}" for c in z.columns], y=z.index,
            colorscale="RdYlGn", zmid=0, text=z.round(4).to_numpy(), texttemplate="%{text}",
            hovertemplate="%{y}, %{x}: %{z:.4f}<extra></extra>"))
        style(fig, f"Прирост {metric} по сегментам ({part})",
              height=max(350, 40 * len(z) + 180))
        fig.update_layout(hovermode="closest", margin={"l": 220})
        return self._auto(fig, "segments")

    def plot_metrics(self, part: str | None = None) -> go.Figure:
        """Прирост всех метрик в процентах от значения базы: строки — кандидаты."""
        part = part or self.info["last"]
        t = self.metrics[self.metrics["part"] == part].copy()
        t["rel"] = t["delta"] / t["base"].abs() * 100
        order = self.summary.dropna(subset=["delta_cv"]).sort_values("delta_cv")["name"]
        z = t.pivot_table(index="name", columns="metric", values="rel").reindex(order)
        z = z[[m for m in self.info["metrics"] if m in z.columns]]
        fig = go.Figure(go.Heatmap(
            z=z.to_numpy(), x=list(z.columns), y=z.index, colorscale="RdYlGn", zmid=0,
            text=z.round(2).to_numpy(), texttemplate="%{text}%",
            hovertemplate="%{y}, %{x}: %{z:.2f}%<extra></extra>"))
        style(fig, f"Прирост метрик, % от базы ({part}; > 0 — лучше)",
              height=max(350, 40 * len(z) + 180))
        fig.update_layout(hovermode="closest", margin={"l": 220})
        return self._auto(fig, "metrics")

    def plot_domain(self, name: str) -> go.Figure:
        """Вклад признаков внутри домена (важность gain в модели с доменом)."""
        if self.features is None:
            raise ValueError("Вклад признаков есть только у результата test_domains.")
        t = self.features[self.features["domain"] == name].sort_values("share")
        if t.empty:
            raise ValueError(f"Нет домена {name!r}. Есть: {list(self.features['domain'].unique())}")
        fig = go.Figure(go.Bar(x=t["share"], y=t["feature"], orientation="h",
                               marker_color=color(0)))
        style(fig, f"Домен «{name}»: доля признака в важности домена",
              height=max(350, 24 * len(t) + 160), xaxis_title="Доля важности (gain)")
        fig.update_layout(hovermode="closest", margin={"l": 240})
        return self._auto(fig, f"domain_{name}")

    def replacement(self, threshold: float = 0.7) -> Result:
        """Проверка «заменить вместо добавить» для кандидатов, похожих на признак базы.

        Кандидат с ``|corr| >= threshold`` с признаком базы почти не даёт прироста сверху —
        информация уже в модели. Здесь для каждого такого кандидата обучается модель «база без
        похожего признака + кандидат» и сравнивается с базой.

        Returns
        -------
        Result
            ``table``: ``candidate, replaces, corr, delta_add, delta_replace, replace_ci_low,
            replace_ci_high, delta_replace_<test>, verdict`` (``better`` — замена значимо лучше
            базы); ``plot()`` — «добавить» против «заменить».
        """
        ctx = self._ctx
        if ctx is None or ctx.how != "retrain":
            raise ValueError("Замена считается только при how='retrain': нужна модель, "
                             "которую можно переобучить без одного из признаков.")
        s = self.summary
        pairs = s[(s["kind"] == "feature") & s["delta_cv"].notna()
                  & (s["max_corr_base"] >= threshold)]
        rows = []
        for r in tqdm(list(pairs.itertuples()), desc="замена", leave=False):
            feats = [f for f in ctx.base.features if f != r.corr_with] + [r.name]
            row, *_ = _evaluate(ctx, r.name, "feature", feats, [r.name])
            rows.append({
                "candidate": r.name, "replaces": r.corr_with, "corr": r.max_corr_base,
                "delta_add": r.delta_cv, "delta_replace": row["delta_cv"],
                "replace_ci_low": row["cv_ci_low"], "replace_ci_high": row["cv_ci_high"],
                f"delta_replace_{ctx.last}": row[f"delta_{ctx.last}"], "verdict": row["verdict"]})
        columns = ["candidate", "replaces", "corr", "delta_add", "delta_replace",
                   "replace_ci_low", "replace_ci_high", f"delta_replace_{ctx.last}", "verdict"]
        info = {"metric": ctx.primary.name, "threshold": threshold}
        return Result("replacement", pd.DataFrame(rows, columns=columns), None, info,
                      plotter=_plot_replacement)

    def save(self, directory: str | Path, *, to_clearml: bool | None = None) -> Path:
        """Таблицы в csv и ``info`` в json.

        ``to_clearml``: ``None`` — дублировать в ClearML при активном ``Experiment(clearml=True)``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for key in ("summary", "metrics", "segments", "features"):
            table = getattr(self, key)
            if table is not None:
                write_csv(table, directory / f"{self.name}_{key}.csv")
        with open(directory / f"{self.name}.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.info), f, ensure_ascii=False, indent=2)
        _maybe_send(self, self.name, to_clearml)
        return directory

    def __repr__(self) -> str:
        counts = self.summary["verdict"].value_counts().to_dict()
        return f"FeatureTest({self.name!r}, rows={len(self.summary)}, verdicts={counts})"

    def _repr_html_(self) -> str:
        head = (f"<b>{self.name}</b> — база: {self.info['base']}, способ: {self.info['how']}, "
                f"метрика: {self.info['metric']}")
        return head + self.summary._repr_html_()


def _collect(name: str, ctx: _Context, rows, metric_rows, segment_rows, features=None,
             **info) -> FeatureTest:
    summary = pd.DataFrame(rows)
    order = summary["delta_cv"].rank(ascending=False, na_option="bottom")
    summary = summary.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)
    info = {"base": ctx.base.label, "how": ctx.how, "metric": ctx.primary.name,
            "metrics": [m.name for m in ctx.metrics], "last": ctx.last, "alpha": ctx.alpha,
            "segment": ctx.segment, "cv_rows": len(ctx.cv_df), "n_folds": len(ctx.folds),
            "rows": {p: len(d) for p, d in ctx.parts.items()}, **info}
    return FeatureTest(
        name=name, summary=summary, metrics=pd.DataFrame(metric_rows),
        segments=pd.DataFrame(segment_rows) if segment_rows else None, features=features,
        info=info, _ctx=ctx)


def test_features(
    split: DataSplit,
    base,
    candidates: Sequence[str],
    *,
    how: str | None = None,
    segment: str | None = None,
    date_col: str | None = None,
    metrics: Sequence[str | Metric] = DEFAULT_METRICS,
    cat_features: Iterable[str] | None = None,
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    max_exact: int = 20,
    max_rows: int | None = 200_000,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> FeatureTest:
    """Что даст добавление новых признаков: каждого по отдельности и всех вместе.

    Parameters
    ----------
    split : DataSplit
        Одна таблица со всеми колонками (признаки базы, кандидаты, сегмент, дата), разбитая на
        train / val / test.
    base : list[str] | модель | str
        С чем сравнивать: список признаков, обученная модель (адаптер библиотеки, путь к ``.pkl``
        или папке экспорта, нативный LightGBM / CatBoost) или имя колонки со скором готовой
        модели (вероятность).
    candidates : list[str]
        Новые признаки.
    how : {"retrain", "on_top"}, optional
        ``"retrain"`` — переобучить базу с кандидатами; ``"on_top"`` — дообучить поверх скора
        базы только на кандидатах. По умолчанию: ``"retrain"``, а если ``base`` — колонка
        скора, то ``"on_top"``.
    segment : str, optional
        Колонка сегмента — прирост считается ещё и в каждом сегменте.
    date_col : str, optional
        Колонка даты — добавляет период заполнения кандидата и максимальный PSI по месяцам
        test относительно train.
    metrics : list
        Метрики; первая — основная: по ней ранжирование, интервалы и вердикт.
    cat_features : list[str], optional
        Категориальные признаки; по умолчанию определяются автоматически.
    params : dict, optional
        Параметры модели поверх параметров базы.
    n_splits, n_repeats : int
        Кросс-валидация на train: фолдов и повторов. Больше повторов — уже интервал.
    max_exact : int
        Если кандидатов больше, сначала идёт быстрый отсев (маленькая модель поверх скора
        базы, дельта на val), точно оцениваются ``max_exact`` лучших и строка ``ALL``.
    max_rows : int, optional
        Кросс-валидация идёт на подвыборке train такого размера; итог на val / test — всегда
        на всех данных.
    n_boot : int
        Число повторов бутстрапа для интервала на отложенной части (0 — без интервала).
    alpha : float
        Уровень значимости интервалов (0.05 — интервалы 95%).

    Returns
    -------
    FeatureTest
        ``summary`` — строка на кандидата и строка ``ALL``; ``plot()``, ``plot_segments()``,
        ``plot_metrics()``, ``replacement()``.
    """
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise ValueError("Список кандидатов пуст.")
    ctx = _setup(split, base, candidates, how=how, segment=segment, date_col=date_col,
                 metrics=metrics, cat_features=cat_features, params=params, n_splits=n_splits,
                 n_repeats=n_repeats, max_rows=max_rows, n_boot=n_boot, alpha=alpha,
                 random_state=random_state)
    base_feats = ctx.base.features if ctx.how == "retrain" else []

    exact, screen = candidates, None
    if len(candidates) > max_exact:
        screen = _screen(ctx, candidates)
        exact = list(screen.index[:max_exact])

    tasks = [(f, "feature", [f]) for f in exact]
    if len(candidates) > 1:
        tasks.append((ALL, "all", candidates))
    rows, metric_rows, segment_rows = [], [], []
    for name, kind, extra in tqdm(tasks, desc="оценка кандидатов", disable=not verbose):
        row, m_rows, s_rows, _ = _evaluate(ctx, name, kind, base_feats + extra, extra)
        rows.append(row)
        metric_rows += m_rows
        segment_rows += s_rows
    for f in candidates:
        if f not in exact:
            rows.append({"name": f, "kind": "feature", "n_features": 1,
                         **_cheap_stats(ctx, [f]), "verdict": "screened out"})
    if screen is not None:
        for row in rows:
            row["screen_delta"] = screen.get(row["name"], np.nan)
    return _collect("feature_test", ctx, rows, metric_rows, segment_rows,
                    candidates=candidates, n_exact=len(exact))


def test_domains(
    split: DataSplit,
    base,
    domains: dict[str, Sequence[str]],
    *,
    how: str | None = None,
    segment: str | None = None,
    date_col: str | None = None,
    metrics: Sequence[str | Metric] = DEFAULT_METRICS,
    cat_features: Iterable[str] | None = None,
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    max_rows: int | None = 200_000,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> FeatureTest:
    """Что даст подключение домена данных — группы признаков из одного источника.

    Каждый домен оценивается целиком и независимо от остальных: база против «база + все
    признаки домена». Параметры — как у :func:`test_features`.

    Parameters
    ----------
    domains : dict[str, list[str]]
        ``{"транзакции": [...], "бюро": [...]}``.

    Returns
    -------
    FeatureTest
        ``summary`` — строка на домен. Кроме колонок :func:`test_features`:

        - ``coverage`` — доля строк, где заполнен хотя бы один признак домена;
        - ``delta_<test>_covered`` — прирост только на этих строках;
        - ``alone_<метрика>_<test>`` — качество модели на одном домене, без базы;
        - ``gain_share`` — доля домена в важности модели «база + домен» (при ``"retrain"``).

        ``features`` — вклад признаков внутри домена: ``domain, feature, importance, share,
        coverage, uni_auc``; график — ``plot_domain(имя)``.
    """
    domains = {name: list(dict.fromkeys(cols)) for name, cols in domains.items()}
    if not domains or not all(domains.values()):
        raise ValueError("domains: нужен словарь {имя домена: непустой список признаков}.")
    columns = list(dict.fromkeys(c for cols in domains.values() for c in cols))
    ctx = _setup(split, base, columns, how=how, segment=segment, date_col=date_col,
                 metrics=metrics, cat_features=cat_features, params=params, n_splits=n_splits,
                 n_repeats=n_repeats, max_rows=max_rows, n_boot=n_boot, alpha=alpha,
                 random_state=random_state)
    base_feats = ctx.base.features if ctx.how == "retrain" else []
    primary, last = ctx.primary, ctx.last
    alone_model = ctx.base.template or make_model("lgbm", params)

    rows, metric_rows, segment_rows, feature_rows = [], [], [], []
    for name, cols in tqdm(domains.items(), desc="оценка доменов", disable=not verbose):
        row, m_rows, s_rows, importance = _evaluate(ctx, name, "domain", base_feats + cols, cols)
        alone = alone_model.clone().fit(
            ctx.parts["train"][cols], ctx.y("train"), eval_set=split.xy("val", cols),
            cat_features=[c for c in ctx.cats if c in cols])
        row[f"alone_{primary.name}_{last}"] = primary(ctx.y(last), alone.predict(ctx.parts[last]))
        row[f"base_{primary.name}_{last}"] = primary(ctx.y(last), ctx.base_run["preds"][last])
        inside = importance.reindex(cols).fillna(0.0)
        row["gain_share"] = (float(inside.sum() / importance.sum())
                             if ctx.how == "retrain" and importance.sum() else np.nan)
        for f in cols:
            stats = _cheap_stats(ctx, [f])
            feature_rows.append({
                "domain": name, "feature": f, "importance": float(inside[f]),
                "share": float(inside[f] / inside.sum()) if inside.sum() else 0.0,
                "coverage": stats["coverage"], "uni_auc": stats["uni_auc"]})
        rows.append(row)
        metric_rows += m_rows
        segment_rows += s_rows
    features = pd.DataFrame(feature_rows).sort_values(
        ["domain", "importance"], ascending=[True, False], ignore_index=True)
    return _collect("domain_test", ctx, rows, metric_rows, segment_rows, features,
                    domains={k: len(v) for k, v in domains.items()})


# имена начинаются с test_ — pytest не должен принимать функции за тесты при импорте
test_features.__test__ = False
test_domains.__test__ = False
