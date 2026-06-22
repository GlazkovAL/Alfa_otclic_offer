import argparse
import json
import math
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")

SEED = 20260619
TARGET = "target_value"
ID_COL = "front_id"
DATE_COL = "decision_day"


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def normalized_rank(pred: np.ndarray) -> np.ndarray:
    return pd.Series(pred).rank(method="average").to_numpy() / len(pred)


def add_features(df: pd.DataFrame, date_mode: str = "general", use_front_id: bool = False) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL])

    if date_mode in {"general", "full"}:
        df["decision_month"] = dt.dt.month.astype("int16")
        df["decision_day_of_month"] = dt.dt.day.astype("int16")
        df["decision_dayofweek"] = dt.dt.dayofweek.astype("int16")
        df["decision_quarter"] = dt.dt.quarter.astype("int16")
        df["is_month_start"] = dt.dt.is_month_start.astype("int8")
        df["is_month_end"] = dt.dt.is_month_end.astype("int8")

    if date_mode == "full":
        df["decision_year"] = dt.dt.year.astype("int16")
        df["decision_weekofyear"] = dt.dt.isocalendar().week.astype("int16")
        df["decision_year_month_cat"] = dt.dt.strftime("%Y-%m").astype("object")
        df["decision_days_from_start"] = (dt - pd.Timestamp("2024-02-01")).dt.days.astype("int16")

    numeric_cols = [c for c in df.columns if c not in {TARGET, DATE_COL, ID_COL}]
    for c in numeric_cols:
        if df[c].isna().any():
            df[f"{c}__isna"] = df[c].isna().astype("int8")

    # Признаки предложения: ставка относительно ключевой ставки.
    df["rate_spread"] = df["offered_rate"] - df["cb_rate"]
    df["rate_ratio"] = safe_div(df["offered_rate"], df["cb_rate"])
    df["rate_abs_diff"] = df["rate_spread"].abs()

    # Признаки лимита и соотношения запрошенной суммы к доступным лимитам.
    df["limit_range"] = df["overdraft_limit_max"] - df["overdraft_limit_min"]
    df["limit_mid"] = (df["overdraft_limit_max"] + df["overdraft_limit_min"]) / 2
    df["loan_to_min_limit"] = safe_div(df["loan_amount_last"], df["overdraft_limit_min"])
    df["loan_to_max_limit"] = safe_div(df["loan_amount_last"], df["overdraft_limit_max"])
    df["loan_to_limit_mid"] = safe_div(df["loan_amount_last"], df["limit_mid"])
    df["loan_minus_min_limit"] = df["loan_amount_last"] - df["overdraft_limit_min"]
    df["loan_minus_max_limit"] = df["loan_amount_last"] - df["overdraft_limit_max"]

    # Признаки динамики активности за 30/90 дней.
    df["sum_deb_ul_30_to_90"] = safe_div(df["sum_deb_ul_30"], df["sum_deb_ul_90"])
    df["cnt_deb_ul_ip_30_to_90"] = safe_div(df["cnt_deb_ul_ip_30"], df["cnt_deb_ul_ip_90"])
    df["sum_deb_ul_90_minus_30"] = df["sum_deb_ul_90"] - df["sum_deb_ul_30"]
    df["cnt_deb_ul_ip_90_minus_30"] = df["cnt_deb_ul_ip_90"] - df["cnt_deb_ul_ip_30"]
    df["avg_deb_ul_ip_90"] = safe_div(df["sum_deb_ul_90"], df["cnt_deb_ul_ip_90"])
    df["avg_deb_ul_ip_30"] = safe_div(df["sum_deb_ul_30"], df["cnt_deb_ul_ip_30"])

    df["loan_credit_to_debit_90"] = safe_div(df["cnt_cred_loan_90"], df["cnt_deb_loan_90"])
    df["loan_credit_debit_diff_90"] = df["cnt_cred_loan_90"] - df["cnt_deb_loan_90"]
    df["total_credit_activity"] = (
        df["fl_hdb_bki_total_active_products"].fillna(0)
        + df["cnt_cred_loan_90"].fillna(0)
        + df["cnt_deb_loan_90"].fillna(0)
    )

    df["db_group_last__fl_adminarea"] = (
        df["db_group_last"].astype("string").fillna("__NA__")
        + "__"
        + df["fl_adminarea"].astype("string").fillna("__NA__")
    )

    if use_front_id:
        fid = df[ID_COL].astype("int64")
        df["front_id_num"] = fid
        df["front_id_mod_10"] = (fid % 10).astype("int16")
        df["front_id_mod_100"] = (fid % 100).astype("int16")
        df["front_id_mod_1000"] = (fid % 1000).astype("int16")

    df = df.drop(columns=[DATE_COL])
    if not use_front_id:
        df = df.drop(columns=[ID_COL])

    return df


def prepare_catboost_data(train: pd.DataFrame, test: pd.DataFrame, date_mode: str, use_front_id: bool):
    all_df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    all_feat = add_features(all_df, date_mode=date_mode, use_front_id=use_front_id)

    cat_cols = [c for c in all_feat.columns if not is_numeric_dtype(all_feat[c])]
    for c in cat_cols:
        all_feat[c] = all_feat[c].astype("string").fillna("__NA__").astype(str)

    X = all_feat.iloc[: len(train)].reset_index(drop=True)
    X_test = all_feat.iloc[len(train) :].reset_index(drop=True)
    y = train[TARGET].astype(int).reset_index(drop=True)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return X, y, X_test, cat_idx


def prepare_tree_data(train: pd.DataFrame, test: pd.DataFrame, date_mode: str, use_front_id: bool):
    all_raw = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    all_feat = add_features(all_raw, date_mode=date_mode, use_front_id=use_front_id)

    cat_cols = [c for c in all_feat.columns if not is_numeric_dtype(all_feat[c])]
    for c in cat_cols:
        s = all_feat[c].astype("string").fillna("__NA__")
        all_feat[f"{c}__freq"] = s.map(s.value_counts(normalize=True)).astype("float32")
        codes, _ = pd.factorize(s, sort=True)
        all_feat[c] = codes.astype("int32")

    X = all_feat.iloc[: len(train)].reset_index(drop=True)
    X_test = all_feat.iloc[len(train) :].reset_index(drop=True)
    y = train[TARGET].astype(int).reset_index(drop=True)
    return X, y, X_test, cat_cols


def temporal_masks(train: pd.DataFrame, train_start: str, valid_start: str):
    dt = pd.to_datetime(train[DATE_COL])
    train_mask = dt >= pd.Timestamp(train_start)
    valid_mask = dt >= pd.Timestamp(valid_start)
    fit_mask = train_mask & ~valid_mask
    final_mask = dt >= pd.Timestamp(train_start)
    return fit_mask, valid_mask, final_mask


def train_catboost(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path):
    cfg = {
        "name": "catboost_all_general_d6",
        "train_start": "2024-02-01",
        "valid_start": "2025-04-01",
        "date_mode": "general",
        "use_front_id": False,
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 20,
        "bagging_temperature": 0.8,
        "random_strength": 2.2,
        "rsm": 0.9,
        "seed": SEED + 3,
    }

    X, y, X_test, cat_idx = prepare_catboost_data(train, test, cfg["date_mode"], cfg["use_front_id"])
    fit_mask, valid_mask, final_mask = temporal_masks(train, cfg["train_start"], cfg["valid_start"])

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=cfg["iterations"],
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        random_seed=cfg["seed"],
        auto_class_weights="Balanced",
        bootstrap_type="Bayesian",
        bagging_temperature=cfg["bagging_temperature"],
        random_strength=cfg["random_strength"],
        rsm=cfg["rsm"],
        border_count=128,
        allow_writing_files=False,
        verbose=250,
        early_stopping_rounds=350,
        thread_count=-1,
    )
    model.fit(
        Pool(X.loc[fit_mask], y.loc[fit_mask], cat_features=cat_idx),
        eval_set=Pool(X.loc[valid_mask], y.loc[valid_mask], cat_features=cat_idx),
        use_best_model=True,
    )
    best_iter = int(model.get_best_iteration() or cfg["iterations"])
    valid_pred = model.predict_proba(Pool(X.loc[valid_mask], cat_features=cat_idx))[:, 1]
    valid_auc = roc_auc_score(y.loc[valid_mask], valid_pred)

    final_iterations = max(300, min(cfg["iterations"], int(math.ceil(best_iter * 1.08))))
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=final_iterations,
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        random_seed=cfg["seed"] + 17,
        auto_class_weights="Balanced",
        bootstrap_type="Bayesian",
        bagging_temperature=cfg["bagging_temperature"],
        random_strength=cfg["random_strength"],
        rsm=cfg["rsm"],
        border_count=128,
        allow_writing_files=False,
        verbose=250,
        thread_count=-1,
    )
    final_model.fit(Pool(X.loc[final_mask], y.loc[final_mask], cat_features=cat_idx))
    test_pred = final_model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]
    save_submission(output_dir / "submission_catboost_single.csv", test, test_pred)

    return {
        "name": cfg["name"],
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "valid_auc": valid_auc,
        "best_iter": best_iter,
        "final_iter": final_iterations,
    }


def train_lightgbm(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path):
    cfg = {
        "name": "lightgbm_all_full_frontid",
        "train_start": "2024-02-01",
        "valid_start": "2025-04-01",
        "date_mode": "full",
        "use_front_id": True,
        "n_estimators": 4500,
        "learning_rate": 0.016,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 150,
        "subsample": 0.84,
        "colsample_bytree": 0.78,
        "reg_alpha": 2.0,
        "reg_lambda": 18.0,
        "extra_trees": True,
        "iter_mult": 1.10,
        "seed": SEED + 304,
    }

    X, y, X_test, cat_cols = prepare_tree_data(train, test, cfg["date_mode"], cfg["use_front_id"])
    fit_mask, valid_mask, final_mask = temporal_masks(train, cfg["train_start"], cfg["valid_start"])

    pos = y.loc[fit_mask].sum()
    neg = fit_mask.sum() - pos
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        subsample_freq=1,
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        scale_pos_weight=float(neg / pos),
        random_state=cfg["seed"],
        n_jobs=-1,
        verbosity=-1,
        extra_trees=cfg["extra_trees"],
    )
    model.fit(
        X.loc[fit_mask],
        y.loc[fit_mask],
        eval_set=[(X.loc[valid_mask], y.loc[valid_mask])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(250), lgb.log_evaluation(250)],
    )
    best_iter = int(model.best_iteration_ or cfg["n_estimators"])
    valid_pred = model.predict_proba(X.loc[valid_mask])[:, 1]
    valid_auc = roc_auc_score(y.loc[valid_mask], valid_pred)

    pos_final = y.loc[final_mask].sum()
    neg_final = final_mask.sum() - pos_final
    final_iter = max(300, min(cfg["n_estimators"], int(math.ceil(best_iter * cfg["iter_mult"]))))
    final_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=final_iter,
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        subsample_freq=1,
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        scale_pos_weight=float(neg_final / pos_final),
        random_state=cfg["seed"] + 1000,
        n_jobs=-1,
        verbosity=-1,
        extra_trees=cfg["extra_trees"],
    )
    final_model.fit(X.loc[final_mask], y.loc[final_mask], categorical_feature=cat_cols)
    test_pred = final_model.predict_proba(X_test)[:, 1]
    save_submission(output_dir / "submission_lightgbm_single.csv", test, test_pred)

    return {
        "name": cfg["name"],
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "valid_auc": valid_auc,
        "best_iter": best_iter,
        "final_iter": final_iter,
    }


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path):
    cfg = {
        "name": "xgboost_all_full_frontid_d4",
        "train_start": "2024-02-01",
        "valid_start": "2025-04-01",
        "date_mode": "full",
        "use_front_id": True,
        "rounds": 3200,
        "eta": 0.018,
        "max_depth": 4,
        "min_child_weight": 35,
        "subsample": 0.86,
        "colsample_bytree": 0.82,
        "reg_alpha": 2.0,
        "reg_lambda": 18.0,
        "iter_mult": 1.12,
        "seed": SEED + 501,
    }

    X, y, X_test, _ = prepare_tree_data(train, test, cfg["date_mode"], cfg["use_front_id"])
    fit_mask, valid_mask, final_mask = temporal_masks(train, cfg["train_start"], cfg["valid_start"])

    pos = y.loc[fit_mask].sum()
    neg = fit_mask.sum() - pos
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": cfg["max_depth"],
        "eta": cfg["eta"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "min_child_weight": cfg["min_child_weight"],
        "lambda": cfg["reg_lambda"],
        "alpha": cfg["reg_alpha"],
        "scale_pos_weight": float(neg / pos),
        "max_bin": 256,
        "seed": cfg["seed"],
        "nthread": -1,
    }

    dtrain = xgb.DMatrix(X.loc[fit_mask], label=y.loc[fit_mask])
    dvalid = xgb.DMatrix(X.loc[valid_mask], label=y.loc[valid_mask])
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["rounds"],
        evals=[(dvalid, "valid")],
        early_stopping_rounds=250,
        verbose_eval=250,
    )
    best_iter = int(booster.best_iteration + 1)
    valid_pred = booster.predict(dvalid, iteration_range=(0, best_iter))
    valid_auc = roc_auc_score(y.loc[valid_mask], valid_pred)

    pos_final = y.loc[final_mask].sum()
    neg_final = final_mask.sum() - pos_final
    final_rounds = max(300, min(cfg["rounds"], int(math.ceil(best_iter * cfg["iter_mult"]))))
    params["scale_pos_weight"] = float(neg_final / pos_final)
    params["seed"] = cfg["seed"] + 1000

    final_model = xgb.train(
        params,
        xgb.DMatrix(X.loc[final_mask], label=y.loc[final_mask]),
        num_boost_round=final_rounds,
        verbose_eval=False,
    )
    test_pred = final_model.predict(xgb.DMatrix(X_test))
    save_submission(output_dir / "submission_xgboost_single.csv", test, test_pred)

    return {
        "name": cfg["name"],
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "valid_auc": valid_auc,
        "best_iter": best_iter,
        "final_iter": final_rounds,
    }


def optimize_blend(y_valid: np.ndarray, model_results: list[dict]) -> dict:
    """Подбор весов blend на temporal validation.

    Случайный поиск по Dirichlet не требует scipy и хорошо подходит для 3 моделей.
    Дополнительно сравниваем blend вероятностей и blend рангов.
    """
    names = [m["name"] for m in model_results]
    valid_prob = np.column_stack([m["valid_pred"] for m in model_results])
    valid_rank = np.column_stack([normalized_rank(m["valid_pred"]) for m in model_results])

    rng = np.random.default_rng(SEED)
    candidates = []
    for _ in range(20000):
        weights = rng.dirichlet(np.ones(len(model_results)) * 2.0)
        candidates.append(("prob", roc_auc_score(y_valid, valid_prob @ weights), weights))
        candidates.append(("rank", roc_auc_score(y_valid, valid_rank @ weights), weights))

    # Несколько ручных весов рядом с лучшей одиночной моделью, чтобы поиск был устойчивее.
    if len(model_results) == 3:
        for weights in [
            np.array([1 / 3, 1 / 3, 1 / 3]),
            np.array([0.20, 0.35, 0.45]),
            np.array([0.18, 0.33, 0.49]),
            np.array([0.15, 0.30, 0.55]),
        ]:
            candidates.append(("prob", roc_auc_score(y_valid, valid_prob @ weights), weights))
            candidates.append(("rank", roc_auc_score(y_valid, valid_rank @ weights), weights))

    kind, auc, weights = max(candidates, key=lambda x: x[1])
    return {
        "kind": kind,
        "auc": float(auc),
        "weights": weights,
        "names": names,
    }


def save_submission(path: Path, test: pd.DataFrame, pred: np.ndarray):
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    sub = pd.DataFrame({ID_COL: test[ID_COL].reset_index(drop=True), TARGET: pred})

    assert list(sub.columns) == [ID_COL, TARGET]
    assert len(sub) == len(test)
    assert sub[ID_COL].equals(test[ID_COL].reset_index(drop=True))
    assert sub[TARGET].notna().all()
    assert np.isfinite(sub[TARGET]).all()
    assert ((sub[TARGET] >= 0) & (sub[TARGET] <= 1)).all()

    sub.to_csv(path, index=False)
    print(f"saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train blend model for Alfa Bank credit offer task.")
    parser.add_argument("--train", type=Path, default=Path("data/train_apps.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_apps.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


args = parse_args()
args.output.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(args.train)
test = pd.read_csv(args.test)

print("train", train.shape, "test", test.shape)
print("target rate", train[TARGET].mean())
print("train dates", train[DATE_COL].min(), train[DATE_COL].max())
print("test dates", test[DATE_COL].min(), test[DATE_COL].max())

results = [
    train_catboost(train, test, args.output),
    train_lightgbm(train, test, args.output),
    train_xgboost(train, test, args.output),
]

_, valid_mask, _ = temporal_masks(train, "2024-02-01", "2025-04-01")
y_valid = train.loc[valid_mask, TARGET].astype(int).to_numpy()

blend_info = optimize_blend(y_valid, results)
test_matrix = np.column_stack([m["test_pred"] for m in results])
if blend_info["kind"] == "rank":
    test_matrix = np.column_stack([normalized_rank(m["test_pred"]) for m in results])
final_pred = test_matrix @ blend_info["weights"]

final_path = args.output / "submission_top3_optimized_blend.csv"
save_submission(final_path, test, final_pred)

metadata = {
    "final_submission": str(final_path),
    "blend_kind": blend_info["kind"],
    "blend_valid_auc": blend_info["auc"],
    "blend_model_names": blend_info["names"],
    "blend_weights": blend_info["weights"].tolist(),
    "single_model_results": [
        {
            "name": m["name"],
            "valid_auc": float(m["valid_auc"]),
            "best_iter": int(m["best_iter"]),
            "final_iter": int(m["final_iter"]),
        }
        for m in results
    ],
}
(args.output / "run_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(metadata, ensure_ascii=False, indent=2))

