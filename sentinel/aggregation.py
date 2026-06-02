from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODULE_LABELS
from .utils import stress_status


MODULE_FEATURES = {
    "m1": ["m1_mad_spread", "m1_mad_ruonia", "m1_overbuffer_flag"],
    "m2": ["m2_mad_cover", "m2_mad_rate_spread", "m2_demand_flag"],
    "m3": ["m3_mad_cover", "m3_mad_yield_spread", "m3_nedospros_flag", "m3_perespros_flag"],
    "m4": ["tax_week_flag", "end_of_month_flag", "end_of_quarter_flag"],
    "m5": ["m5_mad_cbr", "m5_mad_roskazna", "m5_budget_drain_flag"],
}


@dataclass
class AggregationResult:
    frame: pd.DataFrame
    coefficients: pd.DataFrame
    model_quality: dict[str, float]
    feature_columns: list[str]


def combine_modules(raw: pd.DataFrame, modules: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = raw[["date", "ground_truth_liquidity_stress", "latent_stress"]].copy()
    for module_frame in modules.values():
        extra = module_frame.drop(columns=[col for col in ["date"] if col in module_frame.columns])
        combined = pd.concat([combined, extra], axis=1)
    return combined


def _feature_frame(combined: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [feature for features in MODULE_FEATURES.values() for feature in features]
    x = combined[columns].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x["m3_perespros_flag"] *= -0.4
    return x, columns


def aggregate_lsi(combined: pd.DataFrame) -> AggregationResult:
    x, feature_columns = _feature_frame(combined)
    y = combined["ground_truth_liquidity_stress"].astype(float)

    split_date = pd.Timestamp("2022-01-01")
    train_mask = combined["date"] < split_date
    if train_mask.sum() < 500:
        train_mask = pd.Series(True, index=combined.index)

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 2, 25)))
    model.fit(x.loc[train_mask], y.loc[train_mask])
    pred = pd.Series(model.predict(x), index=combined.index).clip(0, 100)

    seasonal_adjusted = (pred * combined["seasonal_factor"].fillna(1.0)).clip(0, 100)
    out = combined.copy()
    out["lsi"] = seasonal_adjusted.round(1)
    out["status"] = out["lsi"].map(stress_status)

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridgecv"]
    standardized = (x - scaler.mean_) / scaler.scale_
    raw_contrib = standardized.mul(ridge.coef_, axis=1)

    module_contrib = pd.DataFrame({"date": out["date"]})
    for module, features in MODULE_FEATURES.items():
        valid = [feature for feature in features if feature in raw_contrib.columns]
        value = raw_contrib[valid].sum(axis=1)
        module_contrib[f"{module}_contribution_raw"] = value

    positive = module_contrib.filter(like="_raw").clip(lower=0)
    denom = positive.sum(axis=1).replace(0, np.nan)
    for module in MODULE_FEATURES:
        raw_name = f"{module}_contribution_raw"
        out[f"{module}_contribution"] = (positive[raw_name] / denom * out["lsi"]).fillna(out["lsi"] / 5).round(2)

    coefs = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": ridge.coef_,
            "module": [MODULE_LABELS[next(key for key, feats in MODULE_FEATURES.items() if feature in feats)] for feature in feature_columns],
        }
    ).sort_values("coefficient", ascending=False)

    r2_train = float(model.score(x.loc[train_mask], y.loc[train_mask]))
    r2_holdout = float(model.score(x.loc[~train_mask], y.loc[~train_mask])) if (~train_mask).sum() else r2_train
    return AggregationResult(out, coefs, {"r2_train": r2_train, "r2_holdout": r2_holdout}, feature_columns)


def sensitivity_analysis(result: AggregationResult) -> pd.DataFrame:
    base = result.frame.copy()
    modules = list(MODULE_FEATURES)
    rows = []
    for module in modules:
        contribution_col = f"{module}_contribution"
        for shock in (-0.2, 0.2):
            shocked = base["lsi"] + base[contribution_col] * shock
            rows.append(
                {
                    "module": MODULE_LABELS[module],
                    "shock": f"{shock:+.0%}",
                    "latest_lsi": float(shocked.iloc[-1].clip(0, 100)),
                    "avg_delta": float((shocked - base["lsi"]).mean()),
                    "max_abs_delta": float((shocked - base["lsi"]).abs().max()),
                }
            )
    return pd.DataFrame(rows)
