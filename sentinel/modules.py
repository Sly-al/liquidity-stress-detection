from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import rolling_mad_score


def module_m1_reserves(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "m1_actual_corr_accounts", "m1_required_reserves", "m1_ruonia", "m1_end_period_flag"]].copy()
    out["m1_reserve_spread"] = out["m1_actual_corr_accounts"] - out["m1_required_reserves"]
    out["m1_mad_spread"] = rolling_mad_score(out["m1_reserve_spread"])
    out["m1_mad_ruonia"] = rolling_mad_score(out["m1_ruonia"])
    out["m1_overbuffer_flag"] = ((out["m1_mad_spread"] > 1.5) & (out["m1_end_period_flag"] == 1)).astype(int)
    return out


def module_m2_repo(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "m2_term_days", "m2_repo_demand", "m2_repo_allotment", "m2_key_rate", "m2_cutoff_rate"]].copy()
    out["m2_cover_ratio"] = (out["m2_repo_demand"] / out["m2_repo_allotment"]).replace([np.inf, -np.inf], np.nan)
    out["m2_rate_spread"] = out["m2_cutoff_rate"] - out["m2_key_rate"]
    out[["m2_term_days", "m2_repo_demand", "m2_repo_allotment", "m2_cutoff_rate"]] = out[
        ["m2_term_days", "m2_repo_demand", "m2_repo_allotment", "m2_cutoff_rate"]
    ].ffill()
    out["m2_cover_ratio"] = out["m2_cover_ratio"].ffill().fillna(1.0)
    out["m2_rate_spread"] = out["m2_rate_spread"].ffill().fillna(0.0)
    out["m2_mad_cover"] = rolling_mad_score(out["m2_cover_ratio"])
    out["m2_mad_rate_spread"] = rolling_mad_score(out["m2_rate_spread"])
    out["m2_demand_flag"] = (out["m2_cover_ratio"] > 2.0).astype(int)
    return out


def module_m3_ofz(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "m3_offer", "m3_demand", "m3_placed", "m3_weighted_yield", "m3_curve_benchmark"]].copy()
    out["m3_cover_ratio"] = (out["m3_demand"] / out["m3_offer"]).replace([np.inf, -np.inf], np.nan)
    out["m3_yield_spread"] = out["m3_weighted_yield"] - out["m3_curve_benchmark"]
    out[["m3_offer", "m3_demand", "m3_placed", "m3_weighted_yield", "m3_curve_benchmark"]] = out[
        ["m3_offer", "m3_demand", "m3_placed", "m3_weighted_yield", "m3_curve_benchmark"]
    ].ffill()
    out["m3_cover_ratio"] = out["m3_cover_ratio"].ffill().fillna(1.8)
    out["m3_yield_spread"] = out["m3_yield_spread"].ffill().fillna(0.0)
    out["m3_stress_cover"] = -out["m3_cover_ratio"]
    out["m3_mad_cover"] = rolling_mad_score(out["m3_stress_cover"])
    out["m3_mad_yield_spread"] = rolling_mad_score(out["m3_yield_spread"])
    out["m3_nedospros_flag"] = (out["m3_cover_ratio"] < 1.2).astype(int)
    out["m3_perespros_flag"] = (out["m3_cover_ratio"] > 2.0).astype(int)
    return out


def module_m4_tax(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["date", "tax_week_flag", "tax_due_core", "end_of_month_flag", "end_of_quarter_flag", "seasonal_factor"]].copy()


def module_m5_treasury(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["date", "m5_bank_treasury_balances", "m5_eks_deposits", "m5_bank_count"]].copy()
    out["m5_week_delta_balances"] = out["m5_bank_treasury_balances"].diff(5)
    out["m5_month_delta_deposits"] = out["m5_eks_deposits"].diff(21)
    out["m5_budget_drain"] = -out["m5_week_delta_balances"]
    out["m5_deposit_drain"] = -out["m5_month_delta_deposits"]
    out["m5_mad_cbr"] = rolling_mad_score(out["m5_budget_drain"].fillna(0.0))
    out["m5_mad_roskazna"] = rolling_mad_score(out["m5_deposit_drain"].fillna(0.0))
    out["m5_budget_drain_flag"] = ((out["m5_budget_drain"] > 350) | (out["m5_deposit_drain"] > 300)).astype(int)
    return out


def build_module_outputs(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "m1": module_m1_reserves(frame),
        "m2": module_m2_repo(frame),
        "m3": module_m3_ofz(frame),
        "m4": module_m4_tax(frame),
        "m5": module_m5_treasury(frame),
    }
