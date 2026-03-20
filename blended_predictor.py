
import ast
import re
import numpy as np
import pandas as pd
import joblib

BENCHMARK_MODEL_PATH = "benchmark_model_district_year_season_area5.joblib"
V2_MODEL_PATH = "benchmark_model_v2_tabular.joblib"
V21_MODEL_PATH = "benchmark_model_v21_tabular.joblib"

benchmark_model = joblib.load(BENCHMARK_MODEL_PATH)
model_v2 = joblib.load(V2_MODEL_PATH)
model_v21 = joblib.load(V21_MODEL_PATH)

WEATHER_FEATURES = [
    "T2M", "T2M_MAX", "T2M_MIN",
    "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"
]

def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        return ast.literal_eval(x)
    return []

def to_float_array(x):
    vals = parse_list_cell(x)
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def detect_eo_cols(df):
    eo_cols = [c for c in df.columns if ("NDVI" in c.upper() or "NDWI" in c.upper())]
    def sort_key(col):
        m = re.search(r"D(\d+)", col.upper())
        dekad = int(m.group(1)) if m else 999
        feat = 0 if "NDVI" in col.upper() else 1
        return (dekad, feat)
    return sorted(eo_cols, key=sort_key)

def eo_matrix_from_row(row, eo_cols):
    dekads = sorted({int(re.search(r"D(\d+)", c.upper()).group(1)) for c in eo_cols})
    eo = np.full((len(dekads), 2), np.nan, dtype=np.float32)
    d2i = {d: i for i, d in enumerate(dekads)}
    for c in eo_cols:
        m = re.search(r"D(\d+)", c.upper())
        if not m:
            continue
        d = int(m.group(1))
        feat_idx = 0 if "NDVI" in c.upper() else 1
        try:
            eo[d2i[d], feat_idx] = float(row[c])
        except Exception:
            eo[d2i[d], feat_idx] = np.nan
    return eo

def max_consecutive(condition_arr):
    best = 0
    cur = 0
    for v in condition_arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def slope_over_time(arr):
    arr = np.array(arr, dtype=np.float32)
    idx = np.where(~np.isnan(arr))[0]
    if len(idx) < 2:
        return np.nan
    x = idx.astype(np.float32)
    y = arr[idx]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

def auc_valid(arr):
    arr = np.array(arr, dtype=np.float32)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nansum(arr))

def build_v2_features_for_one(row_df):
    row = row_df.iloc[0]
    out = {}

    for col in WEATHER_FEATURES:
        arr = to_float_array(row[col])
        out[f"{col}_mean"] = float(np.nanmean(arr)) if len(arr) else np.nan
        out[f"{col}_std"] = float(np.nanstd(arr)) if len(arr) else np.nan
        out[f"{col}_min"] = float(np.nanmin(arr)) if len(arr) else np.nan
        out[f"{col}_max"] = float(np.nanmax(arr)) if len(arr) else np.nan

    rain = to_float_array(row["PRECTOTCORR"])
    tmax = to_float_array(row["T2M_MAX"])
    tmean = to_float_array(row["T2M"])
    solar = to_float_array(row["ALLSKY_SFC_SW_DWN"])
    rh = to_float_array(row["RH2M"])

    out["rain_total"] = float(np.nansum(rain)) if len(rain) else np.nan
    out["rainy_days_ge_1mm"] = int(np.sum(np.nan_to_num(rain, nan=0.0) >= 1.0))
    out["rainy_days_ge_5mm"] = int(np.sum(np.nan_to_num(rain, nan=0.0) >= 5.0))
    out["tmax_days_ge_35"] = int(np.sum(np.nan_to_num(tmax, nan=-999.0) >= 35.0))
    out["tmax_days_ge_38"] = int(np.sum(np.nan_to_num(tmax, nan=-999.0) >= 38.0))
    out["tmean_days_ge_30"] = int(np.sum(np.nan_to_num(tmean, nan=-999.0) >= 30.0))
    out["solar_total"] = float(np.nansum(solar)) if len(solar) else np.nan
    out["rh_mean"] = float(np.nanmean(rh)) if len(rh) else np.nan
    out["weather_seq_len"] = len(rain)

    eo_cols = detect_eo_cols(row_df)
    eo = eo_matrix_from_row(row, eo_cols)
    if eo.size == 0:
        out.update({
            "eo_len": 0, "eo_completeness": 0.0,
            "ndvi_mean": np.nan, "ndvi_std": np.nan, "ndvi_min": np.nan, "ndvi_max": np.nan,
            "ndvi_early": np.nan, "ndvi_mid": np.nan, "ndvi_late": np.nan, "ndvi_delta_first_last": np.nan,
            "ndwi_mean": np.nan, "ndwi_std": np.nan, "ndwi_min": np.nan, "ndwi_max": np.nan,
            "ndwi_early": np.nan, "ndwi_mid": np.nan, "ndwi_late": np.nan, "ndwi_delta_first_last": np.nan,
        })
    else:
        ndvi = eo[:, 0]
        ndwi = eo[:, 1]
        ndvi_valid = ndvi[~np.isnan(ndvi)]
        ndwi_valid = ndwi[~np.isnan(ndwi)]
        out.update({
            "eo_len": eo.shape[0],
            "eo_completeness": float(np.mean(~np.isnan(eo))),
            "ndvi_mean": float(np.nanmean(ndvi)), "ndvi_std": float(np.nanstd(ndvi)),
            "ndvi_min": float(np.nanmin(ndvi)), "ndvi_max": float(np.nanmax(ndvi)),
            "ndvi_early": float(np.nanmean(ndvi[:max(1, len(ndvi)//3)])),
            "ndvi_mid": float(np.nanmean(ndvi[len(ndvi)//3:max(2, 2*len(ndvi)//3)])),
            "ndvi_late": float(np.nanmean(ndvi[max(0, 2*len(ndvi)//3):])),
            "ndvi_delta_first_last": float(ndvi_valid[-1] - ndvi_valid[0]) if len(ndvi_valid) >= 2 else np.nan,
            "ndwi_mean": float(np.nanmean(ndwi)), "ndwi_std": float(np.nanstd(ndwi)),
            "ndwi_min": float(np.nanmin(ndwi)), "ndwi_max": float(np.nanmax(ndwi)),
            "ndwi_early": float(np.nanmean(ndwi[:max(1, len(ndwi)//3)])),
            "ndwi_mid": float(np.nanmean(ndwi[len(ndwi)//3:max(2, 2*len(ndwi)//3)])),
            "ndwi_late": float(np.nanmean(ndwi[max(0, 2*len(ndwi)//3):])),
            "ndwi_delta_first_last": float(ndwi_valid[-1] - ndwi_valid[0]) if len(ndwi_valid) >= 2 else np.nan,
        })

    out["district_std"] = row["district_std"]
    out["crop_year"] = int(row["crop_year"])
    out["season"] = row["season"]
    out["area_ha"] = float(row.get("area_ha", np.nan))
    return pd.DataFrame([out])

def build_v21_features_for_one(row_df):
    row = row_df.iloc[0]
    out = {}

    tmean = to_float_array(row["T2M"])
    tmax = to_float_array(row["T2M_MAX"])
    tmin = to_float_array(row["T2M_MIN"])
    rain = to_float_array(row["PRECTOTCORR"])
    rh = to_float_array(row["RH2M"])
    solar = to_float_array(row["ALLSKY_SFC_SW_DWN"])

    out["rain_total"] = float(np.nansum(rain)) if len(rain) else np.nan
    out["rain_days_ge_1"] = int(np.sum(np.nan_to_num(rain, nan=0.0) >= 1))
    out["rain_days_ge_10"] = int(np.sum(np.nan_to_num(rain, nan=0.0) >= 10))
    out["dry_spell_max"] = max_consecutive(np.nan_to_num(rain, nan=0.0) < 1)
    out["hot_spell_max_35"] = max_consecutive(np.nan_to_num(tmax, nan=-999.0) >= 35)
    out["hot_spell_max_38"] = max_consecutive(np.nan_to_num(tmax, nan=-999.0) >= 38)
    out["gdd_base10"] = float(np.nansum(np.maximum(tmean - 10.0, 0))) if len(tmean) else np.nan
    out["temp_range_mean"] = float(np.nanmean(tmax - tmin)) if len(tmax) else np.nan
    out["rh_mean"] = float(np.nanmean(rh)) if len(rh) else np.nan
    out["solar_total"] = float(np.nansum(solar)) if len(solar) else np.nan
    out["weather_seq_len"] = len(rain)

    eo_cols = detect_eo_cols(row_df)
    eo = eo_matrix_from_row(row, eo_cols)
    if eo.size == 0:
        out.update({
            "eo_len": 0,
            "ndvi_mean": np.nan, "ndvi_max": np.nan, "ndvi_min": np.nan, "ndvi_slope": np.nan, "ndvi_auc": np.nan,
            "ndwi_mean": np.nan, "ndwi_max": np.nan, "ndwi_min": np.nan, "ndwi_slope": np.nan, "ndwi_auc": np.nan,
        })
    else:
        ndvi = eo[:, 0]
        ndwi = eo[:, 1]
        out.update({
            "eo_len": eo.shape[0],
            "ndvi_mean": float(np.nanmean(ndvi)),
            "ndvi_max": float(np.nanmax(ndvi)),
            "ndvi_min": float(np.nanmin(ndvi)),
            "ndvi_slope": slope_over_time(ndvi),
            "ndvi_auc": auc_valid(ndvi),
            "ndwi_mean": float(np.nanmean(ndwi)),
            "ndwi_max": float(np.nanmax(ndwi)),
            "ndwi_min": float(np.nanmin(ndwi)),
            "ndwi_slope": slope_over_time(ndwi),
            "ndwi_auc": auc_valid(ndwi),
        })

    out["district_std"] = row["district_std"]
    out["crop_year"] = int(row["crop_year"])
    out["season"] = row["season"]
    out["area_ha"] = float(row.get("area_ha", np.nan))
    return pd.DataFrame([out])

def predict_blended_yield(payload):
    row_df = pd.DataFrame([payload])

    # The benchmark model expects area_ha alongside district/year/season
    x_bench = row_df[["district_std", "crop_year", "season", "area_ha"]].copy()
    x_v2 = build_v2_features_for_one(row_df)
    x_v21 = build_v21_features_for_one(row_df)

    pred_bench = float(benchmark_model.predict(x_bench)[0])
    pred_v2 = float(model_v2.predict(x_v2)[0])
    pred_v21 = float(model_v21.predict(x_v21)[0])

    pred_rmse_blend = 0.2 * pred_bench + 0.5 * pred_v2 + 0.3 * pred_v21
    pred_mae_blend = 0.7 * pred_bench + 0.2 * pred_v2 + 0.1 * pred_v21

    return {
        "pred_benchmark": pred_bench,
        "pred_v2": pred_v2,
        "pred_v21": pred_v21,
        "pred_best_rmse_blend": pred_rmse_blend,
        "pred_best_mae_blend": pred_mae_blend,
    }
