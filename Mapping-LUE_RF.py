"""
Random Forest Regression for LUE Upscaling with Spatial LOOCV and Dual Conformal Prediction
==========================================================================================
# Author: Yong Lin
# Affiliation: Institute of Geographic Sciences and Natural Resources Research, CAS
# Contact: linyong0018@igsnrr.ac.cn
# Python Version: 3.11.4
Description:
    This script implements an advanced Random Forest regression workflow for upscaling three 
    ecosystem Light Use Efficiency (LUE) metrics (LUEmax, LUEinc, LUEact). Compared to 
    standard LOOCV, this version incorporates spatial independence and rigorous uncertainty 
    quantification. The core workflow includes:
    1. Data preprocessing (site-level imputation and spatial coordinate validation)
    2. Spatial LOOCV training (Buffer-LOOCV) with a 100km geographic exclusion radius
    3. Dual Conformal Prediction (Split & Spatial LOOCV) for robust 95% prediction intervals
    4. SHAP feature importance analysis for model interpretability and ranking
    5. Spatial upscaling with bootstrap uncertainty estimation (mean, std, 95% CI)
    6. Export of high-resolution GeoTIFFs for point predictions and uncertainty bounds
Dependencies:
    pandas, numpy, matplotlib, scikit-learn, shap, rasterio, tqdm, joblib, pathlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
from pathlib import Path
import warnings
import joblib
import rasterio
from tqdm import tqdm
import math

# ==============================================================================
# Core Configuration (Spatial LOOCV + Upscaling + Dual Conformal Prediction)
# ==============================================================================
# Target LUE metrics for regression
TARGETS = ['LUEmax', 'LUEact', 'LUEinc']
# Predictor variables (matched with raster file names)
FEATURES = ['TA_GS', 'FD_GS', 'VPD_GS', 'P_GS', 'LAI_GS',
            'GSL', 'CHL', 'SLA', 'LN', 'LP', 'CNR', 'SOC', 'SP', 'CI']
# Modeling hyperparameters
MIN_SAMPLES = 20  # Minimum samples required for model training
RANDOM_STATE = 42  # Fixed seed for reproducibility
N_BOOTSTRAP = 100  # Bootstrap iterations for uncertainty estimation
# Conformal Prediction parameters
CONFORMAL_COVER = 0.95  # Desired coverage level (95% prediction interval)
CONFORMAL_CALIB_FRAC = 0.2  # Split Conformal: 20% calibration data
# Spatial LOOCV parameter
SPATIAL_EXCLUSION_RADIUS = 100  # 100km exclusion radius for spatial LOOCV
# File paths (update according to local environment)
EXCEL_FILE = r"\Example data_LUE.xlsx"
RASTER_PATHS = {
    'TA_GS': r"\TA.tif",
    'FD_GS': r"\FD.tif",
    'VPD_GS': r"\VPD.tif",
    'P_GS': r"\P.tif",
    'LAI_GS': r"\LAI.tif",
    'GSL': r"\GSL.tif",
    'CHL': r"\LCC.tif",
    'SLA': r"\SLA.tif",
    'LN': r"\LN.tif",
    'LP': r"\LP.tif",
    'CNR': r"\CNR.tif",
    'SN': r"\SN.tif",
    'SP': r"\SP.tif",
    'SOC': r"\SOC.tif",
    'MLA': r"\MLA.tif",
    'CI': r"\CI.tif"
}
# Result directory (automatically created if not exists)
MAIN_ROOT_DIR = Path.home() / 'Desktop' / 'RF_Upscaling_Spatial_LOOCV_Validation_100km'

# ==============================================================================
# Basic Configuration (No modification required)
# ==============================================================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial']  # Standard font for scientific publications
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')  # Non-interactive backend for server-side execution


# ==============================================================================
# Utility Functions
# ==============================================================================
def check_file_exists(file_path: Path | str, desc: str = "File") -> None:
    """
    Check if a file exists and raise FileNotFoundError if not.

    Parameters
    ----------
    file_path : Path or str
        Path to the target file
    desc : str, optional
        Description of the file for error message (default: "File")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path.absolute()}")


def validate_raster_size(raster_paths: dict, reference_feat: str = None) -> tuple:
    """
    Validate all raster files have consistent dimensions and return reference metadata.

    Parameters
    ----------
    raster_paths : dict
        Dictionary mapping feature names to raster file paths
    reference_feat : str, optional
        Reference feature for dimension comparison (default: first feature in dict)

    Returns
    -------
    tuple
        (reference height/width, reference raster metadata)
    """
    if not reference_feat:
        reference_feat = list(raster_paths.keys())[0]
    check_file_exists(raster_paths[reference_feat], f"Reference raster ({reference_feat})")

    with rasterio.open(raster_paths[reference_feat]) as src:
        ref_dims = src.shape
        ref_meta = src.meta.copy()

    for feat, path in raster_paths.items():
        check_file_exists(path, f"Feature raster ({feat})")
        with rasterio.open(path) as src:
            if src.shape != ref_dims:
                raise ValueError(
                    f"Raster dimension mismatch: {feat} ({src.shape}) vs reference {reference_feat} ({ref_dims})"
                )
    return ref_dims, ref_meta


def print_separator(msg: str, length: int = 80) -> None:
    """
    Print a formatted separator line for log readability.

    Parameters
    ----------
    msg : str
        Message to display in the separator
    """
    print(f"\n{'=' * length}\n{msg}\n{'=' * length}")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance (km) between two points using haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371


def get_sites_beyond_radius(df: pd.DataFrame, target_site: str, radius_km: float) -> list:
    """
    Get list of sites that are more than radius_km away from the target site.
    """
    target_coords = df[df['Site'] == target_site][['LAT', 'LON']].iloc[0]
    target_lat, target_lon = target_coords['LAT'], target_coords['LON']

    site_distances = {}
    for site in df['Site'].unique():
        if site == target_site:
            continue
        site_coords = df[df['Site'] == site][['LAT', 'LON']].iloc[0]
        dist = haversine_distance(target_lat, target_lon, site_coords['LAT'], site_coords['LON'])
        site_distances[site] = dist

    return [site for site, dist in site_distances.items() if dist > radius_km]


# ==============================================================================
# Core Modeling Functions
# ==============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target_var: str,
                      site: str = None) -> dict:
    """
    Calculate regression metrics (R², RMSE) for model evaluation.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {'target_variable': target_var, 'model': model_name, 'R2': r2, 'RMSE': rmse}
    if site:
        metrics['left_out_site'] = site
    return metrics


def calculate_cv_metrics(fold_metrics: list) -> tuple:
    """
    Calculate global cross-validation metrics from Spatial LOOCV fold results.
    """
    cv_df = pd.DataFrame(fold_metrics)
    y_true_all = np.concatenate(cv_df['y_true'].values)
    y_pred_all = np.concatenate(cv_df['y_pred'].values)
    global_r2 = r2_score(y_true_all, y_pred_all)
    global_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    avg_metrics = pd.DataFrame([{
        'target_variable': cv_df['target_variable'].iloc[0],
        'model': model,
        'mean_R2': round(global_r2, 3),
        'std_R2': 0.0,
        'mean_RMSE': round(global_rmse, 3),
        'std_RMSE': 0.0
    } for model in cv_df['model'].unique()])

    return avg_metrics, cv_df


def run_spatial_leave_one_site_out(X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame, target_var: str) -> list:
    """
    Perform Spatial Leave-One-Site-Out (Spatial LOOCV) cross-validation with Random Forest.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable vector
    df_full : pd.DataFrame
        Full dataset including Site and Coordinates
    target_var : str
        Name of the target variable

    Returns
    -------
    list
        List of dictionaries containing fold-wise results
    """
    spatial_loso_results = []
    valid_sites = df_full['Site'].unique()

    if len(valid_sites) < 2:
        raise ValueError(f"Insufficient valid sites ({len(valid_sites)}) for Spatial LOOCV validation")

    print(
        f"Performing Spatial LOOCV validation ({SPATIAL_EXCLUSION_RADIUS}km exclusion): {len(valid_sites)} valid sites")

    base_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    for fold, leave_site in enumerate(valid_sites, 1):
        # Identify training sites outside the spatial buffer
        train_sites = get_sites_beyond_radius(df_full, leave_site, SPATIAL_EXCLUSION_RADIUS)

        if len(train_sites) < 1:
            warnings.warn(f"Skipping site {leave_site}: No training sites beyond {SPATIAL_EXCLUSION_RADIUS}km")
            continue

        X_train, X_val = X[df_full['Site'].isin(train_sites)], X[df_full['Site'] == leave_site]
        y_train, y_val = y[df_full['Site'].isin(train_sites)], y[df_full['Site'] == leave_site]

        if len(X_train) < MIN_SAMPLES:
            warnings.warn(f"Skipping site {leave_site}: Insufficient training samples ({len(X_train)})")
            continue

        model = base_model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        spatial_loso_results.append({
            'target_variable': target_var,
            'model': 'RandomForest',
            'left_out_site': leave_site,
            'y_true': y_val.values,
            'y_pred': y_pred
        })

    if not spatial_loso_results:
        raise RuntimeError("No valid results generated from Spatial LOOCV validation")

    return spatial_loso_results


def save_cv_results(output_dir: Path, X: pd.DataFrame, y: pd.Series, target_var: str, fold_metrics: list,
                    cv_avg_metrics: pd.DataFrame) -> tuple:
    """
    Save Spatial LOOCV validation results (metrics, plots, SHAP analysis).
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    suffix = "Spatial_LOOCV_Validation"

    # 1. Save detailed and average metrics
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(output_dir / f'{target_var}_RF_{suffix}_detailed_metrics.csv', index=False)

    avg_df = cv_avg_metrics[cv_avg_metrics['model'] == 'RandomForest'].copy()
    avg_df.to_csv(output_dir / f'{target_var}_RF_{suffix}_average_metrics.csv', index=False)

    # 2. SHAP feature importance analysis
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_importance = pd.DataFrame({
            'feature': FEATURES,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(shap_importance['feature'], shap_importance['shap_importance'], color='#3C5E20')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'{target_var} - Random Forest Feature Importance (SHAP, {suffix})')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{target_var}_RF_SHAP_importance_{suffix}.png', dpi=300, bbox_inches='tight')

        shap_importance['rank'] = range(1, len(shap_importance) + 1)
        shap_importance[['rank', 'feature', 'shap_importance']].to_csv(
            output_dir / f'{target_var}_RF_SHAP_importance_ranking_{suffix}.csv', index=False)
    except Exception as e:
        warnings.warn(f"SHAP analysis failed for {target_var}: {str(e)}")
    finally:
        plt.close()

    # 3. Validation scatter plot
    try:
        all_y_true = np.concatenate([fm['y_true'] for fm in fold_metrics])
        all_y_pred = np.concatenate([fm['y_pred'] for fm in fold_metrics])
        val_range = [min(all_y_true.min(), all_y_pred.min()), max(all_y_true.max(), all_y_pred.max())]

        plt.figure(figsize=(8, 6))
        plt.scatter(all_y_true, all_y_pred, alpha=0.6, color='#3C5E20', s=30)
        plt.plot(val_range, val_range, 'k--', lw=2)
        plt.xlabel('Observed Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{target_var} - Random Forest ({suffix}, R² = {avg_df["mean_R2"].iloc[0]:.3f})')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{target_var}_RF_{suffix}_scatter.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        warnings.warn(f"Scatter plot failed for {target_var}: {str(e)}")
    finally:
        plt.close()

    return fold_df, avg_df


def bootstrap_uncertainty_estimation(model, X: np.ndarray) -> tuple:
    """
    Estimate prediction uncertainty using bootstrap resampling.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model with stored X_train_ and y_train_
    X : np.ndarray
        Feature matrix for prediction

    Returns
    -------
    tuple
        (mean, std, 2.5 percentile, 97.5 percentile)
    """
    if not (hasattr(model, 'X_train_') and hasattr(model, 'y_train_')):
        raise AttributeError("Model must have X_train_ and y_train_ attributes for bootstrap")

    X_train, y_train = model.X_train_, model.y_train_
    X_clean = X.copy()
    if np.isnan(X_clean).any():
        train_mean = np.nanmean(X_train, axis=0)
        X_clean[np.isnan(X_clean)] = np.take(train_mean, np.where(np.isnan(X_clean))[1])

    bootstrap_preds = []
    for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrap iterations"):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        boot_model = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE + i, n_jobs=-1
        ).fit(X_train.iloc[idx], y_train.iloc[idx])
        bootstrap_preds.append(boot_model.predict(X_clean))

    bootstrap_preds = np.array(bootstrap_preds)
    return (np.mean(bootstrap_preds, axis=0), np.std(bootstrap_preds, axis=0),
            np.percentile(bootstrap_preds, 2.5, axis=0), np.percentile(bootstrap_preds, 97.5, axis=0))


# ==============================================================================
# Conformal Prediction Core Functions
# ==============================================================================
def train_split_conformal_model(X: pd.DataFrame, y: pd.Series, target_var: str, model_dir: Path) -> tuple:
    """
    Train Random Forest model with Split Conformal Prediction calibration.
    """
    print(f"\n[Split Conformal] Training for {target_var} (coverage={CONFORMAL_COVER})")
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=CONFORMAL_CALIB_FRAC, random_state=RANDOM_STATE, shuffle=True
    )

    base_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    calib_residuals = np.abs(y_calib - base_model.predict(X_calib))

    n_calib = len(calib_residuals)
    q_level = np.ceil((n_calib + 1) * CONFORMAL_COVER) / n_calib
    conformal_quantile = np.quantile(calib_residuals, q_level, method='higher')

    joblib.dump({'model': base_model, 'conformal_quantile': conformal_quantile, 'calib_residuals': calib_residuals,
                 'coverage': CONFORMAL_COVER, 'method': 'split_conformal'},
                model_dir / f'{target_var}_RF_split_conformal_model.joblib')

    return base_model, conformal_quantile, calib_residuals


def train_spatial_loocv_conformal_model(X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame, target_var: str,
                                        model_dir: Path) -> tuple:
    """
    Train Random Forest model with Spatial LOOCV Conformal Prediction (100km exclusion).
    """
    print(f"\n[Spatial LOOCV Conformal] Training for {target_var} (coverage={CONFORMAL_COVER}, 100km exclusion)")
    valid_sites = df_full['Site'].unique()
    all_residuals = []

    for fold, leave_site in enumerate(valid_sites, 1):
        train_sites = get_sites_beyond_radius(df_full, leave_site, SPATIAL_EXCLUSION_RADIUS)
        if len(train_sites) < 1: continue

        X_train, X_val = X[df_full['Site'].isin(train_sites)], X[df_full['Site'] == leave_site]
        y_train, y_val = y[df_full['Site'].isin(train_sites)], y[df_full['Site'] == leave_site]

        if len(X_train) < MIN_SAMPLES: continue

        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
        all_residuals.extend(np.abs(y_val - model.predict(X_val)))

    calib_residuals = np.array(all_residuals)
    n_calib = len(calib_residuals)
    q_level = np.ceil((n_calib + 1) * CONFORMAL_COVER) / n_calib
    conformal_quantile = np.quantile(calib_residuals, q_level, method='higher')

    final_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
    joblib.dump({'model': final_model, 'conformal_quantile': conformal_quantile, 'calib_residuals': calib_residuals,
                 'coverage': CONFORMAL_COVER, 'method': 'spatial_loocv_conformal'},
                model_dir / f'{target_var}_RF_spatial_loocv_conformal_model.joblib')

    return final_model, conformal_quantile, calib_residuals


def conformal_prediction_interval(model, conformal_quantile, X: np.ndarray) -> tuple:
    """
    Generate prediction intervals using Conformal Prediction (General for Split/Spatial).
    """
    X_clean = X.copy()
    if np.isnan(X_clean).any():
        train_mean = np.nanmean(
            model.X_train_ if hasattr(model, 'X_train_') else model.feature_importances_.reshape(1, -1), axis=0)
        X_clean[np.isnan(X_clean)] = np.take(train_mean, np.where(np.isnan(X_clean))[1])

    point_pred = model.predict(X_clean)
    lower_bound = np.maximum(point_pred - conformal_quantile, 0.0)
    upper_bound = point_pred + conformal_quantile

    return point_pred, lower_bound, upper_bound


# ==============================================================================
# Raster Loading & Saving Functions
# ==============================================================================
def load_feature_raster(features: list) -> tuple:
    """
    Load and validate raster data for predictor variables.
    """
    print_separator("Loading raster feature data")
    ref_feat = features[0]
    with rasterio.open(RASTER_PATHS[ref_feat]) as src:
        ref_meta, valid_mask = src.meta.copy(), np.ones(src.shape, dtype=bool)

    data_arrays = []
    for feat in features:
        with rasterio.open(RASTER_PATHS[feat]) as src:
            band_data = src.read(1).astype(np.float32)
            nodata_mask = np.isnan(band_data) | (band_data == src.nodata) if src.nodata else np.isnan(band_data)
            band_data[nodata_mask], valid_mask = np.nan, valid_mask & ~nodata_mask
            data_arrays.append(band_data.flatten())

    X_full = np.vstack(data_arrays).T
    print(f"Raster loading complete: {np.sum(valid_mask)} valid pixels")
    return X_full, valid_mask, ref_meta


def save_tif(output_path: Path, data: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    """
    Save prediction results to GeoTIFF format with georeferencing.
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    out_array = np.full(mask.shape, np.nan, dtype=np.float32)
    out_array.flat[np.where(mask.flatten())[0]] = data

    out_meta = meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan, 'compress': 'lzw'})
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_array, 1)
    print(f"Raster saved successfully: {output_path}")


# ==============================================================================
# Main Workflow Functions
# ==============================================================================
def train_spatial_loso_model(df_clean: pd.DataFrame, output_dir: Path) -> None:
    """
    Train Random Forest models with Spatial LOOCV and save results (including Dual Conformal).
    """
    print_separator("Starting Spatial LOOCV Validation Modeling (Dual Conformal Prediction)")
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    for target_var in TARGETS:
        print(f"\nProcessing target variable: {target_var}")
        if df_clean[target_var].notna().sum() < MIN_SAMPLES: continue

        X, y = df_clean[FEATURES], df_clean[target_var]
        loso_results = run_spatial_leave_one_site_out(X, y, df_clean, target_var)
        avg_metrics, fold_metrics = calculate_cv_metrics(loso_results)
        save_cv_results(output_dir, X, y, target_var, loso_results, avg_metrics)

        # 1. Base Bootstrap Model
        final_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X, y)
        final_model.X_train_, final_model.y_train_ = X, y
        joblib.dump(final_model, model_dir / f'{target_var}_RF_Spatial_LOOCV_model.joblib')

        # 2. Conformal Models
        train_split_conformal_model(X, y, target_var, model_dir)
        train_spatial_loocv_conformal_model(X, y, df_clean, target_var, model_dir)


def upscale_spatial_loso_model() -> None:
    """
    Upscale Spatial LOOCV models to spatial raster predictions with Bootstrap and Dual Conformal.
    """
    print_separator("Starting Spatial Upscaling with Dual Conformal Prediction")
    validate_raster_size(RASTER_PATHS)
    X_full, valid_mask, out_meta = load_feature_raster(FEATURES)
    X_pred = X_full[valid_mask.flatten()]

    output_dir = MAIN_ROOT_DIR / 'Strategy_Spatial_LOOCV_Validation'
    upscale_dir, model_dir = output_dir / 'upscaling_results', output_dir / 'models'
    split_conf_dir, spat_conf_dir = upscale_dir / 'split_conformal_prediction', upscale_dir / 'spatial_loocv_conformal_prediction'

    for target_var in TARGETS:
        print(f"\nUpscaling {target_var}...")

        # 1. Bootstrap Uncertainty
        boot_path = model_dir / f'{target_var}_RF_Spatial_LOOCV_model.joblib'
        if boot_path.exists():
            model = joblib.load(boot_path)
            try:
                p_mean, p_std, p_low, p_high = bootstrap_uncertainty_estimation(model, X_pred)
                save_tif(upscale_dir / f'{target_var}_Spatial_LOOCV_pred_mean.tif', p_mean, valid_mask, out_meta)
                save_tif(upscale_dir / f'{target_var}_Spatial_LOOCV_pred_std.tif', p_std, valid_mask, out_meta)
                save_tif(upscale_dir / f'{target_var}_Spatial_LOOCV_95CI_lower.tif', p_low, valid_mask, out_meta)
                save_tif(upscale_dir / f'{target_var}_Spatial_LOOCV_95CI_upper.tif', p_high, valid_mask, out_meta)
            except Exception as e:
                warnings.warn(f"Bootstrap failed for {target_var}: {str(e)}")

        # 2. Split Conformal
        split_conf_path = model_dir / f'{target_var}_RF_split_conformal_model.joblib'
        if split_conf_path.exists():
            art = joblib.load(split_conf_path)
            p_pt, p_low, p_high = conformal_prediction_interval(art['model'], art['conformal_quantile'], X_pred)
            save_tif(split_conf_dir / f'{target_var}_conformal_point_pred.tif', p_pt, valid_mask, out_meta)
            save_tif(split_conf_dir / f'{target_var}_conformal_95PI_lower.tif', p_low, valid_mask, out_meta)
            save_tif(split_conf_dir / f'{target_var}_conformal_95PI_upper.tif', p_high, valid_mask, out_meta)

        # 3. Spatial LOOCV Conformal
        spat_conf_path = model_dir / f'{target_var}_RF_spatial_loocv_conformal_model.joblib'
        if spat_conf_path.exists():
            art = joblib.load(spat_conf_path)
            p_pt, p_low, p_high = conformal_prediction_interval(art['model'], art['conformal_quantile'], X_pred)
            save_tif(spat_conf_dir / f'{target_var}_conformal_point_pred.tif', p_pt, valid_mask, out_meta)
            save_tif(spat_conf_dir / f'{target_var}_conformal_95PI_lower.tif', p_low, valid_mask, out_meta)
            save_tif(spat_conf_dir / f'{target_var}_conformal_95PI_upper.tif', p_high, valid_mask, out_meta)


def main():
    """Main workflow: data preparation → Spatial LOOCV modeling → spatial upscaling (bootstrap + dual conformal)"""
    MAIN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print_separator(f"Starting RF Upscaling Workflow (Spatial LOOCV 100km) - Results saved to: {MAIN_ROOT_DIR}")

    check_file_exists(EXCEL_FILE, "Training Excel dataset")
    df = pd.read_excel(EXCEL_FILE)

    required_cols = FEATURES + TARGETS + ['Site', 'LON', 'LAT']
    if any(col not in df.columns for col in required_cols):
        raise ValueError("Missing required columns in dataset.")

    # Data preprocessing
    df_imputed = df.copy()
    for site in df['Site'].unique():
        mask = df_imputed['Site'] == site
        for col in FEATURES + TARGETS:
            if df_imputed[col].dtype in [np.float64, np.int64]:
                site_mean = df_imputed.loc[mask, col].mean()
                if not pd.isna(site_mean):
                    df_imputed.loc[mask & df_imputed[col].isna(), col] = site_mean

    df_clean = df_imputed.dropna(subset=FEATURES + TARGETS)
    output_dir = MAIN_ROOT_DIR / 'Strategy_Spatial_LOOCV_Validation'

    train_spatial_loso_model(df_clean, output_dir)
    upscale_spatial_loso_model()
    print_separator("Spatial LOOCV Workflow completed successfully!")


if __name__ == '__main__':
    main()
