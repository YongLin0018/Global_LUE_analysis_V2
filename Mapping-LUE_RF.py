"""
Random Forest Regression for LUE Upscaling with Leave-One-Site-Out Cross Validation (LOOCV)
==========================================================================================
# Author: Yong Lin
# Affiliation: Institute of Geographic Sciences and Natural Resources Research, CAS
# Contact: linyong0018@igsnrr.ac.cn
# Python Version: 3.11.4
Description:
    This script implements a Random Forest regression workflow for upscaling three ecosystem Light Use Efficiency (LUE) metrics
    (LUEmax, LUEinc, LUEact) with Leave-One-Site-Out Cross Validation (LOOCV). The core workflow includes:
    1. Data preprocessing (missing value imputation, input validation)
    2. LOOCV model training and evaluation (R², RMSE metrics)
    3. SHAP feature importance analysis for model interpretability
    4. Spatial upscaling via raster data with bootstrap uncertainty estimation
    5. Export of prediction results (mean, std, 95% CI) to GeoTIFF format
Dependencies:
    pandas, numpy, matplotlib, scikit-learn, shap, rasterio, tqdm, joblib, pathlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
from pathlib import Path
import warnings
import joblib
import rasterio
from tqdm import tqdm

# ==============================================================================
# Core Configuration (LOOCV Validation + Upscaling)
# ==============================================================================
# Target LUE metrics for regression
TARGETS = ['LUEmax', 'LUEinc', 'LUEact']
# Predictor variables (matched with raster file names)
FEATURES = ['TA_GS', 'FD_GS', 'VPD_GS', 'P_GS', 'LAI_GS',
            'GSL', 'CHL', 'SLA', 'LN', 'LP', 'CNR', 'SOC', 'SP', 'CI']
# Modeling hyperparameters
MIN_SAMPLES = 20  # Minimum samples required for model training
RANDOM_STATE = 42  # Fixed seed for reproducibility
N_BOOTSTRAP = 100  # Bootstrap iterations for uncertainty estimation
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
MAIN_ROOT_DIR = Path.home() / 'Desktop' / 'RF_Upscaling_LOOCV_Validation'

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

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path
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

    Raises
    ------
    ValueError
        If raster dimensions mismatch across features
    FileNotFoundError
        If reference raster file is missing
    """
    if not reference_feat:
        reference_feat = list(raster_paths.keys())[0]
    check_file_exists(raster_paths[reference_feat], f"Reference raster ({reference_feat})")

    with rasterio.open(raster_paths[reference_feat]) as src:
        ref_dims = src.shape
        ref_meta = src.meta.copy()

    # Validate all rasters against reference
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
    length : int, optional
        Total length of the separator line (default: 80)
    """
    print(f"\n{'=' * length}\n{msg}\n{'=' * length}")

# ==============================================================================
# Core Modeling Functions
# ==============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target_var: str,
                      site: str = None) -> dict:
    """
    Calculate regression metrics (R², RMSE) for model evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Model predictions
    model_name : str
        Name of the model being evaluated
    target_var : str
        Target variable name
    site : str, optional
        Left-out site for LOOCV validation (default: None)

    Returns
    -------
    dict
        Dictionary containing calculated metrics and metadata
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {
        'target_variable': target_var,
        'model': model_name,
        'R2': r2,
        'RMSE': rmse
    }
    if site:
        metrics['left_out_site'] = site
    return metrics


def calculate_cv_metrics(fold_metrics: list) -> tuple:
    """
    Calculate global cross-validation metrics from LOOCV fold results.

    Parameters
    ----------
    fold_metrics : list
        List of metric dictionaries from each LOOCV fold

    Returns
    -------
    tuple
        (average metrics DataFrame, detailed fold metrics DataFrame)
    """
    cv_df = pd.DataFrame(fold_metrics)

    # Global metrics for LOOCV
    y_true_all = np.concatenate(cv_df['y_true'].values)
    y_pred_all = np.concatenate(cv_df['y_pred'].values)
    global_r2 = r2_score(y_true_all, y_pred_all)
    global_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    # Format results
    avg_metrics = pd.DataFrame([{
        'target_variable': cv_df['target_variable'].iloc[0],
        'model': model,
        'mean_R2': round(global_r2, 3),
        'std_R2': 0.0,  # No variance for LOOCV global metric
        'mean_RMSE': round(global_rmse, 3),
        'std_RMSE': 0.0
    } for model in cv_df['model'].unique()])

    return avg_metrics, cv_df


def run_leave_one_site_out(X: pd.DataFrame, y: pd.Series, sites: np.ndarray, target_var: str) -> list:
    """
    Perform Leave-One-Site-Out (LOOCV) cross-validation with Random Forest.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable vector
    sites : np.ndarray
        Site labels for grouping
    target_var : str
        Name of the target variable

    Returns
    -------
    list
        List of dictionaries containing fold-wise results

    Raises
    ------
    ValueError
        If fewer than 2 valid sites exist for cross-validation
    RuntimeError
        If no valid results are generated from LOOCV
    """
    loso_results = []
    logo = LeaveOneGroupOut()
    valid_sites = np.unique(sites)

    if len(valid_sites) < 2:
        raise ValueError(f"Insufficient valid sites ({len(valid_sites)}) for LOOCV validation")

    print(f"Performing LOOCV validation: {len(valid_sites)} valid sites")

    # Initialize base model
    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups=sites), 1):
        leave_site = np.unique(sites[val_idx])[0]

        # Prepare training/validation data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Skip if training data is insufficient
        if len(X_train) < 2:
            warnings.warn(f"Skipping site {leave_site}: Insufficient training samples ({len(X_train)})")
            continue

        # Train model and make predictions
        model = base_model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Store results
        loso_results.append({
            'target_variable': target_var,
            'model': 'RandomForest',
            'left_out_site': leave_site,
            'y_true': y_val.values,
            'y_pred': y_pred
        })

    if not loso_results:
        raise RuntimeError("No valid results generated from LOOCV validation")

    return loso_results


def save_cv_results(output_dir: Path, X: pd.DataFrame, y: pd.Series, target_var: str, fold_metrics: list,
                    cv_avg_metrics: pd.DataFrame) -> tuple:
    """
    Save LOOCV validation results (metrics, plots, SHAP analysis).

    Parameters
    ----------
    output_dir : Path
        Directory to save results
    X : pd.DataFrame
        Full feature matrix for SHAP analysis
    y : pd.Series
        Full target vector for model training
    target_var : str
        Target variable name
    fold_metrics : list
        Detailed fold metrics from LOOCV
    cv_avg_metrics : pd.DataFrame
        Average cross-validation metrics

    Returns
    -------
    tuple
        (detailed fold metrics DataFrame, average metrics DataFrame)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    suffix = "LOOCV_Validation"

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

        # Calculate and save SHAP importance
        shap_importance = pd.DataFrame({
            'feature': FEATURES,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=True)

        # Plot SHAP importance
        plt.figure(figsize=(10, 8))
        plt.barh(shap_importance['feature'], shap_importance['shap_importance'], color='#3C5E20')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'{target_var} - Random Forest Feature Importance (SHAP, {suffix})')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{target_var}_RF_SHAP_importance_{suffix}.png', dpi=300, bbox_inches='tight')

        # Save SHAP values with ranking
        shap_importance['rank'] = range(1, len(shap_importance) + 1)
        shap_importance = shap_importance[['rank', 'feature', 'shap_importance']]
        shap_importance.to_csv(output_dir / f'{target_var}_RF_SHAP_importance_ranking_{suffix}.csv', index=False)

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
        Trained Random Forest model with stored training data (X_train_, y_train_)
    X : np.ndarray
        Feature matrix for prediction

    Returns
    -------
    tuple
        (mean prediction, standard deviation, 2.5th percentile, 97.5th percentile)

    Raises
    ------
    AttributeError
        If model lacks stored training data
    ValueError
        If training sample size is insufficient for bootstrap
    """
    # Check for stored training data
    if not (hasattr(model, 'X_train_') and hasattr(model, 'y_train_')):
        raise AttributeError("Model must have X_train_ and y_train_ attributes for bootstrap")

    X_train, y_train = model.X_train_, model.y_train_
    if len(X_train) < 10:
        raise ValueError(f"Insufficient training samples ({len(X_train)}) for bootstrap")

    # Clean prediction data (impute missing values with training mean)
    X_clean = X.copy()
    if np.isnan(X_clean).any():
        train_mean = np.nanmean(X_train, axis=0)
        X_clean[np.isnan(X_clean)] = np.take(train_mean, np.where(np.isnan(X_clean))[1])

    # Bootstrap iterations
    bootstrap_preds = []
    for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrap iterations"):
        # Resample training data
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot, y_boot = X_train.iloc[idx], y_train.iloc[idx]

        # Train bootstrap model
        boot_model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE + i,
            n_jobs=-1
        ).fit(X_boot, y_boot)

        bootstrap_preds.append(boot_model.predict(X_clean))

    # Calculate uncertainty metrics
    bootstrap_preds = np.array(bootstrap_preds)
    return (
        np.mean(bootstrap_preds, axis=0),
        np.std(bootstrap_preds, axis=0),
        np.percentile(bootstrap_preds, 2.5, axis=0),
        np.percentile(bootstrap_preds, 97.5, axis=0)
    )


def load_feature_raster(features: list) -> tuple:
    """
    Load and validate raster data for predictor variables.

    Parameters
    ----------
    features : list
        List of features to load from raster files

    Returns
    -------
    tuple
        (flattened feature matrix, valid pixel mask, raster metadata)

    Raises
    ------
    ValueError
        If some features lack corresponding raster files
    """
    print_separator("Loading raster feature data")

    # Validate feature-raster mapping
    missing_feats = set(features) - set(RASTER_PATHS.keys())
    if missing_feats:
        raise ValueError(f"No raster files found for features: {missing_feats}")

    # Load reference raster metadata
    ref_feat = features[0]
    with rasterio.open(RASTER_PATHS[ref_feat]) as src:
        ref_meta = src.meta.copy()
        valid_mask = np.ones(src.shape, dtype=bool)

    # Load all features and create valid mask
    data_arrays = []
    for feat in features:
        with rasterio.open(RASTER_PATHS[feat]) as src:
            band_data = src.read(1).astype(np.float32)

            # Mask invalid values (NaN + NoData)
            nodata_mask = np.isnan(band_data) | (band_data == src.nodata) if src.nodata else np.isnan(band_data)
            band_data[nodata_mask] = np.nan
            valid_mask &= ~nodata_mask

            data_arrays.append(band_data.flatten())

    # Combine features into matrix
    X_full = np.vstack(data_arrays).T
    valid_pixels = np.sum(valid_mask)
    print(f"Raster loading complete: {valid_pixels} valid pixels out of {valid_mask.size} total")

    if valid_pixels < 100:
        warnings.warn(f"Very few valid pixels ({valid_pixels}) - upscaling results may be unreliable")

    return X_full, valid_mask, ref_meta


def save_tif(output_path: Path, data: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    """
    Save prediction results to GeoTIFF format with georeferencing.

    Parameters
    ----------
    output_path : Path
        Output file path for the GeoTIFF
    data : np.ndarray
        Prediction data (flattened for valid pixels)
    mask : np.ndarray
        Boolean mask of valid pixels (2D raster shape)
    meta : dict
        Raster metadata for georeferencing

    Raises
    ------
    ValueError
        If data length does not match number of valid pixels
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    height, width = mask.shape
    out_array = np.full((height, width), np.nan, dtype=np.float32)

    # Validate data dimensions
    valid_indices = np.where(mask.flatten())[0]
    if len(valid_indices) != len(data):
        raise ValueError(f"Data length mismatch: {len(data)} vs {len(valid_indices)} valid pixels")

    # Fill valid pixels
    out_array.flat[valid_indices] = data

    # Update metadata for output
    out_meta = meta.copy()
    out_meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan,
        'compress': 'lzw'  # Lossless compression for scientific data
    })

    # Write to GeoTIFF
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_array, 1)
    print(f"Raster saved successfully: {output_path}")

# ==============================================================================
# Main Workflow Functions
# ==============================================================================
def train_loso_model(df_clean: pd.DataFrame, output_dir: Path) -> None:
    """
    Train Random Forest models with LOOCV and save results.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned training dataset with features, targets and site labels
    output_dir : Path
        Directory to save model results and artifacts
    """
    print_separator("Starting LOOCV Validation Modeling")
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid sites (at least 1 sample per site)
    valid_sites = df_clean['Site'].value_counts()[df_clean['Site'].value_counts() >= 1].index
    df_loso = df_clean[df_clean['Site'].isin(valid_sites)].copy()

    if len(valid_sites) < 2:
        raise ValueError(f"Insufficient valid sites ({len(valid_sites)}) for LOOCV validation")

    # Train model for each target variable
    for target_var in TARGETS:
        print(f"\nProcessing target variable: {target_var}")

        # Skip if insufficient valid samples
        if df_loso[target_var].notna().sum() < MIN_SAMPLES:
            warnings.warn(f"Skipping {target_var}: Insufficient valid samples")
            continue

        # Prepare feature matrix and target vector
        X = df_loso[FEATURES]
        y = df_loso[target_var]
        sites = df_loso['Site'].values

        # Run LOOCV validation
        loso_results = run_leave_one_site_out(X, y, sites, target_var)

        # Calculate metrics
        avg_metrics, fold_metrics = calculate_cv_metrics(loso_results)

        # Save results and visualizations
        save_cv_results(output_dir, X, y, target_var, loso_results, avg_metrics)

        # Save final model for upscaling
        final_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X, y)
        final_model.X_train_ = X  # Store training data for bootstrap
        final_model.y_train_ = y
        joblib.dump(final_model, model_dir / f'{target_var}_RF_LOOCV_model.joblib')


def upscale_loso_model() -> None:
    """
    Upscale LOOCV-validated models to spatial raster predictions with uncertainty estimation.
    """
    print_separator("Starting Spatial Upscaling with LOOCV Models")

    # Validate raster data and load features
    validate_raster_size(RASTER_PATHS)
    X_full, valid_mask, out_meta = load_feature_raster(FEATURES)

    # Extract valid pixels for prediction
    X_pred = X_full[valid_mask.flatten()]
    if len(X_pred) < 100:
        raise ValueError(f"Insufficient valid pixels ({len(X_pred)}) for upscaling")

    # Define output directories
    output_dir = MAIN_ROOT_DIR / 'Strategy_LOOCV_Validation'
    upscale_dir = output_dir / 'upscaling_results'
    upscale_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / 'models'

    # Upscale each target variable
    for target_var in TARGETS:
        print(f"\nUpscaling {target_var}...")
        model_path = model_dir / f'{target_var}_RF_LOOCV_model.joblib'

        # Skip if model file not found
        if not model_path.exists():
            warnings.warn(f"LOOCV model not found for {target_var}: Skipping upscaling")
            continue

        # Load trained model
        model = joblib.load(model_path)

        # Generate predictions with uncertainty
        try:
            pred_mean, pred_std, pred_lower, pred_upper = bootstrap_uncertainty_estimation(model, X_pred)

            # Save uncertainty rasters
            save_tif(upscale_dir / f'{target_var}_LOOCV_pred_mean.tif', pred_mean, valid_mask, out_meta)
            save_tif(upscale_dir / f'{target_var}_LOOCV_pred_std.tif', pred_std, valid_mask, out_meta)
            save_tif(upscale_dir / f'{target_var}_LOOCV_95CI_lower.tif', pred_lower, valid_mask, out_meta)
            save_tif(upscale_dir / f'{target_var}_LOOCV_95CI_upper.tif', pred_upper, valid_mask, out_meta)

        except Exception as e:
            warnings.warn(f"Bootstrap failed for {target_var}: {str(e)} - Generating basic prediction")
            # Fallback to basic prediction without uncertainty
            X_clean = X_pred.copy()
            X_clean[np.isnan(X_clean)] = np.nanmean(model.X_train_, axis=0)
            pred_basic = model.predict(X_clean)
            save_tif(upscale_dir / f'{target_var}_LOOCV_pred_basic.tif', pred_basic, valid_mask, out_meta)

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    """Main workflow: data preparation → LOOCV modeling → spatial upscaling"""
    # Create result directory
    MAIN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print_separator(f"Starting RF Upscaling Workflow - Results saved to: {MAIN_ROOT_DIR}")

    # 1. Data preprocessing
    check_file_exists(EXCEL_FILE, "Training Excel dataset")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Raw dataset shape: {df.shape}")

    # Validate required columns
    required_cols = FEATURES + TARGETS + ['Site', 'LON', 'LAT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Impute missing values by site (mean imputation for numeric features)
    df_imputed = df.copy()
    for site in df['Site'].unique():
        site_mask = df_imputed['Site'] == site
        for col in FEATURES + TARGETS:
            if df_imputed[col].dtype in [np.float64, np.int64]:
                site_mean = df_imputed.loc[site_mask, col].mean()
                if not pd.isna(site_mean):
                    df_imputed.loc[site_mask & df_imputed[col].isna(), col] = site_mean

    # Remove remaining missing values
    df_clean = df_imputed.dropna(subset=FEATURES + TARGETS)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    if len(df_clean) < MIN_SAMPLES:
        raise ValueError(f"Insufficient samples after cleaning ({len(df_clean)}) - Minimum required: {MIN_SAMPLES}")

    print("Site distribution (top 10):")
    print(df_clean['Site'].value_counts().head(10))

    # 2. Run LOOCV modeling
    output_dir = MAIN_ROOT_DIR / 'Strategy_LOOCV_Validation'
    train_loso_model(df_clean, output_dir)

    # 3. Run spatial upscaling
    upscale_loso_model()

    print_separator("Workflow completed successfully! All results saved to the specified directory.")


if __name__ == '__main__':
    main()