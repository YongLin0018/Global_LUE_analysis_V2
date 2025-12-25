"""
Mixed Linear Model (LMM) Analysis for Light Use Efficiency (LUE) Drivers
========================================================================
# Author: Yong Lin
# Affiliation: Institute of Geographic Sciences and Natural Resources Research, CAS
# Contact: linyong0018@igsnrr.ac.cn
# Python Version: 3.11.4
Description:
    This code implements Linear Mixed Model (LMM) for identifying drivers of Light Use Efficiency (LUE) metrics (LUEmax, LUEinc, LUEact)
    including:
    1. Multi-collinearity analysis using Variance Inflation Factor (VIF)
    2. Mixed Linear Model (LMM) fitting with fixed/random effects
    3. Variance decomposition and marginal/conditional R² calculation
    4. Result export to Excel (VIF, model summary, variance) and TXT (model details)
Dependencies:
    pandas, numpy, statsmodels, scikit-learn, warnings, os
"""

# ----------------------
# Import Required Libraries
# ----------------------
import pandas as pd
import numpy as np
import warnings
import os
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# ----------------------
# Global Configuration
# ----------------------
# File paths
DATA_PATH = r"\Example data_LUE.xlsx"
OUTPUT_DIR = r"\LMM_analysis_results\\"

# Analysis parameters
RANDOM_EFFECT = 'Block2'  # Random effect: Block1: Climate; Block2: Vegetation
DEPENDENT_VARS = ['LUEmax', 'LUEinc', 'LUEact']  # LUE metrics (dependent variables)
INDEPENDENT_VARS = [
    'TA_GS', 'FD_GS', 'VPD_GS', 'P_GS',
    'LAI_GS', 'GSL', 'CHL', 'SLA', 'LN', 'LP', 'CNR', 'SOC', 'SP', 'CI'
]  # Predictor variables (independent variables)

# ----------------------
# Variable Group Configuration (for result categorization)
# ----------------------
# Variable group classification (for organizing results by category)
GROUP_CONFIG = {
    'Climate': ['FD', 'TA', 'P', 'VPD', ],
    'Canopy structure': ['LAI', 'SLA', 'CI'],
    'Canopy physiology': ['CHL', 'GSL', 'LN', 'LP'],
    'Nutrient': ['SOC', 'SP', 'CNR']
}

# Build variable-to-group mapping
prefix_to_group = {}
for group, prefixes in GROUP_CONFIG.items():
    for p in prefixes:
        prefix_to_group[p] = group

var_to_group = {}
for var in INDEPENDENT_VARS:
    prefix = var.split('_')[0]
    group = prefix_to_group.get(prefix, 'Other')
    var_to_group[var] = group

# ----------------------
# Core Analysis Functions
# ----------------------
def calc_mixed_r2_and_variance(model_result):
    """
    Calculate marginal/conditional R² and variance decomposition for LMM.

    Parameters:
        model_result: Fitted MixedLM result object from statsmodels

    Returns:
        marginal_r2 (float): R² for fixed effects only
        conditional_r2 (float): R² for fixed + random effects
        variance_components (dict): Variance breakdown (fixed/random/residual)
    """
    y_pred_fixed = model_result.fittedvalues
    var_fixed = np.var(y_pred_fixed)

    # Calculate random effect variance
    try:
        var_random = model_result.cov_re.iloc[0, 0] if not model_result.cov_re.empty else 0
    except:
        var_random = 0

    var_residual = model_result.resid.var()
    var_total = var_fixed + var_random + var_residual

    # R² calculation
    marginal_r2 = var_fixed / var_total if var_total > 0 else np.nan
    conditional_r2 = (var_fixed + var_random) / var_total if var_total > 0 else np.nan

    # Variance decomposition (percentage)
    variance_components = {
        'Fixed_var': var_fixed,
        'Random_var': var_random,
        'Residual_var': var_residual,
        'Total_var': var_total,
        'Fixed_ratio': (var_fixed / var_total * 100) if var_total > 0 else np.nan,
        'Random_ratio': (var_random / var_total * 100) if var_total > 0 else np.nan,
        'Residual_ratio': (var_residual / var_total * 100) if var_total > 0 else np.nan
    }

    return marginal_r2, conditional_r2, variance_components


def calculate_vif(df, variables):
    """
    Calculate Variance Inflation Factor (VIF) to detect multi-collinearity.

    Parameters:
        df (pd.DataFrame): Input dataset with predictor variables
        variables (list): List of independent variables for VIF calculation

    Returns:
        vif_df (pd.DataFrame): VIF results with variable group categorization
    """
    df_vif = df[variables].dropna().copy()

    # Standardize variables for consistent VIF calculation
    scaler = StandardScaler()
    df_vif[variables] = scaler.fit_transform(df_vif[variables])

    # Add constant term required for VIF computation
    X = sm.add_constant(df_vif[variables])

    # Calculate VIF for each predictor
    vif_data = []
    for i in range(1, X.shape[1]):
        var_name = variables[i - 1]
        try:
            vif_value = variance_inflation_factor(X.values, i)
        except:
            vif_value = np.nan

        # Categorize VIF severity (High: >10, Moderate: 5-10, Low: ≤5)
        vif_category = 'High' if vif_value > 10 else ('Moderate' if vif_value > 5 else 'Low')
        vif_data.append({
            'Variable': var_name,
            'Group': var_to_group.get(var_name, 'Other'),
            'VIF': vif_value,
            'VIF_Category': vif_category
        })

    # Sort results by VIF (descending) for easy multi-collinearity identification
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif_df

# ----------------------
# Main Analysis Pipeline
# ----------------------
if __name__ == "__main__":
    # Create output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load and validate input dataset
    print("\n" + "=" * 50)
    print("Loading dataset...")
    print("=" * 50)
    df = pd.read_excel(DATA_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # ----------------------
    # Step 1: Multicollinearity Analysis (VIF)
    # ----------------------
    print("\n" + "=" * 50)
    print("Calculating Variance Inflation Factor (VIF)...")
    print("=" * 50)

    vif_results = calculate_vif(df, INDEPENDENT_VARS)
    print("\nVIF Analysis Results (sorted by VIF descending):")
    print(vif_results.to_string(index=False))

    # Save VIF results to Excel
    vif_results.to_excel(OUTPUT_DIR + "VIF_Analysis_Results.xlsx", index=False)
    vif_dict = dict(zip(vif_results['Variable'], vif_results['VIF']))

    # ----------------------
    # Step 2: Mixed Effects Model Fitting & Analysis
    # ----------------------
    results_summary = []  # Store full model parameter results
    variance_summary = []  # Store variance decomposition results

    for dep_var in DEPENDENT_VARS:
        print(f"\n{'=' * 50}")
        print(f"Analyzing dependent variable: {dep_var}")
        print(f"{'=' * 50}")

        # Prepare model data (remove missing values)
        model_data = df[[dep_var] + INDEPENDENT_VARS + [RANDOM_EFFECT]].dropna().copy()
        print(f"Valid sample size for {dep_var}: {model_data.shape[0]}")

        # Standardize independent variables for model fitting
        scaler = StandardScaler()
        model_data[INDEPENDENT_VARS] = scaler.fit_transform(model_data[INDEPENDENT_VARS])

        # Initialize default values for failed model fits
        mixed_r2_marginal = np.nan
        mixed_r2_conditional = np.nan
        variance_components = {
            'Fixed_var': np.nan, 'Random_var': np.nan, 'Residual_var': np.nan,
            'Total_var': np.nan, 'Fixed_ratio': np.nan, 'Random_ratio': np.nan, 'Residual_ratio': np.nan
        }

        # Fit Mixed Linear Model
        try:
            formula = f"{dep_var} ~ {' + '.join(INDEPENDENT_VARS)}"
            model_mixed = MixedLM.from_formula(formula, groups=model_data[RANDOM_EFFECT], data=model_data)
            result_mixed = model_mixed.fit()

            # Calculate R² and variance decomposition
            mixed_r2_marginal, mixed_r2_conditional, variance_components = calc_mixed_r2_and_variance(result_mixed)

            # Generate and save model summary (TXT)
            mixed_summary = result_mixed.summary().as_text()
            mixed_summary += f"\n\nMarginal R² (fixed effects only): {mixed_r2_marginal:.4f}"
            mixed_summary += f"\nConditional R² (fixed + random effects): {mixed_r2_conditional:.4f}"
            mixed_summary += f"\n\nVariance Decomposition:"
            mixed_summary += f"\nFixed effect variance: {variance_components['Fixed_var']:.4f} ({variance_components['Fixed_ratio']:.1f}%)"
            mixed_summary += f"\nRandom effect variance: {variance_components['Random_var']:.4f} ({variance_components['Random_ratio']:.1f}%)"
            mixed_summary += f"\nResidual variance: {variance_components['Residual_var']:.4f} ({variance_components['Residual_ratio']:.1f}%)"

            print(mixed_summary)
            with open(OUTPUT_DIR + f"{dep_var}_Mixed_Effects_Model_Standardized.txt", 'w', encoding='utf-8') as f:
                f.write(mixed_summary)

            # Extract model parameters (coefficients, SE, p-values)
            mixed_coefs = result_mixed.params.drop(['Intercept', 'Group Var'], errors='ignore')
            mixed_se = result_mixed.bse.drop(['Intercept', 'Group Var'], errors='ignore')
            mixed_pvalues = result_mixed.pvalues.drop(['Intercept', 'Group Var'], errors='ignore')
        except Exception as e:
            print(f"Model fitting failed for {dep_var}: {str(e)}")
            # Assign NaN for failed fits
            mixed_coefs = pd.Series(np.nan, index=INDEPENDENT_VARS)
            mixed_se = pd.Series(np.nan, index=INDEPENDENT_VARS)
            mixed_pvalues = pd.Series(np.nan, index=INDEPENDENT_VARS)

        # Save variance decomposition results
        variance_summary.append({
            'Dependent_Variable': dep_var,
            **variance_components,
            'Marginal_R2': mixed_r2_marginal,
            'Conditional_R2': mixed_r2_conditional
        })

        # Compile full results for Excel export
        for var in INDEPENDENT_VARS:
            results_summary.append({
                'Dependent_Variable': dep_var,
                'Independent_Variable': var,
                'Group': var_to_group[var],
                'VIF': vif_dict.get(var, np.nan),
                'VIF_Category': 'High' if vif_dict.get(var, 0) > 10 else (
                    'Moderate' if vif_dict.get(var, 0) > 5 else 'Low'),
                'Mixed_Effects_Coefficient': mixed_coefs.get(var, np.nan),
                'Mixed_Effects_SE': mixed_se.get(var, np.nan),
                'Mixed_Effects_p_value': mixed_pvalues.get(var, np.nan),
                'Mixed_Effects_Marginal_R²': mixed_r2_marginal,
                'Mixed_Effects_Conditional_R²': mixed_r2_conditional,
                'Random_Effect_Variance': variance_components['Random_var'],
                'Random_Effect_Contribution_Ratio': variance_components['Random_ratio']
            })

    # ----------------------
    # Step 3: Export Results to Excel
    # ----------------------
    # Full model parameter summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_excel(OUTPUT_DIR + "Mixed_Effects_Model_Analysis_Summary_Standardized.xlsx", index=False)

    # Variance decomposition summary
    variance_df = pd.DataFrame(variance_summary)
    variance_df.to_excel(OUTPUT_DIR + "Variance_Decomposition_Summary.xlsx", index=False)

    # ----------------------
    # Completion Message
    # ----------------------
    print(f"\nAnalysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("- VIF_Analysis_Results.xlsx (VIF values and multicollinearity categorization)")
    print("- Variance_Decomposition_Summary.xlsx (fixed/random/residual variance breakdown)")
    for dep_var in DEPENDENT_VARS:
        print(f"- {dep_var}_Mixed_Effects_Model_Standardized.txt (detailed LMM summary with R²)")
    print("- Mixed_Effects_Model_Analysis_Summary_Standardized.xlsx (full model parameter results)")