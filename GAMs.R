# ==============================================================================
# GAM Modeling Pipeline for LUE Analysis
# ==============================================================================
# Author: Yong Lin
# Affiliation: Institute of Geographic Sciences and Natural Resources Research, CAS
# Contact: linyong0018@igsnrr.ac.cn
# R Version: 4.5.2 
# ==============================================================================
# Core Functions: 
# 1. Fit GAM models with Gamma distribution and log link function
# 2. Export model summaries (formula/smooth stats/diagnostics) to TXT file
# ==============================================================================

# ---------------------------
# 1. Load Required Libraries
# ---------------------------
library(mgcv)      
library(gratia)    
library(dplyr)     
library(tidyr)     
library(readxl)    

# ---------------------------
# 2. Global Parameter Settings
# ---------------------------
### 2.1 File Paths 
data_path <- "/Example data_LUE.xlsx"  # Input data path
output_dir <- "/GAM_result"  # Output directory for results

### 2.2 Variable Definitions
dependent_vars <- c('LUEmax', 'LUEinc', 'LUEact')  # Response variables 
predictor_vars <- c('TA_GS', 'FD_GS', 'VPD_GS', 'LAI_GS','P_GS', 'GSL', 'CHL','CNR','SLA','LN','LP','SOC','SP','CI')  # Predictor variables

### 2.3 Export Parameters
txt_file_name <- "GAM_Model_Results_Summary.txt"  # Name of model result TXT file

# ---------------------------
# 3. Data Preparation
# ---------------------------
### 3.1 Create output directory if not exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

### 3.2 Read input Excel data and remove missing values
df_raw <- read_excel(data_path)
df <- df_raw %>% 
  drop_na(all_of(c(dependent_vars, predictor_vars)))

# ---------------------------
# 4. GAM Model Fitting
# ---------------------------
# Initialize list to store model results
model_results <- list()

# Fit GAM models for each dependent variable
for (dep_var in dependent_vars) {
  # Step 1: Build GAM formula (smooth terms for all predictors)
  gam_formula <- as.formula(
    paste(dep_var, "~", paste0("s(", predictor_vars, ")", collapse = " + "))
  )
  
  # Step 2: Fit GAM model (REML estimation, Gamma distribution + log link)
  gam_model <- gam(
    formula = gam_formula,
    data = df,
    method = "REML",
    family = Gamma(link = "log")
  )
  
  # Step 3: Extract smooth terms from the model
  smooth_terms <- smooths(gam_model)
  
  # Step 4: Store core model results
  model_results[[dep_var]] <- list(
    model = gam_model,
    summary = summary(gam_model),
    smooth_terms = smooth_terms,
    formula = gam_formula
  )
}

# ---------------------------
# 5. Export Model Results to TXT File
# ---------------------------
### 5.1 Define TXT file path
txt_path <- file.path(output_dir, txt_file_name)

### 5.2 Write model results to TXT (use sink() to capture all output)
sink(txt_path)

cat("=============================================\n")
cat("GAM Model Results Summary\n")
cat("Analysis Date: ", Sys.time(), "\n")
cat("Raw Data Path: ", data_path, "\n")
cat("Model Dataset: ", nrow(df), " rows (no missing values) | ", length(predictor_vars), " predictors\n")
cat("Missing Rows Removed: ", nrow(df_raw)-nrow(df), "\n")
cat("Model Settings: REML estimation | Gamma distribution | log link function\n")
cat("=============================================\n\n")

# Write results for each response variable
for (dep_var in dependent_vars) {
  cat("====================================================================\n")
  cat("MODEL FOR RESPONSE VARIABLE: ", dep_var, "\n")
  cat("====================================================================\n\n")
  
  # 5.2.1 Model Formula
  cat("1. Model Formula:\n")
  print(model_results[[dep_var]]$formula)
  cat("\n")
  
  # 5.2.2 Model Summary (full output)
  cat("2. Model Summary (REML, Gamma + log link):\n")
  print(model_results[[dep_var]]$summary)
  cat("\n")
  
  # 5.2.3 Smooth Term Key Statistics (simplified)
  cat("3. Smooth Term Key Statistics:\n")
  smooth_table <- as.data.frame(model_results[[dep_var]]$summary$s.table)
  rownames(smooth_table) <- gsub("s\\(|\\)", "", rownames(smooth_table))
  smooth_table <- smooth_table %>% 
    mutate(
      p_value = round(`p-value`, 4),
      F_value = round(F, 2),
      EDF = round(edf, 2)
    ) %>% 
    select(EDF, F_value, p_value)
  print(smooth_table)
  cat("\n")
  
  # 5.2.4 Model Diagnostics (deviance/df/residuals)
  cat("4. Model Diagnostics:\n")
  gam_model <- model_results[[dep_var]]$model
  cat("   - Residual Deviance: ", round(gam_model$deviance, 2), "\n", sep = "")
  cat("   - Null Deviance:     ", round(gam_model$null.deviance, 2), "\n", sep = "")
  cat("   - Residual df:       ", gam_model$df.residual, "\n", sep = "")
  cat("   - Null df:           ", gam_model$df.null, "\n", sep = "")
  cat("   - Generalized R²:    ", round(model_results[[dep_var]]$summary$r.sq, 4), "\n", sep = "")
  cat("   - REML Score:        ", round(gam_model$gcv.ubre, 4), "\n", sep = "")
  cat("\n\n")
}

# Close sink (stop redirecting output)
sink()