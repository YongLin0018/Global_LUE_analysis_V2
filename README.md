# Global_LUE_analysis_V2
This repository contains core datasets and code scripts for ecosystem Light Use Efficiency (LUE) analysis at site scale and global upscaling, mainly including site-scale data, R script for Generalized Additive Models, Python scripts for Linear Mixed-Effects Models and Random Forest-based LUE mapping. 

1.Example data_LUE.xlsx: site-scale dataset, which contains LUE measurements and matched bioclimatic, ecological explanatory variables. 

2.GAMs.R: R script dedicated to Generalized Additive Models analysis, exploring the nonlinear responses of LUE to various explanatory variables. 

3.LMMs.py: Python script for Linear Mixed-Effects Models calculation, it quantifies the effects of drivers on LUE. 

4.Mapping-LUE_RF.py: Python script for Random Forest modeling and LUE global upscaling, it first trains an RF model with site-scale data then conducts Buffered Leave-One-Out Cross-Validation (B-LOO CV) to test model generalization ability, and further upscales the optimized model to global gridded datasets to generate global LUE maps. 

Author: Yong Lin

Affiliation: Institute of Geographic Sciences and Natural Resources Research, CAS

Contact: linyong0018@igsnrr.ac.cn 
