# Homework 6: Tree-Based Methods, Class Imbalance and SMOTE (DSCI 552)

**Author:** Mohammad Ayaan Khan (GitHub: MohammadAyaanKhan)

## Contents
- `notebook/Khan_MohammadAyaan_HW6.ipynb`: the executed notebook with all code, outputs, plots and written answers, including the ISLR exercises 6.6.3, 6.6.5, 8.4.5 and 9.7.3.
- `data/`: the UCI data set used.
  - `aps_failure_training_set.csv.gz`, `aps_failure_test_set.csv.gz`, `aps_failure_description.txt`: APS Failure at Scania Trucks (the CSV files are gzip-compressed to stay under GitHub's 25 MB upload limit; pandas reads them directly)

The notebook reads the data from `../data/` and downloads it from UCI automatically if the files are missing.

## Summary
1. **Data preparation (APS Failure).** 99% of the training rows have at least one missing value, so missing values are filled with the median (the training set with training medians and the test set with test medians). The data set is highly imbalanced: 59,000 negative and 1,000 positive trucks in the training set.
2. **Random forests.** Without compensation: test AUC 0.9944, test error 0.75%, 274 of 375 failing trucks caught (OOB error 0.59%). A balanced random forest catches 369 of 375 failing trucks, at the cost of more false alarms (test error 5.19%), and lowers the total misclassification cost from 50,690 to 11,250.
3. **XGBoost with an L1 penalty.** Without compensation (α = 10 by 5-fold CV): test AUC 0.9961, test error 0.81%, 271 of 375 failing trucks caught. With SMOTE applied inside the cross-validation folds (α = 0.01): test error 1.24%, 334 of 375 failing trucks caught, total cost 22,070 instead of 52,260.
4. **ISLR exercises.** 6.6.3 and 6.6.5 (lasso and ridge), 8.4.5 (majority vote vs average probability in bagging) and 9.7.3 (maximal margin classifier), answered in markdown at the end of the notebook.

## Running
Requires Python 3 with numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn and xgboost (`pip install -r requirements.txt`). Open the notebook from the `notebook/` folder and run all cells. A full run takes about 10 to 20 minutes, mostly for SMOTE and the cross-validated XGBoost searches.
