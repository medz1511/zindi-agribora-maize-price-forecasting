![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Zindi](https://img.shields.io/badge/Zindi-Top%2010%20Rank-green)
# AgriBORA Commodity Price Forecasting Challenge - 10th Place Solution

**Username:** Medz1511
**Submission ID:** 4aJsZ6uu
**Final Filename:** submission_final_hybrid.csv

## 1. Solution Overview
My solution implements a robust hybrid approach specifically designed to handle the volatility of recent weeks (weeks 50-52).
Instead of relying solely on a machine learning model, I combined it with a naive trend estimator to better capture short-term market movements.

### Key Methodology:
1.  **Imputation:** Missing KAMIS data is imputed using geographical neighbors (Haversine distance).
2.  **Recent Data Integration:** The model integrates the latest weekly updates provided during the competition (up to week 51).
3.  **Hybrid Prediction Logic (The "Secret Sauce"):**
    * **60%** Histogram Gradient Boosting Regressor (trained with heavy weights on 2024 data).
    * **40%** Naive Prediction (Last observed price + recent trend).
4.  **Post-Processing:**
    * **Bias Correction:** I calculate the median residual on the validation set for each county and apply 70% of this bias to correct the final prediction.
    * **Constraints:** Predictions are clipped within ±10% of the last observed price to prevent unrealistic spikes.

## 2. Setup & Requirements
- **OS:** Windows / Linux / MacOS
- **Python version:** 3.9+
- **Hardware:** Standard CPU (No GPU required), < 1GB RAM.

## 3. Data Setup
Please ensure the following files are placed in the input/ directory:
kamis_maize_prices.csv

* agribora_maize_prices.csv

* agriBORA_maize_prices_weeks_46_47.csv

* agriBORA_maize_prices_weeks_46_47_48.csv

* agriBORA_maize_prices_weeks_46_to_49.csv

* agriBORA_maize_prices_weeks_46_to_51.csv

## 4. How to Run the Code

1.   Navigate to the root of the folder.raphical 
2.  **Run the script:** python src/main.py
3.  The final submission file **submission_final_hybrid.csv** will be generated in the root or output/ folder (depending on execution context).








# AgriBORA Commodity Price Forecasting - 10th Place Solution
**Author:** Medz1511
**Submission ID:** 4aJsZ6uu
**Private Leaderboard Score:** 0.3554

---

## 1. Overview and Objectives
This solution aims to forecast the average weekly wholesale prices of maize in five Kenyan counties (Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu). 

The primary challenge was dealing with high volatility in recent weeks (Weeks 50-52). A standard Machine Learning model often lagged behind sharp price corrections. Therefore, the objective was to build a **Hybrid System** that combines the stability of a Gradient Boosting model with the reactivity of a Naive Trend Estimator.

**Expected Outcome:** A robust forecasting pipeline that minimizes RMSE and MAE by adapting quickly to recent market shifts while correcting systematic biases per county.

## 2. Architecture Diagram
The data flow is a linear pipeline designed for rapid inference on a local CPU.

```text
[Raw CSV Data] 
       |
       v
[ETL & Preprocessing] --> (Imputation, Date Parsing, Normalization)
       |
       v
[Feature Engineering] --> (Lags, Trends, Rolling Means, Kamis Ratio)
       |
       v
[Hybrid Modeling] 
   |-- Branch A: HistGradientBoostingRegressor (Scikit-Learn)
   |-- Branch B: Naive Trend Estimator (Last Value + Trend)
       |
       v
[Ensemble & Post-Processing] --> (Weighted Avg + Bias Correction + Clipping)
       |
       v
[Final CSV Submission]

```
## 3. ETL Process (Extract, Transform, Load)

### Extract
Data is extracted from the CSV files provided by Zindi.
- **Sources:** `kamis_maize_prices.csv` (historical external data) and `agribora_maize_prices.csv` (target data).
- **Updates:** The solution specifically integrates the latest available updates: `agriBORA_maize_prices_weeks_46_to_51.csv`.

### Transform
- **Normalization:** County names are stripped of whitespace. Dates are converted to `Period('W')` to align weekly data.
- **Kamis Imputation:** Missing values in the external KAMIS dataset are imputed using **Geographic Neighbor Imputation** (based on Haversine distance) followed by temporal interpolation.
- **Merging:** AgriBORA prices are merged with KAMIS smoothed prices on `[County, Week]`.

### Load
The transformed data is loaded into a Pandas DataFrame (`df_full`) which serves as the single source of truth for both training and inference.

## 4. Data Modeling

### Feature Engineering
We focused on a minimal but effective feature set to avoid overfitting on the small dataset:
- **Lags:** `lag_1` (1 week ago), `lag_2` (2 weeks ago).
- **Trend:** `trend` (Difference between `lag_1` and `lag_2`).
- **Rolling Stats:** `roll_mean_3` (3-week moving average).
- **External:** `kamis_smooth` (Smoothed price from the Ministry of Agriculture).
- **Encoding:** One-Hot Encoding for counties.

### Model Details
- **Algorithm:** `HistGradientBoostingRegressor` (Scikit-Learn).
- **Why this model?** It handles missing values natively (NaNs) and is robust to outliers.
- **Hyperparameters:** - `max_iter=80`, `max_depth=3`, `learning_rate=0.03` (Conservative settings to prevent overfitting).
  - `l2_regularization=3.0` (High regularization for stability).
- **Training Strategy:** The model is trained on data up to 2024, with sample weights doubling the importance of 2024 data (Weight=2.0) to prioritize recent market behavior.

### Validation
We used a temporal split (**Train** < Oct 2024, **Validation** > Oct 2024) to estimate performance and calculate county-specific biases.

## 5. Inference Strategy (The Hybrid Approach)

The model output is not used raw. It passes through a robust inference layer:

1.  **Prediction A (ML):** Output from `HistGradientBoosting`.
2.  **Prediction B (Naive):** $Price_{t-1} + (Trend_{recent} \times 0.3)$.
3.  **Weighted Ensemble:** $$FinalPrediction = (0.6 \times Pred_A) + (0.4 \times Pred_B)$$
4.  **Rationale:** The Naive component helps capture immediate "shocks" that the ML model smooths out.
5.  **Bias Correction:** We add 70% of the median residual observed on the validation set for each county.
6.  **Safety Clipping:** Predictions are constrained to be within ±10% of the last observed price (`lag_1`) to ensure economic realism.

## 6. Run Time
- **Total Execution Time:** < 10 seconds.
- **Hardware:** Standard Laptop CPU (Intel i5/i7 equivalent). No GPU required.

## 7. Performance Metrics
- **Metric:** Weighted average of MAE (50%) and RMSE (50%).
- **Public Leaderboard:** 0.186342719
- **Private Leaderboard:**0.774627287 (Rank 10)


## 8. Error Handling and Logging
- **Input Checks:** The script checks for the existence of input files in the `input/` directory and falls back to root directory if not found.
- **Fallback Prediction:** If a county has insufficient historical data to generate features (less than 2 weeks), the system falls back to a safe default value (40.0) or the last known value, preventing the pipeline from crashing.

## 9. Maintenance and Monitoring
- **Updating:** To update the model for future weeks, simply add the new `agriBORA...csv` file to the `input/` folder and update the `recent_files` list in `src/main.py`.
- **Scaling:** The code is vectorized using Pandas and Scikit-Learn. It can easily handle 10x more data without performance degradation.
- **Retraining:** Retraining is recommended weekly as new price points become available to keep the `recent_trends` accurate.

## 10. Conclusion
This solution prioritizes stability and recent trend adaptation over complex deep learning. By acknowledging that agricultural prices have strong serial correlation and momentum, the hybrid approach outperformed pure ML solutions during the volatile final weeks of 2025.


### 2. Organize Folder
Place all downloaded files into the `input/` directory. Your project structure should look like this:

```text
agribora-maize-price-forecasting/
├── input/
│   ├── kamis_maize_prices.csv
│   ├── agribora_maize_prices.csv
│   └── ... (other CSV files)
├── output/
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

## License

[MIT](https://choosealicense.com/licenses/mit/)