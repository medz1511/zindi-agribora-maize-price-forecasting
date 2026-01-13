import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Define relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
COORDS = {
    "Kiambu": (-1.1714, 36.8356),
    "Kirinyaga": (-0.5000, 37.2750),
    "Mombasa": (-4.0435, 39.6682),
    "Nairobi": (-1.2921, 36.8219),
    "Uasin-Gishu": (0.5143, 35.2698)
}

# ==============================================================================
# UTILS
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """Calculate geographic distance between two points."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_nearest_county(county, coords_dict):
    """Find the nearest neighbor county."""
    curr_lat, curr_lon = coords_dict[county]
    distances = []
    for other, (lat, lon) in coords_dict.items():
        if other != county:
            dist = haversine(curr_lat, curr_lon, lat, lon)
            distances.append((dist, other))
    return min(distances)[1]

def norm_county(s):
    return s.strip() if isinstance(s, str) else s

# ==============================================================================
# DATA LOADING
# ==============================================================================
try:
    # Read from input directory
    kamis_df = pd.read_csv(os.path.join(INPUT_DIR, 'kamis_maize_prices.csv'), parse_dates=['Date'])
    agri_df = pd.read_csv(os.path.join(INPUT_DIR, 'agribora_maize_prices.csv'), parse_dates=['Date'])
except FileNotFoundError:
    # Fallback to local read
    kamis_df = pd.read_csv('kamis_maize_prices.csv', parse_dates=['Date'])
    agri_df = pd.read_csv('agribora_maize_prices.csv', parse_dates=['Date'])

# Filter for White Maize
kamis_df = kamis_df[kamis_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()
agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()

# Normalize counties and dates
for df in [kamis_df, agri_df]:
    df["county_norm"] = df["County"].apply(norm_county)
    df["week_start"] = df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    
kamis_df = kamis_df[kamis_df["county_norm"].isin(TARGET_COUNTIES)]
agri_df = agri_df[agri_df["county_norm"].isin(TARGET_COUNTIES)]

kamis_df['price'] = pd.to_numeric(kamis_df['Wholesale'], errors='coerce')
agri_df['price'] = pd.to_numeric(agri_df['WholeSale'], errors='coerce')

# Weekly aggregation
kamis_week = kamis_df.groupby(['county_norm', 'week_start'], as_index=False)['price'].mean().rename(columns={'price': 'kamis_price'})
agri_week = agri_df.groupby(['county_norm', 'week_start'], as_index=False)['price'].mean().rename(columns={'price': 'agri_price'})

# ==============================================================================
# KAMIS DATA IMPUTATION (Gap filling)
# ==============================================================================
full_dates = pd.date_range(start='2021-01-01', end='2026-01-31', freq='W-MON')
kamis_filled_list = []

for county in TARGET_COUNTIES:
    c_data = kamis_week[kamis_week['county_norm'] == county].set_index('week_start')
    c_full = pd.DataFrame(index=full_dates).join(c_data['kamis_price'])
    
    # If too many missing values, use geographic neighbor
    if c_full['kamis_price'].isnull().sum() > len(c_full) * 0.3:
        nearest = get_nearest_county(county, COORDS)
        neighbor_data = kamis_week[kamis_week['county_norm'] == nearest].set_index('week_start')
        
        common_idx = c_full['kamis_price'].notna() & neighbor_data['kamis_price'].notna()
        if common_idx.sum() > 10:
            ratio = (c_full.loc[common_idx, 'kamis_price'] / neighbor_data.loc[common_idx, 'kamis_price']).median()
        else:
            ratio = 1.0
        
        c_full['kamis_price'] = c_full['kamis_price'].fillna(neighbor_data['kamis_price'] * ratio)
    
    # Final interpolation and smoothing
    c_full['kamis_price'] = c_full['kamis_price'].interpolate(method='time').ffill().bfill()
    c_full['kamis_smooth'] = c_full['kamis_price'].rolling(3, min_periods=1).mean()
    c_full['county_norm'] = county
    kamis_filled_list.append(c_full.reset_index().rename(columns={'index': 'week_start'}))

kamis_clean = pd.concat(kamis_filled_list)

# ==============================================================================
# RECENT DATA INTEGRATION (Zindi updates)
# ==============================================================================
recent_files = [
    'agriBORA_maize_prices_weeks_46_47.csv',
    'agriBORA_maize_prices_weeks_46_47_48.csv',
    'agriBORA_maize_prices_weeks_46_to_49.csv',
    'agriBORA_maize_prices_weeks_46_to_51.csv'
]

agri_full = agri_week.copy()
for filename in recent_files:
    try:
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            filepath = filename
            
        recent_df = pd.read_csv(filepath)
        recent_df['Date'] = pd.to_datetime(recent_df['Date'])
        recent_df['week_start'] = recent_df['Date'].dt.to_period("W").apply(lambda p: p.start_time)
        recent_df['county_norm'] = recent_df['County'].apply(norm_county)
        recent_df['price'] = pd.to_numeric(recent_df['WholeSale'], errors='coerce')
        
        recent_week = recent_df.groupby(['county_norm', 'week_start'], as_index=False)['price'].mean().rename(columns={'price': 'agri_price'})
        agri_full = pd.concat([agri_full, recent_week], ignore_index=True)
    except FileNotFoundError:
        continue

# Always keep the most recent data in case of duplicates
agri_full = agri_full.drop_duplicates(subset=['county_norm', 'week_start'], keep='last')

# ==============================================================================
# RECENT TREND ANALYSIS
# ==============================================================================
recent_trends = {}
recent_means = {}

for county in TARGET_COUNTIES:
    county_data = agri_full[agri_full['county_norm'] == county].sort_values('week_start')
    last_3 = county_data['agri_price'].tail(3).values
    last_6 = county_data['agri_price'].tail(6).values
    
    if len(last_3) >= 3:
        trend = last_3[-1] - last_3[0]
        recent_mean = np.mean(last_6)
        recent_trends[county] = trend
        recent_means[county] = recent_mean

# ==============================================================================
# FEATURE ENGINEERING & TRAINING
# ==============================================================================
def create_features(df, kamis_data):
    df = pd.merge(df, kamis_data[['week_start', 'county_norm', 'kamis_smooth']], 
                  on=['week_start', 'county_norm'], how='left')
    df = df.sort_values(['county_norm', 'week_start'])
    
    # Lag features
    df['lag_1'] = df.groupby('county_norm')['agri_price'].shift(1)
    df['lag_2'] = df.groupby('county_norm')['agri_price'].shift(2)
    
    # Trend feature
    df['trend'] = df['lag_1'] - df['lag_2']
    
    # Rolling mean
    df['roll_mean_3'] = df.groupby('county_norm')['agri_price'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    
    # One-Hot Encoding for counties
    df = pd.get_dummies(df, columns=['county_norm'], prefix='county', dtype=float)
    return df

df_full = create_features(agri_full, kamis_clean)

# Train / Validation Split
train_cutoff = pd.Timestamp('2024-09-30')
val_cutoff = pd.Timestamp('2025-12-31')

df_train = df_full[df_full['week_start'] <= train_cutoff].copy()
df_val = df_full[(df_full['week_start'] > train_cutoff) & (df_full['week_start'] <= val_cutoff)].copy()

feature_cols = [c for c in df_train.columns if 
                'lag_' in c or 'roll_' in c or 'county_' in c or 
                c in ['kamis_smooth', 'trend']]

train_clean = df_train.dropna(subset=['agri_price', 'lag_1'])
val_clean = df_val.dropna(subset=['agri_price', 'lag_1'])

# Gradient Boosting Model
model = HistGradientBoostingRegressor(
    max_iter=80, learning_rate=0.03, max_depth=3,
    min_samples_leaf=15, l2_regularization=3.0, random_state=42
)

# Weighting: give more weight to 2024 data
weights = train_clean['week_start'].apply(lambda x: 2.0 if x.year >= 2024 else 1.0)
model.fit(train_clean[feature_cols], train_clean['agri_price'], sample_weight=weights)

# Validation and bias calculation per county
val_pred = model.predict(val_clean[feature_cols])
county_bias = {}
for county in TARGET_COUNTIES:
    county_col = f'county_{county}'
    if county_col in val_clean.columns:
        mask = val_clean[county_col] == 1
        if mask.sum() > 0:
            residuals = val_clean[mask]['agri_price'].values - val_pred[mask]
            county_bias[county] = np.median(residuals)

# ==============================================================================
# PREDICTION GENERATION
# ==============================================================================
dates_to_predict = [pd.Timestamp('2024-12-23'), pd.Timestamp('2024-12-30')]
submission_data = []

for date in dates_to_predict:
    week_num = date.isocalendar().week
    
    for county in TARGET_COUNTIES:
        # Retrieve historical data
        county_data = agri_full[agri_full['county_norm'] == county].sort_values('week_start')
        hist = county_data['agri_price'].tail(6).values
        
        if len(hist) >= 2:
            l1, l2 = hist[-1], hist[-2]
            trend = l1 - l2
            roll_mean = np.mean(hist[-3:])
            
            # Corresponding Kamis data
            k_val = kamis_clean[(kamis_clean['county_norm'] == county) & (kamis_clean['week_start'] == date)]['kamis_smooth'].values
            k_curr = k_val[0] if len(k_val) > 0 else l1
            
            # Create feature row
            row_feat = {
                'lag_1': l1, 'lag_2': l2,
                'trend': trend,
                'roll_mean_3': roll_mean,
                'kamis_smooth': k_curr
            }
            for c in TARGET_COUNTIES:
                row_feat[f'county_{c}'] = 1.0 if c == county else 0.0
            
            # Hybrid Strategy
            # 1. ML Model
            pred_model = model.predict(pd.DataFrame([row_feat])[feature_cols])[0]
            # 2. Naive Approach (Last value + recent trend)
            pred_naive = l1 + (recent_trends.get(county, 0) * 0.3)
            
            # 3. Mix (60% ML, 40% Naïve)
            prediction = 0.6 * pred_model + 0.4 * pred_naive
            
            # 4. Bias correction (70%)
            prediction = prediction + county_bias.get(county, 0) * 0.7
            
            # 5. Safety clip (+/- 10%)
            prediction = np.clip(prediction, l1 * 0.90, l1 * 1.10)
            
            sub_id = f"{county}_Week_{week_num}"
            submission_data.append([sub_id, prediction, prediction])
        else:
            # Fallback if no data
            prediction = 40.0
            sub_id = f"{county}_Week_{week_num}"
            submission_data.append([sub_id, prediction, prediction])

# ==============================================================================
# SAVE SUBMISSION
# ==============================================================================
submission = pd.DataFrame(submission_data, columns=['ID', 'Target_RMSE', 'Target_MAE']).round(2)
output_path = os.path.join(OUTPUT_DIR, 'submission_final_hybrid.csv')
submission.to_csv(output_path, index=False)

print(f"File generated successfully: {output_path}")