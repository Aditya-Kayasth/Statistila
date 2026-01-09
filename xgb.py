import pandas as pd
import numpy as np
import os
import time
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import hstack

# --- CONFIGURATION ---
# Assumes train.csv and test.csv are in the same folder as this script
CURRENT_DIR = os.getcwd()
TRAIN_PATH = os.path.join(CURRENT_DIR, "train.csv")
TEST_PATH = os.path.join(CURRENT_DIR, "test.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "xgb_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(OUTPUT_DIR, "best_xgb_model_gpu.pkl")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_xgb.csv")

# --- DATA PROCESSING ---
def clean_dataframe(df):
    df = df.copy()
    
    # Fill missing text
    text_cols = ['Headline', 'Reasoning', 'Key Insights', 'Lead Types', 'Power Mentions', 'Agencies', 'Tags']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    
    # Parse list-like strings
    list_cols = ['Lead Types', 'Power Mentions', 'Agencies', 'Tags']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(';') if item.strip()])
    
    # Generate Count Features
    df['count_leads'] = df['Lead Types'].apply(len)
    df['count_mentions'] = df['Power Mentions'].apply(len)
    df['count_agencies'] = df['Agencies'].apply(len)
    df['text_len'] = df['Reasoning'].apply(len)
    
    # Combine text for TF-IDF
    df['full_text'] = (
        df['Headline'] + " " + 
        df['Reasoning'] + " " + 
        df['Key Insights']
    )
    return df

print(f"Loading datasets from {CURRENT_DIR}...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Cleaning data...")
train_clean = clean_dataframe(train)
test_clean = clean_dataframe(test)

print("Vectorizing text (TF-IDF)...")
# Increased features to 5000 since you have 24GB VRAM
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
train_text_features = tfidf.fit_transform(train_clean['full_text'])
test_text_features = tfidf.transform(test_clean['full_text'])

print("Encoding categories...")
mlb = MultiLabelBinarizer()
train_lead_features = mlb.fit_transform(train_clean['Lead Types'])
test_lead_features = mlb.transform(test_clean['Lead Types'])

print("Stacking features...")
num_cols = ['count_leads', 'count_mentions', 'count_agencies', 'text_len']
X_train = hstack([train_clean[num_cols].values, train_lead_features, train_text_features])
X_test = hstack([test_clean[num_cols].values, test_lead_features, test_text_features])
y_train = train_clean['Importance Score']
test_ids = test_clean['id']

# --- MODEL TRAINING ---
print("Initializing XGBoost with GPU support...")

# Configuration for RTX 4090
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',      # Modern GPU method
    device='cuda',           # Activates GPU
    n_jobs=-1,
    random_state=42
)

# Expanded search space for high-end hardware
param_dist = {
    'n_estimators': [1000, 2000, 3000],
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [6, 8, 10, 12],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2]
}

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=15,             # Increased iterations
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=6,             # Requested verbosity
    n_jobs=-1,
    random_state=42
)

print("Starting Randomized Search on GPU...")
start_time = time.time()

search.fit(X_train, y_train)

elapsed = (time.time() - start_time) / 60
print(f"Training complete in {elapsed:.2f} minutes.")
print(f"Best RMSE: {-search.best_score_}")
print(f"Best Params: {search.best_params_}")

# Save artifacts
joblib.dump(search.best_estimator_, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# Prediction
print("Generating predictions...")
preds = search.predict(X_test)

submission_df = pd.DataFrame({
    'id': test_ids,
    'Importance Score': preds
})

submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to {SUBMISSION_FILE}")