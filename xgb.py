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
CURRENT_DIR = os.getcwd()
TRAIN_PATH = os.path.join(CURRENT_DIR, "train.csv")
TEST_PATH = os.path.join(CURRENT_DIR, "test.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "xgb_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(OUTPUT_DIR, "best_xgb_model_gpu.pkl")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_xgb.csv")

# Leave 2 CPU cores free so the computer doesn't freeze
N_JOBS = max(1, os.cpu_count() - 2)

def clean_dataframe(df):
    df = df.copy()
    text_cols = ['Headline', 'Reasoning', 'Key Insights', 'Lead Types', 'Power Mentions', 'Agencies', 'Tags']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    
    list_cols = ['Lead Types', 'Power Mentions', 'Agencies', 'Tags']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(';') if item.strip()])
    
    df['count_leads'] = df['Lead Types'].apply(len)
    df['count_mentions'] = df['Power Mentions'].apply(len)
    df['count_agencies'] = df['Agencies'].apply(len)
    df['text_len'] = df['Reasoning'].apply(len)
    
    df['full_text'] = (df['Headline'] + " " + df['Reasoning'] + " " + df['Key Insights'])
    return df

print(f"Loading datasets from {CURRENT_DIR}...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Cleaning data...")
train_clean = clean_dataframe(train)
test_clean = clean_dataframe(test)

print("Vectorizing text (TF-IDF)...")
# max_features=3000 is the sweet spot for accuracy vs memory usage
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2), dtype=np.float32)
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

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda',
    n_jobs=N_JOBS,
    random_state=42
)

param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10, 
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=6,
    n_jobs=N_JOBS,
    random_state=42
)

print("Starting Randomized Search on GPU...")
start_time = time.time()
search.fit(X_train, y_train)
elapsed = (time.time() - start_time) / 60

print(f"Training complete in {elapsed:.2f} minutes.")
print(f"Best RMSE: {-search.best_score_}")
print(f"Best Params: {search.best_params_}")

joblib.dump(search.best_estimator_, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

print("Generating predictions...")
preds = search.predict(X_test)

submission_df = pd.DataFrame({'id': test_ids, 'Importance Score': preds})
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to {SUBMISSION_FILE}")