import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import hstack

# --- CONFIGURATION ---
CURRENT_DIR = os.getcwd()
TRAIN_PATH = os.path.join(CURRENT_DIR, "train.csv")
TEST_PATH = os.path.join(CURRENT_DIR, "test.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "lgbm_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_lgbm.csv")

# Leave 2 CPU cores free
N_JOBS = max(1, os.cpu_count() - 2)

def clean_dataframe(df):
    df = df.copy()
    text_cols = ['Headline', 'Reasoning', 'Key Insights', 'Lead Types', 'Power Mentions', 'Agencies', 'Tags']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    
    df['count_leads'] = df['Lead Types'].apply(lambda x: len(str(x).split(';')))
    df['count_mentions'] = df['Power Mentions'].apply(lambda x: len(str(x).split(';')))
    df['text_len'] = df['Reasoning'].apply(len)
    
    df['full_text'] = df['Headline'] + " " + df['Reasoning'] + " " + df['Key Insights']
    return df

print("Loading Data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train_clean = clean_dataframe(train)
test_clean = clean_dataframe(test)

print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2), dtype=np.float32)
train_text = tfidf.fit_transform(train_clean['full_text'])
test_text = tfidf.transform(test_clean['full_text'])

train_clean['Lead Types List'] = train_clean['Lead Types'].astype(str).apply(lambda x: [i.strip() for i in x.split(';') if i.strip()])
test_clean['Lead Types List'] = test_clean['Lead Types'].astype(str).apply(lambda x: [i.strip() for i in x.split(';') if i.strip()])

mlb = MultiLabelBinarizer()
train_leads = mlb.fit_transform(train_clean['Lead Types List'])
test_leads = mlb.transform(test_clean['Lead Types List'])

print("Stacking Features...")
num_cols = ['count_leads', 'count_mentions', 'text_len']
X_train = hstack([train_clean[num_cols].values, train_leads, train_text])
X_test = hstack([test_clean[num_cols].values, test_leads, test_text])
y_train = train_clean['Importance Score']

# --- MODEL TRAINING ---
print("Initializing LightGBM...")

lgbm_model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    random_state=42,
    n_jobs=N_JOBS,
    device='cpu' 
)

param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [-1, 10],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

search = RandomizedSearchCV(
    lgbm_model,
    param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=6,
    n_jobs=N_JOBS,
    random_state=42
)

print("Starting LightGBM Search...")
search.fit(X_train, y_train)

print(f"Best RMSE: {-search.best_score_}")
print(f"Best Params: {search.best_params_}")

preds = search.predict(X_test)
sub = pd.DataFrame({'id': test['id'], 'Importance Score': preds})
sub.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to {SUBMISSION_FILE}")