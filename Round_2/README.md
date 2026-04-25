
# Epstein Documents Importance Ranker

This repository contains Machine Learning pipelines designed to predict the "Importance Score" of investigative documents from the U.S. House Oversight Epstein estate release.

The project uses **XGBoost (GPU-optimized)** and **LightGBM** to analyze text features (TF-IDF) and metadata (power mentions, lead types) to identify high-value documents.

## Prerequisite: Setup Virtual Environment

To ensure all dependencies work correctly and do not interfere with your system Python, **please use a virtual environment**.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

```

### 2. Create a Virtual Environment (`venv`)

**For Windows:**

```bash
python -m venv venv

```

**For macOS / Linux:**

```bash
python3 -m venv venv

```

### 3. Activate the Virtual Environment

**For Windows:**

```bash
.\venv\Scripts\activate

```


### 4. Install Dependencies

Once the virtual environment is active, install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```

---
