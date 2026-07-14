"""
Project 4: Prediction of Agriculture Crop Production in India
UpSkill Campus x UniConverge Technologies (UCT) Internship

Real dataset: crop_data.csv (data.gov.in, via Kaggle "Crop Production in India")
Columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
DATA_PATH = "crop_data.csv"
TARGET_COL = "Production"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all").T)

# ------------------------------------------------------------------
# 2. BASIC CLEANING
# ------------------------------------------------------------------
df = df.drop_duplicates()

# Season has trailing whitespace in the raw file (e.g. "Kharif     ")
df["Season"] = df["Season"].str.strip()

# Drop District_Name: 646 unique values is too high-cardinality for
# one-hot encoding to be practical; State_Name already captures most
# of the same geographic signal.
df = df.drop(columns=["District_Name"])

df = df.dropna(subset=[TARGET_COL])  # can't train without a target

# Fill remaining categorical NaNs with "Unknown", numeric with median
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown")
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Remove invalid rows (zero/negative area, negative production)
df = df[df["Area"] > 0]
df = df[df[TARGET_COL] >= 0]

# The raw data mixes units across crops (tonnes, nuts, bales, bunches),
# which makes Production extremely right-skewed (max ~1.25 billion vs
# median ~730). A log1p transform on both Area and Production stabilizes
# this for modeling; predictions are converted back with expm1 before
# computing error metrics, so reported MAE/RMSE are in original units.
df["Area_log"] = np.log1p(df["Area"])
df["Production_log"] = np.log1p(df[TARGET_COL])

print("\nShape after cleaning:", df.shape)

# ------------------------------------------------------------------
# 3. EDA (quick, printed — expand with matplotlib/seaborn in notebook)
# ------------------------------------------------------------------
print("\nTop crops by count:\n", df["Crop"].value_counts().head())
print("\nTop states by count:\n", df["State_Name"].value_counts().head())
print("\nCorrelation with target (numeric cols):")
numeric_df = df[["Crop_Year", "Area_log", "Production_log"]]
print(numeric_df.corr()["Production_log"].sort_values(ascending=False))

# ------------------------------------------------------------------
# 4. FEATURE / TARGET SPLIT
# ------------------------------------------------------------------
y = df["Production_log"]  # model on log scale; converted back for metrics
X = df.drop(columns=[TARGET_COL, "Production_log", "Area"])  # use Area_log instead of raw Area

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",  # keep numeric cols as-is
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. MODELS — baseline (Linear Regression) + main (Random Forest)
# ------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=60, max_depth=14, random_state=42, n_jobs=-1
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds_log = pipe.predict(X_test)

    # Convert both predictions and actuals back to real production units
    preds_real = np.expm1(preds_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    r2 = r2_score(y_test, preds_log)  # R2 measured on the modeled (log) scale

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    fitted_pipelines[name] = pipe
    print(f"\n{name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2 (log scale): {r2:.4f}")

# ------------------------------------------------------------------
# 6. PICK BEST MODEL, SAVE IT
# ------------------------------------------------------------------
# NOTE: we select by MAE in real production units, not log-scale R2.
# On this dataset the two metrics disagree (LinearRegression edges out
# on log-scale R2, but RandomForest has far lower real-unit MAE/RMSE) -
# MAE is what actually matters for a usable production estimate.
best_name = min(results, key=lambda k: results[k]["MAE"])
best_pipeline = fitted_pipelines[best_name]
print(f"\nBest model (by MAE): {best_name} (MAE={results[best_name]['MAE']:.2f}, R2={results[best_name]['R2']:.4f})")

joblib.dump(best_pipeline, "crop_production_model.joblib")
print("Saved best model to crop_production_model.joblib")

# ------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (only meaningful for tree-based model)
# ------------------------------------------------------------------
if best_name == "RandomForest":
    ohe = best_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(cat_feature_names) + numeric_cols
    importances = best_pipeline.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 10 feature importances:\n", fi.head(10))

# ------------------------------------------------------------------
# 8. SAMPLE INFERENCE
# ------------------------------------------------------------------
sample = X_test.iloc[[0]]
pred = best_pipeline.predict(sample)[0]
actual = y_test.iloc[0]
print(f"\nSample prediction: {pred:.2f} | Actual: {actual:.2f}")