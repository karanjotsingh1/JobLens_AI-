# ============================================================
# models/skill_model.py
#
# PURPOSE: Train an XGBoost model that predicts the TOP IN-DEMAND
#          SKILLS for a given job role + experience level.
#
# HOW IT WORKS:
#   - Input:  role (e.g. "ML Engineer") + level (e.g. "Fresher")
#   - Output: Top 10 skills ranked by frequency/importance + SHAP chart
#
# ALGORITHM CHOICE:
#   XGBoost is chosen because:
#   1. Works well on tabular/categorical data (role, level → skill)
#   2. Very fast training even on large datasets
#   3. SHAP (SHapley Additive exPlanations) integrates natively
#   4. Industry standard for structured ML problems
# ============================================================

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Use non-GUI backend so it works in Streamlit
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import classification_report
from xgboost                  import XGBClassifier

# Add parent to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ML_DATASET_PATH, MODEL_SAVE_PATH, XGBOOST_PARAMS


# ─────────────────────────────────────────────────────────────
# 1. LOAD & VALIDATE DATA
# ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the prepared CSV dataset.
    Each row = one (role, level, skill) triplet from a real job posting.
    """
    if not os.path.exists(ML_DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {ML_DATASET_PATH}.\n"
            "Please run: python data/prepare_dataset.py first."
        )

    df = pd.read_csv(ML_DATASET_PATH)

# Remove rare skill classes (less than 2 samples)
    skill_counts = df["skill"].value_counts()
    valid_skills = skill_counts[skill_counts >= 2].index
    df = df[df["skill"].isin(valid_skills)].reset_index(drop=True)
    print(f"✅ Loaded dataset: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """
    Convert categorical columns (role, level) into numeric form
    so XGBoost can process them.

    LabelEncoder maps each category to an integer:
      e.g. "ML Engineer" → 3, "Data Scientist" → 1
    """
    le_role  = LabelEncoder()
    le_level = LabelEncoder()
    le_skill = LabelEncoder()

    # Fit and transform each categorical column
    df["role_enc"]  = le_role.fit_transform(df["role"])
    df["level_enc"] = le_level.fit_transform(df["level"])
    df["skill_enc"] = le_skill.fit_transform(df["skill"])

    print(f"   Unique roles:  {len(le_role.classes_)}")
    print(f"   Unique levels: {len(le_level.classes_)}")
    print(f"   Unique skills: {len(le_skill.classes_)}")

    # X = features (what role + what level)
    # y = target   (what skill is needed for that role+level)
    X = df[["role_enc", "level_enc"]]
    y = df["skill_enc"]

    return X, y, le_role, le_level, le_skill


# ─────────────────────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────────────────────
def train_model(X, y):
    """
    Train XGBoost multi-class classifier.
    This predicts which skill is most likely needed for a role+level combo.
    """
    # Split 80% for training, 20% for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🤖 Training XGBoost model...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50   # Print progress every 50 trees
    )

    # Evaluate accuracy on test set
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, X_test, y_test


# ─────────────────────────────────────────────────────────────
# 4. SAVE MODEL + ENCODERS
# ─────────────────────────────────────────────────────────────
def save_artifacts(model, le_role, le_level, le_skill):
    """
    Save the trained model and all label encoders to disk.
    We bundle everything in one dict so loading is easy.
    """
    artifacts = {
        "model":    model,
        "le_role":  le_role,
        "le_level": le_level,
        "le_skill": le_skill,
    }
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────
# 5. LOAD MODEL (called from Streamlit UI)
# ─────────────────────────────────────────────────────────────
def load_artifacts() -> dict:
    """
    Load the saved model and encoders from disk.
    Returns a dict with keys: model, le_role, le_level, le_skill
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_SAVE_PATH}.\n"
            "Please run: python models/skill_model.py first."
        )

    with open(MODEL_SAVE_PATH, "rb") as f: #Python creates a file object as f
        artifacts = pickle.load(f)

    return artifacts


# ─────────────────────────────────────────────────────────────
# 6. PREDICT TOP SKILLS (main function called from UI)
# ─────────────────────────────────────────────────────────────
def predict_top_skills(role: str, level: str, top_n: int = 10) -> pd.DataFrame:
    """
    Given a job role and experience level, return the top N in-demand skills
    ranked by model probability.

    Args:
        role:  e.g. "ML Engineer", "Data Scientist", "Software Engineer"
        level: e.g. "Fresher", "Mid-Level", "Senior"
        top_n: how many skills to return (default 15)

    Returns:
        DataFrame with columns: skill, probability, rank
    """
    # Load model artifacts from disk
    arts    = load_artifacts()
    model   = arts["model"]
    le_role = arts["le_role"]
    le_level= arts["le_level"]
    le_skill= arts["le_skill"]

    # Encode the input role and level to numbers
    # Handle unknown categories gracefully
    if role not in le_role.classes_:
        role = le_role.classes_[0]   # fallback to first class
    if level not in le_level.classes_:
        level = le_level.classes_[0]

    role_enc  = le_role.transform([role])[0]
    level_enc = le_level.transform([level])[0]

    # Create input feature row for model
    X_input = pd.DataFrame(
        [[role_enc, level_enc]],
        columns=["role_enc", "level_enc"]
    )

    # Get probability for EACH possible skill class
    # probas shape: (1, num_unique_skills)
    probas = model.predict_proba(X_input)[0]

    # Map each skill index → skill name + probability
    skill_names = le_skill.classes_

    results = pd.DataFrame({
        "skill":       skill_names,
        "probability": probas,
    })

    # Sort by probability descending, take top N
    results = results.sort_values("probability", ascending=False).head(top_n)
    results["rank"] = range(1, len(results) + 1)
    results["probability_pct"] = (results["probability"] * 100).round(1)

    return results.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 7. GENERATE SHAP CHART
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# 7. GENERATE SHAP CHART
# ─────────────────────────────────────────────────────────────
def generate_shap_chart(role: str, level: str) -> plt.Figure:
    """
    Generate a SHAP Feature Importance chart for the selected
    Job Role and Experience Level.

    This visualizes how much each input feature contributes
    to the model's prediction.
    """

    # Load trained model and encoders
    arts = load_artifacts()
    model = arts["model"]
    le_role = arts["le_role"]
    le_level = arts["le_level"]

    # Encode user inputs
    role_enc = le_role.transform([role])[0] if role in le_role.classes_ else 0
    level_enc = le_level.transform([level])[0] if level in le_level.classes_ else 0

    # Create input dataframe
    X_input = pd.DataFrame(
        [[role_enc, level_enc]],
        columns=["role_enc", "level_enc"]
    )

    # Create SHAP explainer and compute explanations
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_input)

    # Extract feature importance values
    values = np.abs(explanation.values[0])

    # Average across classes for multi-class models
    if values.ndim == 2:
        values = values.mean(axis=1)

    feature_names = ["Job Role", "Experience Level"]

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.barh(
        feature_names,
        values,
        color=["#6C63FF", "#FF6584"]
    )

    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"SHAP Feature Importance\nRole: {role} | Level: {level}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────
# 8. TRAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  JobLens AI — Skill Trend Model Training")
    print("=" * 60)

    df              = load_data()
    X, y, le_role, le_level, le_skill = engineer_features(df)
    model, X_test, y_test             = train_model(X, y)
    save_artifacts(model, le_role, le_level, le_skill)

    print("\n🎉 Training complete! You can now run the Streamlit app.")
    print("   Command: streamlit run ui/app.py")
