# ============================================================
# data/prepare_dataset.py
#
# PURPOSE: Download & prepare the job skills dataset for ML training.
#
# WHY THIS DATASET?
#   We use the "LinkedIn Job Postings 2023-2024" dataset from Kaggle.
#   It has 33,000+ real job postings with skills, titles, and levels.
#   This is more relevant than older datasets because it reflects
#   TODAY's job market — LLMs, GenAI, LangChain etc. are all present.
#
# HOW TO GET THE DATA:
#   1. Go to https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
#   2. Download "job_skills.csv" and "job_postings.csv"
#   3. Place both files in data/raw/ folder
#   4. Run this script: python data/prepare_dataset.py
# ============================================================

import pandas as pd
import numpy as np
import os
import sys

# Add parent folder to path so we can import config
#__file__ is a special Python variable. It automatically stores the path of the currently executing file.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ML_DATASET_PATH


def prepare_dataset():
    """
    Merges job postings with their skills, cleans the data,
    and saves a single clean CSV ready for ML training.
    """

    # ── Step 1: Load raw files ───────────────────────────────
    print("📂 Loading raw Kaggle files...")

    postings_path = "data/raw/job_postings.csv"
    skills_path   = "data/raw/job_skills.csv"

    # Check if files exist before trying to load
    if not os.path.exists(postings_path) or not os.path.exists(skills_path):
        print("❌ ERROR: Raw files not found!")
        print("   Please download from Kaggle:")
        print("   https://www.kaggle.com/datasets/arshkon/linkedin-job-postings")
        print("   Place 'job_postings.csv' and 'job_skills.csv' in data/raw/")
        return

    # Load both CSV files
    postings = pd.read_csv(postings_path)
    skills   = pd.read_csv(skills_path)

    print(f"   ✅ Postings loaded: {len(postings):,} rows")
    print(f"   ✅ Skills loaded:   {len(skills):,} rows")

    # ── Step 2: Keep only needed columns from postings ───────
    # title       → job role name (e.g., "Machine Learning Engineer")
    # formatted_experience_level → Fresher / Mid / Senior
    # job_id      → foreign key to join with skills
    keep_cols = ["job_id", "title", "formatted_experience_level", "location"]
    postings  = postings[keep_cols].dropna(subset=["title"])
    # This checks the title column for missing values and drops those rows, since we can't categorize a job without a title.

    # ── Step 3: Merge postings with skills on job_id ─────────
    # skills table has: job_id | skill_abr (e.g., "PYTH", "SQL", "MLEARN")
    merged = postings.merge(skills, on="job_id", how="inner")

    print(f"   ✅ Merged dataset:  {len(merged):,} rows")

    # ── Step 4: Normalize job titles into broad categories ───
    # We map messy titles → clean role names for ML training
    def categorize_role(title: str) -> str:
        """Map raw job title to one of 8 clean role categories."""
        title = str(title).lower()
        if any(k in title for k in ["machine learning", "ml engineer", "ai engineer"]):
            return "ML Engineer"
        elif any(k in title for k in ["data scientist", "data science"]):
            return "Data Scientist"
        elif any(k in title for k in ["data analyst", "business analyst", "bi analyst"]):
            return "Data Analyst"
        elif any(k in title for k in ["data engineer", "etl", "pipeline"]):
            return "Data Engineer"
        elif any(k in title for k in ["software engineer", "software developer", "sde", "swe", "full stack", "backend", "frontend"]):
            return "Software Engineer"
        elif any(k in title for k in ["devops", "cloud engineer", "site reliability", "sre"]):
            return "DevOps / Cloud Engineer"
        elif any(k in title for k in ["nlp", "natural language", "llm", "generative"]):
            return "NLP / GenAI Engineer"
        elif any(k in title for k in ["product manager", "product owner"]):
            return "Product Manager"
        else:
            return "Other"

    merged["role_category"] = merged["title"].apply(categorize_role)

    # ── Step 5: Normalize experience level ───────────────────
    def normalize_level(level: str) -> str:
        """Simplify experience level to Fresher / Mid-Level / Senior."""
        level = str(level).lower()
        if any(k in level for k in ["entry", "intern", "associate", "fresher"]):
            return "Fresher"
        elif any(k in level for k in ["mid", "middle", "experienced"]):
            return "Mid-Level"
        elif any(k in level for k in ["senior", "lead", "principal", "staff", "director"]):
            return "Senior"
        else:
            return "Mid-Level"  # default to mid if unknown

    merged["experience_level"] = merged["formatted_experience_level"].apply(normalize_level)

    # ── Step 6: Remove "Other" category (not useful for training) ──
    merged = merged[merged["role_category"] != "Other"]

    # ── Step 7: Keep only relevant columns and save ──────────
    final = merged[["role_category", "experience_level", "skill_abr"]].copy() 
    #.copy() creates a new DataFrame to make sure changes done in final don't affect merged DataFrame
    final.columns = ["role", "level", "skill"]

    # Remove rows with missing skill
    final = final.dropna(subset=["skill"])

    # Save to the path defined in config and donot need to take index column in the CSV file, so index=False
    final.to_csv(ML_DATASET_PATH, index=False)

    print(f"\n✅ Dataset prepared and saved to: {ML_DATASET_PATH}")
    print(f"   Total rows: {len(final):,}")
    print(f"   Unique roles: {final['role'].nunique()}")
    print(f"   Unique skills: {final['skill'].nunique()}")
    print(f"\n   Role distribution:")
    print(final["role"].value_counts().to_string())


# This allows the script to be run directly from the command line, and it will execute the prepare_dataset function.
if __name__ == "__main__":
    prepare_dataset()
