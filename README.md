# DecodeLabs-Internship-project-3

 Project 3 — Tech Stack Recommender

# Overview
A content-based filtering recommendation engine that maps a user's raw skills to the most relevant tech job roles. Powered by TF-IDF vectorization and Cosine Similarity.

#Tech Stack
Python · scikit-learn · pandas · TF-IDF · Cosine Similarity

#How It Works

User provides 3+ skills (e.g., Python, Cloud, Automation)
      ↓
Skills mapped to TF-IDF vector space
      ↓
Cosine Similarity calculated against job role vectors
      ↓
Top 3 matching job roles returned

#Pipeline (4 Steps)
StepAction

1. IngestionCapture user skill inputs
2. ScoringCalculate cosine similarity scores
3. SortingRank items by score (descending)
4. FilteringReturn Top-N results only
