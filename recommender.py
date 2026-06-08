# ============================================================
# DecodeLabs AI Internship - Project 3
# Tech Stack Recommender
# Intern: Hashini Deepa
# Concepts: TF-IDF, Cosine Similarity, Content-Based Filtering
# ============================================================

# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Step 2: Load Dataset
# ============================================================
df = pd.read_csv("raw_skills.csv")
print("=" * 50)
print("        Tech Stack Recommender - DecoBot")
print("=" * 50)
print(f"\n✅ Loaded {len(df)} job roles from dataset.\n")

# ============================================================
# Step 3: Build TF-IDF Matrix for Job Roles
# ============================================================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["skills"])

# ============================================================
# Step 4: Take User Input (minimum 3 skills)
# ============================================================
print("Enter your skills one by one.")
print("Type 'done' when finished (minimum 3 skills required).\n")

user_skills = []
while True:
    skill = input(f"Skill {len(user_skills) + 1}: ").strip()
    if skill.lower() == "done":
        if len(user_skills) < 3:
            print("Please enter at least 3 skills!\n")
            continue
        else:
            break
    elif skill == "":
        print("Skill cannot be empty. Try again.\n")
        continue
    else:
        user_skills.append(skill)
        print(f"   ✔ Added: {skill}")

print(f"\n📌 Your Skills: {', '.join(user_skills)}\n")

# ============================================================
# Step 5: Vectorize User Profile using same TF-IDF vocabulary
# ============================================================
user_profile = " ".join(user_skills)
user_vector = vectorizer.transform([user_profile])

# ============================================================
# Step 6: Calculate Cosine Similarity Scores
# ============================================================
similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

# ============================================================
# Step 7: Sort and Filter - Top 3 Results
# ============================================================
top_n = 3
top_indices = similarity_scores.argsort()[::-1][:top_n]

# ============================================================
# Step 8: Display Recommendations
# ============================================================
print("=" * 50)
print("        🏆 Top 3 Career Recommendations")
print("=" * 50)

for rank, idx in enumerate(top_indices, start=1):
    role = df.iloc[idx]["job_role"]
    score = similarity_scores[idx]
    match_percent = round(score * 100, 2)

    print(f"\n  #{rank} {role}")
    print(f"      Match Score : {score:.4f}")
    print(f"      Match %     : {match_percent}%")
    print(f"      Key Skills  : {df.iloc[idx]['skills'].replace('_', ' ')}")

print("\n" + "=" * 50)
print("Tip: Add more specific skills to improve your match!")
print("Powered by DecodeLabs | Intern: Hashini Deepa")
print("=" * 50)
