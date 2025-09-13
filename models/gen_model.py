from pathlib import Path
import sqlite3, json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings("ignore")

Path("models").mkdir(exist_ok=True)

# yaha pe model ko train karna hai
np.random.seed(42)
SKILLS = [
    "flying", "navigation", "engineering", "radar", "medicine",
    "logistics", "communications", "cybersec", "maintenance", "intelligence",
    "leadership", "electronics", "weapon_systems", "surveillance", "meteorology"
]

def random_skills(n=3):
    return list(np.random.choice(SKILLS, size=n, replace=False))

def gen_personnel(n=200):
    rows = []
    for i in range(1, n+1):
        skills = random_skills(np.random.randint(2,5))
        health = int(np.clip(np.random.normal(85,10), 50, 100))
        experience = int(np.random.poisson(6))
        rank = np.random.choice(["Junior", "Mid", "Senior", "Commander"], p=[0.4,0.3,0.2,0.1])
        performance = float(np.clip(np.random.normal(70,10), 40, 100))
        attr_prob = (60 - performance)/100 + (experience < 3)*0.1 + (rank=="Junior")*0.05 + np.random.normal(0,0.05)
        attr = int(np.random.rand() < np.clip(attr_prob, 0, 0.6))
        recent_missions = np.random.randint(0,8)
        readiness_score = np.clip((health*0.4 + performance*0.4 + (5-recent_missions)*4)/100, 0, 1)
        rows.append({
            "person_id": i,
            "name": f"Officer_{i}",
            "rank": rank,
            "skills": skills,
            "health": health,
            "experience": experience,
            "performance": performance,
            "recent_missions": recent_missions,
            "attrition": attr,
            "readiness_score": readiness_score
        })
    return pd.DataFrame(rows)

def gen_roles():
    roles = [
        {"role_id": 1, "role": "Pilot", "required_skills": ["flying","navigation"], "min_health": 80, "min_experience": 4, "leadership_required": False},
        {"role_id": 2, "role": "Engineer", "required_skills": ["engineering","electronics"], "min_health": 70, "min_experience": 3, "leadership_required": False},
        {"role_id": 3, "role": "Radar Specialist", "required_skills": ["radar","surveillance"], "min_health": 70, "min_experience": 2, "leadership_required": False},
        {"role_id": 4, "role": "Medical Officer", "required_skills": ["medicine"], "min_health": 85, "min_experience": 2, "leadership_required": False},
        {"role_id": 5, "role": "Unit Commander", "required_skills": ["leadership","communications"], "min_health": 75, "min_experience": 8, "leadership_required": True},
    ]
    return pd.DataFrame(roles)

personnel_df = gen_personnel(200)
roles_df = gen_roles()

DB_PATH = Path("iaf_human_mgmt.db")
if DB_PATH.exists():
    DB_PATH.unlink()
con = sqlite3.connect(DB_PATH, check_same_thread=False)

tmp = personnel_df.copy()
tmp["skills"] = tmp["skills"].apply(json.dumps)
tmp.to_sql("personnel", con, index=False)
roles_df_copy = roles_df.copy()
roles_df_copy["required_skills"] = roles_df_copy["required_skills"].apply(json.dumps)
roles_df_copy.to_sql("roles", con, index=False)
con.commit()

mlb = MultiLabelBinarizer(classes=SKILLS)
skill_matrix = mlb.fit_transform(personnel_df["skills"])
skill_df = pd.DataFrame(skill_matrix, columns=mlb.classes_)
skill_df['person_id'] = personnel_df['person_id']

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(skill_matrix)
personnel_df['skill_cluster'] = clusters

tmp2 = personnel_df.copy()
tmp2["skills"] = tmp2["skills"].apply(json.dumps)
tmp2.to_sql("personnel", con, index=False, if_exists='replace')
con.commit()

X_attr = pd.concat([skill_df.drop(columns=["person_id"]).reset_index(drop=True),
                    personnel_df[["health","experience","performance","recent_missions"]].reset_index(drop=True)], axis=1)
y_attr = personnel_df["attrition"]
X_train, X_test, y_train, y_test = train_test_split(X_attr, y_attr, test_size=0.2, random_state=42)
attr_model = RandomForestClassifier(n_estimators=150, random_state=42)
attr_model.fit(X_train, y_train)
attr_score = attr_model.score(X_test, y_test)

X_read = X_attr.copy()
y_read = personnel_df["readiness_score"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_read, y_read, test_size=0.2, random_state=42)
read_model = RandomForestRegressor(n_estimators=150, random_state=42)
read_model.fit(Xr_train, yr_train)
read_score = read_model.score(Xr_test, yr_test)

personnel_df['has_leadership_skill'] = personnel_df['skills'].apply(lambda s: ('leadership' in s))
X_lead = pd.concat([X_attr.reset_index(drop=True),
                    personnel_df[['has_leadership_skill']].astype(int).reset_index(drop=True)], axis=1)
y_lead = (personnel_df['performance'] > 78).astype(int)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lead, y_lead, test_size=0.2, random_state=42)
lead_model = RandomForestClassifier(n_estimators=150, random_state=42)
lead_model.fit(Xl_train, yl_train)
lead_score = lead_model.score(Xl_test, yl_test)

joblib.dump(mlb, "models/mlb_skills.joblib")
joblib.dump(kmeans, "models/kmeans_skills.joblib")
joblib.dump(attr_model, "models/attr_model.joblib")
joblib.dump(read_model, "models/read_model.joblib")
joblib.dump(lead_model, "models/lead_model.joblib")

app = FastAPI(title="IAF Human Management - AI Backend Prototype")

class RoleRequest(BaseModel):
    role_id: int
    top_k: int = 5

class PersonMatch(BaseModel):
    person_id: int
    name: str
    score: float
    health: int
    experience: int
    missing_skills: list

def load_personnel():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql("SELECT * FROM personnel", c)
    df['skills'] = df['skills'].apply(json.loads)
    return df

def load_roles():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql("SELECT * FROM roles", c)
    df['required_skills'] = df['required_skills'].apply(json.loads)
    return df

def score_candidate_for_role(person_row, role_row, mlb, attr_model, read_model, lead_model):
    person_skills = set(person_row['skills'])
    required_skills = set(role_row['required_skills'])
    match_count = len(person_skills.intersection(required_skills))
    skill_ratio = match_count / max(len(required_skills), 1)
    health_norm = person_row['health'] / 100.0
    exp_norm = np.tanh(person_row['experience'] / 10.0)
    skill_vec = mlb.transform([person_row['skills']])[0]
    feat = np.concatenate([skill_vec, [person_row['health'], person_row['experience'], person_row['performance'], person_row['recent_missions']]])
    attr_prob = float(attr_model.predict_proba([feat])[0][1])
    read_pred = float(read_model.predict([feat])[0])
    lead_score = 0.0
    if role_row.get('leadership_required', False):
        feat_l = np.concatenate([skill_vec, [person_row['health'], person_row['experience'], person_row['performance'], person_row['recent_missions'], int('leadership' in person_row['skills'])]])
        lead_score = float(lead_model.predict_proba([feat_l])[0][1])
    composite = (0.45 * skill_ratio) + (0.15 * health_norm) + (0.15 * exp_norm) + (0.15 * read_pred) + (0.1 * (lead_score if role_row.get('leadership_required', False) else 0))
    composite = composite * (1 - 0.5 * attr_prob)
    return composite, {"skill_ratio": skill_ratio, "attr_prob": attr_prob, "read_pred": read_pred, "lead_score": lead_score}

@app.post("/match_role", response_model=list[PersonMatch])
def match_role(req: RoleRequest):
    roles = load_roles()
    role_row = roles[roles['role_id'] == req.role_id]
    if role_row.empty:
        raise HTTPException(status_code=404, detail="Role not found")
    role_row = role_row.iloc[0].to_dict()
    personnel = load_personnel()
    results = []
    for _, p in personnel.iterrows():
        if p['health'] < role_row['min_health'] or p['experience'] < role_row['min_experience']:
            continue
        composite, meta = score_candidate_for_role(p, role_row, mlb, attr_model, read_model, lead_model)
        missing = list(set(role_row['required_skills']) - set(p['skills']))
        results.append({"person_id": int(p['person_id']), "name": p['name'], "score": float(composite), "health": int(p['health']), "experience": int(p['experience']), "missing_skills": missing})
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:req.top_k]
    return results_sorted

@app.get("/attrition_risks")
def attrition_risks():
    df = load_personnel()
    skill_vecs = mlb.transform(df['skills'].tolist())
    X = np.concatenate([skill_vecs, df[["health","experience","performance","recent_missions"]].values], axis=1)
    probs = attr_model.predict_proba(X)[:,1]
    df['attrition_risk'] = probs
    top = df.sort_values("attrition_risk", ascending=False)[["person_id","name","rank","attrition_risk","health","experience"]].head(10)
    return top.to_dict(orient="records")

@app.get("/readiness_scores")
def readiness_scores():
    df = load_personnel()
    skill_vecs = mlb.transform(df['skills'].tolist())
    X = np.concatenate([skill_vecs, df[["health","experience","performance","recent_missions"]].values], axis=1)
    preds = read_model.predict(X)
    df['predicted_readiness'] = preds
    return df[["person_id","name","rank","predicted_readiness","health","experience"]].to_dict(orient="records")

@app.post("/recommend_training")
def recommend_training(role_id: int):
    roles = load_roles()
    role_row = roles[roles['role_id'] == role_id]
    if role_row.empty:
        raise HTTPException(status_code=404, detail="Role not found")
    role_row = role_row.iloc[0].to_dict()
    df = load_personnel()
    recs = []
    for _, p in df.iterrows():
        missing = list(set(role_row['required_skills']) - set(p['skills']))
        if missing:
            recs.append({"person_id": int(p['person_id']), "name": p['name'], "missing_skills": missing})
    return recs[:30]

@app.get("/dashboard_summary")
def dashboard_summary():
    df = load_personnel()
    total = len(df)
    by_rank = df['rank'].value_counts().to_dict()
    skill_lists = df['skills'].explode().value_counts().to_dict()
    cluster_counts = df['skill_cluster'].value_counts().to_dict() if 'skill_cluster' in df.columns else {}
    return {"total_personnel": total, "by_rank": by_rank, "skill_counts": skill_lists, "cluster_counts": cluster_counts}

client = TestClient(app)
resp1 = client.post("/match_role", json={"role_id":1, "top_k":5})
resp2 = client.get("/attrition_risks")
resp5 = client.get("/dashboard_summary")

print("Match Role (Pilot) - Top 5 Results:")
print(json.dumps(resp1.json(), indent=2))

print("\nTop Attrition Risks (sample):")
print(json.dumps(resp2.json(), indent=2))

print("\nDashboard Summary:")
print(json.dumps(resp5.json(), indent=2))

print(f"\nModels trained: Attrition acc={attr_score:.3f}, Readiness R2={read_score:.3f}, Leadership acc={lead_score:.3f}")