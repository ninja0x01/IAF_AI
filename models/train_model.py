import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from datetime import datetime
import joblib
import os

def create_features(df):
    features_df = df.copy()
    
    rank_encoder = LabelEncoder()
    unit_encoder = LabelEncoder()
    role_encoder = LabelEncoder()
    
    features_df['rank_encoded'] = rank_encoder.fit_transform(features_df['rank'])
    features_df['unit_encoded'] = unit_encoder.fit_transform(features_df['unit'])
    features_df['role_encoded'] = role_encoder.fit_transform(features_df['role'])
    
    features_df['skills_count'] = features_df['skills'].apply(lambda x: len(x.split(',')))
    
    features_df['last_training_date'] = pd.to_datetime(features_df['last_training_date'])
    features_df['days_since_training'] = (datetime.now() - features_df['last_training_date']).dt.days
    
    feature_columns = [
        'years_service', 'medical_score', 'performance_score', 'leadership_rating',
        'skills_count', 'days_since_training', 'rank_encoded', 'unit_encoded', 'role_encoded'
    ]
    
    X = features_df[feature_columns]
    y = features_df['attrition_risk']
    
    return X, y, {
        'feature_columns': feature_columns,
        'rank_encoder': rank_encoder,
        'unit_encoder': unit_encoder,
        'role_encoder': role_encoder
    }

def train_model():
    
    # yaha pe data load karna hai
    data_file = "../data/personnel_data.csv"
    if not os.path.exists(data_file):
        data_file = "data/personnel_data.csv"
    
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} records")
    
    X, y, encoders = create_features(df)
    print(f"Created features: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training LogisticRegression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n=== Model Performance ===")
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.3f}")
    print(f"Train ROC AUC: {roc_auc_score(y_train, train_proba):.3f}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, test_proba):.3f}")
    
    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_test, test_pred))
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_importance': feature_importance
    }
    
    os.makedirs('.', exist_ok=True)
    
    # bhai yaha pe Save kiya hai model ko
    model_path = 'attrition_model.joblib'
    joblib.dump(model_artifacts, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model_artifacts

if __name__ == "__main__":
    train_model()