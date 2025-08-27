import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = []
    
    def safe_prob(p):
        return np.clip(p, 0.01, 0.99)  # Garantit 1% ≤ probabilité ≤ 99%
    
    for _ in range(n_samples):
        # Données de base
        age = max(18, min(80, int(np.random.normal(45, 15))))
        weight = max(40, min(150, int(np.random.normal(75, 20))))
        waist = max(60, min(140, int(weight * 0.85 + np.random.normal(0, 10))))
        
        # Choix simples
        activity = np.random.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1])
        diet = np.random.choice([0, 1, 2], p=[0.4, 0.45, 0.15])
        family_history = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Probabilités protégées
        hypertension_prob = safe_prob(0.1 + (age - 18) * 0.005 + max(0, weight - 70) * 0.003)
        cholesterol_prob = safe_prob(0.15 + (age - 18) * 0.005 + (diet == 0) * 0.15)
        hypertension = np.random.choice([0, 1], p=[1-hypertension_prob, hypertension_prob])
        cholesterol = np.random.choice([0, 1], p=[1-cholesterol_prob, cholesterol_prob])
        
        # Autres variables
        gestational_diabetes = np.random.choice([0, 1], p=[0.9, 0.1])
        pcos = np.random.choice([0, 1], p=[0.85, 0.15])
        ethnicity = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.1, 0.1])
        symptoms_count = min(7, max(0, np.random.poisson(1)))
        
        # Risque simplifié
        risk_score = (
            2 if age > 45 else 0 +
            2 if age > 60 else 0 +
            2 if weight > 90 else 0 +
            1 if waist > 100 else 0 +
            (2 if activity == 0 else 1 if activity == 1 else 0) +
            (2 if diet == 0 else 1 if diet == 1 else 0) +
            3 if family_history else 0 +
            2 if hypertension else 0 +
            1 if cholesterol else 0 +
            2 if gestational_diabetes else 0 +
            1 if pcos else 0 +
            (1 if ethnicity in [1, 2] else 0) +
            symptoms_count +
            np.random.normal(0, 0.5)
        )
        
        diabetes_prob = safe_prob(1 / (1 + np.exp(-(risk_score - 8) * 0.3)))
        diabetes = np.random.choice([0, 1], p=[1-diabetes_prob, diabetes_prob])
        
        if diabetes == 1:
            symptoms_count = min(7, symptoms_count + np.random.poisson(0.5))
        
        data.append([
            age, weight, waist, activity, diet, family_history,
            hypertension, cholesterol, gestational_diabetes, pcos,
            ethnicity, symptoms_count, diabetes
        ])
    
    return pd.DataFrame(data, columns=[
        'age', 'weight', 'waist', 'activity', 'diet', 'family_history',
        'hypertension', 'cholesterol', 'gestational_diabetes', 'pcos',
        'ethnicity', 'symptoms_count', 'diabetes'
    ])

def train_model():
    print("🔄 Génération des données synthétiques...")
    df = generate_synthetic_data(5000)
    
    print("🔄 Entraînement du modèle...")
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump({'model': model, 'scaler': scaler}, 'model.pkl')
    print("✅ Modèle sauvegardé dans 'model.pkl'")

if __name__ == "__main__":
    print("🚀 Démarrage de l'entraînement du modèle")
    train_model()