from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import time  # Nouvelle importation ajoutée

app = Flask(__name__)

# Chargement du modèle
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    else:
        return None

model_data = load_model()

# Ajout pour forcer le rechargement du CSS
@app.context_processor
def inject_cache_buster():
    return dict(cache_buster=str(int(time.time())))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data is None:
            return jsonify({
                'error': 'Modèle non disponible. Veuillez d\'abord entraîner le modèle avec train_model.py'
            }), 500

        # Récupération des données du formulaire
        data = request.get_json()
        
        # Validation des champs obligatoires
        required_fields = ['age', 'weight', 'waist', 'activity', 'diet', 'family_history', 
                          'hypertension', 'cholesterol', 'gestational_diabetes', 'pcos', 'ethnicity']
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                return jsonify({'error': f'Le champ {field} est obligatoire'}), 400

        # Préparation des features
        features = []
        
        # Features numériques
        features.append(float(data['age']))
        features.append(float(data['weight']))
        features.append(float(data['waist']))
        
        # Activité physique (0-3)
        activity_map = {
            'none': 0,
            'low': 1,
            'moderate': 2,
            'high': 3
        }
        features.append(activity_map.get(data['activity'], 0))
        
        # Alimentation (0-2)
        diet_map = {
            'unhealthy': 0,
            'balanced': 1,
            'healthy': 2
        }
        features.append(diet_map.get(data['diet'], 1))
        
        # Features binaires
        features.append(1 if data['family_history'] == 'yes' else 0)
        features.append(1 if data['hypertension'] == 'yes' else 0)
        features.append(1 if data['cholesterol'] == 'yes' else 0)
        features.append(1 if data['gestational_diabetes'] == 'yes' else 0)
        features.append(1 if data['pcos'] == 'yes' else 0)
        
        # Origine ethnique (0-4)
        ethnicity_map = {
            'european': 0,
            'african': 1,
            'asian': 2,
            'middle_east': 3,
            'other': 4
        }
        features.append(ethnicity_map.get(data['ethnicity'], 0))
        
        # Symptômes (comptage)
        symptoms = data.get('symptoms', [])
        features.append(len(symptoms))
        
        # Prédiction
        features_array = np.array([features])
        scaled_features = model_data['scaler'].transform(features_array)
        
        # Probabilité de risque
        risk_proba = model_data['model'].predict_proba(scaled_features)[0][1]
        risk_percentage = int(risk_proba * 100)
        
        # Détermination du niveau de risque
        if risk_percentage < 30:
            risk_level = 'low'
            risk_text = 'Faible risque'
            risk_icon = '✅'
            risk_color = '#28A745'
        elif risk_percentage < 70:
            risk_level = 'moderate'
            risk_text = 'Risque modéré'
            risk_icon = '⚠️'
            risk_color = '#FFC107'
        else:
            risk_level = 'high'
            risk_text = 'Risque élevé'
            risk_icon = '❗'
            risk_color = '#DC3545'
        
        # Génération de recommandations personnalisées
        recommendations = generate_recommendations(data, risk_level)
        
        return jsonify({
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_text': risk_text,
            'risk_icon': risk_icon,
            'risk_color': risk_color,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

def generate_recommendations(data, risk_level):
    recommendations = []
    
    # Recommandations basées sur l'activité physique
    if data['activity'] in ['none', 'low']:
        recommendations.append("🏃‍♂️ Augmentez votre activité physique : au moins 30 minutes par jour, 5 jours par semaine")
    
    # Recommandations alimentaires
    if data['diet'] == 'unhealthy':
        recommendations.append("🥗 Adoptez une alimentation équilibrée : plus de légumes, fruits et fibres, moins de sucres raffinés")
    
    # Gestion du poids
    weight = float(data['weight'])
    if weight > 80:  # Critère simple
        recommendations.append("⚖️ Maintenez un poids santé par une alimentation équilibrée et de l'exercice régulier")
    
    # Suivi médical
    if data['hypertension'] == 'yes' or data['cholesterol'] == 'yes':
        recommendations.append("🩺 Consultez régulièrement votre médecin pour le suivi de votre hypertension/cholestérol")
    
    # Surveillance des symptômes
    symptoms = data.get('symptoms', [])
    if len(symptoms) >= 3:
        recommendations.append("👨‍⚕️ Consultez un médecin rapidement en raison du nombre de symptômes présents")
    
    # Recommandations générales par niveau de risque
    if risk_level == 'high':
        recommendations.append("🚨 Consultez un endocrinologue dans les plus brefs délais")
        recommendations.append("📊 Effectuez un bilan glycémique complet (glycémie à jeun, HbA1c)")
    elif risk_level == 'moderate':
        recommendations.append("🔍 Effectuez un contrôle glycémique annuel")
        recommendations.append("💪 Renforcez vos habitudes de vie saines")
    else:
        recommendations.append("👍 Continuez vos bonnes habitudes de vie")
        recommendations.append("🔄 Refaites le test dans un an ou en cas de changement de situation")
    
    return recommendations

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)