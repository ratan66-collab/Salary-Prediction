from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Global variables for model and encoders
model = None
le_job = None
le_seniority = None

def init_model():
    global model, le_job, le_seniority
    print("Loading dataset and training model...")
    
    data_path = 'eda_data.csv'
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    df = pd.read_csv(data_path)
    
    # Preprocess
    features = ['Rating', 'python_yn', 'job_simp', 'seniority']
    df_model = df[features + ['avg_salary']].copy()
    
    le_job = LabelEncoder()
    df_model['job_simp_encoded'] = le_job.fit_transform(df_model['job_simp'].astype(str))
    
    le_seniority = LabelEncoder()
    df_model['seniority_encoded'] = le_seniority.fit_transform(df_model['seniority'].astype(str))
    
    X = df_model[['Rating', 'python_yn', 'job_simp_encoded', 'seniority_encoded']]
    y = df_model['avg_salary']
    
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save column names to avoid warnings
    model.feature_names = list(X.columns)
    print("Model initialized!")

# Initialize on startup
init_model()

@app.route('/')
def home():
    if not le_job or not le_seniority:
        return "Model not initialized properly. Check console.", 500
        
    return render_template('index.html', 
                           jobs=list(le_job.classes_), 
                           seniority=list(le_seniority.classes_))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        rating = float(data.get('rating', 3.0))
        python_yn = int(data.get('python_yn', 0))
        job = str(data.get('job', 'na')).lower()
        seniority = str(data.get('seniority', 'na')).lower()

        # Handle unknown categories gracefully
        if job not in le_job.classes_: job = 'na'
        if seniority not in le_seniority.classes_: seniority = 'na'

        job_encoded = le_job.transform([job])[0]
        seniority_encoded = le_seniority.transform([seniority])[0]

        # Predict
        prediction_df = pd.DataFrame(
            [[rating, python_yn, job_encoded, seniority_encoded]], 
            columns=model.feature_names
        )
        base_prediction = model.predict(prediction_df)[0]
        
        # Enforce Rating vs Salary correlation: 
        # Base prediction is considered "average" around a 3.0 rating.
        # - 5.0 rating = +20% boost
        # - 1.0 rating = -20% penalty
        # This overrides the Random Forest's organic output for 'Rating' to ensure strict linearity.
        rating_multiplier = 1.0 + ((rating - 3.0) * 0.1)
        adjusted_prediction = base_prediction * rating_multiplier
        
        # Format response
        salary_thousands = adjusted_prediction * 1000
        
        return jsonify({
            'success': True,
            'prediction': f"${salary_thousands:,.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
