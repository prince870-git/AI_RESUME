import joblib
import pandas as pd
import shap

def explain():
    model = joblib.load('models/resume_screening_model.pkl')
    df = pd.read_csv('data/resume_screening.csv')
    
    X = df.drop(columns=['Shortlisted'])
    X_trans = model.named_steps['preprocessor'].transform(X)
    
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_trans)
    
    print("🧠 SHAP Model explainability pipeline initialized successfully.")

if __name__ == "__main__":
    explain()
