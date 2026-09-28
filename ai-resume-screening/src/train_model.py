import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_pipeline():
    df = pd.read_csv('data/resume_screening.csv')
    X = df.drop(columns=['Shortlisted'])
    y = df['Shortlisted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), ['Education_Level'])], 
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss'))
    ])
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("📊 Model Evaluation Report:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    
    joblib.dump(model, 'models/resume_screening_model.pkl')
    print("💾 Model artifact saved to models/resume_screening_model.pkl")

if __name__ == "__main__":
    train_pipeline()
