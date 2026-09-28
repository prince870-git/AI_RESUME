import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🎯 AI-Based Resume Screening System")

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/resume_screening_model.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model file not found. Please execute 'python src/train_model.py' first.")
else:
    st.header("📝 Candidate Profile Input")
    exp = st.slider("Years of Experience", 0.0, 15.0, 4.0, 0.5)
    skill = st.slider("Skill Alignment Score", 0.0, 1.0, 0.75, 0.05)
    edu = st.selectbox("Highest Education Level", ["Bachelors", "Masters", "PhD"])
    proj = st.slider("Project Exposure Score", 1, 10, 5)
    quality = st.slider("Resume Quality Score", 1, 10, 7)
    os_contrib = st.checkbox("Active Open Source Contributions?")

    if st.button("Evaluate Application"):
        input_df = pd.DataFrame([{
            'Experience_Years': exp, 'Skill_Alignment_Score': skill,
            'Education_Level': edu, 'Project_Exposure_Score': proj,
            'Resume_Quality_Score': quality, 'Open_Source_Contributions': int(os_contrib)
        }])
        
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        st.subheader("📊 Screening Results")
        st.metric(label="Match Confidence", value=f"{prob*100:.1f}%")
        if prediction == 1:
            st.success("✅ **Recommendation: Shortlist Candidate**")
        else:
            st.error("❌ **Recommendation: Do Not Shortlist**")
