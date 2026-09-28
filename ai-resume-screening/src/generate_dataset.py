import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    exp_years = np.random.uniform(0, 15, n_samples).round(1)
    skill_align = np.random.uniform(0.1, 1.0, n_samples).round(2)
    education = np.random.choice(['Bachelors', 'Masters', 'PhD'], n_samples, p=[0.6, 0.3, 0.1])
    proj_score = np.random.randint(1, 11, n_samples)
    resume_quality = np.random.randint(1, 11, n_samples)
    open_source = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    edu_mapping = {'Bachelors': 1.0, 'Masters': 1.2, 'PhD': 1.5}
    edu_weights = np.array([edu_mapping[e] for e in education])
    
    logit = (exp_years * 0.3) + (skill_align * 4.0) + (proj_score * 0.2) + (resume_quality * 0.1) + (open_source * 0.5) + (edu_weights * 0.5) - 5.0
    prob = 1 / (1 + np.exp(-logit))
    shortlisted = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'Experience_Years': exp_years,
        'Skill_Alignment_Score': skill_align,
        'Education_Level': education,
        'Project_Exposure_Score': proj_score,
        'Resume_Quality_Score': resume_quality,
        'Open_Source_Contributions': open_source,
        'Shortlisted': shortlisted
    })
    
    df.to_csv('data/resume_screening.csv', index=False)
    print("✅ Synthetic dataset created at data/resume_screening.csv")

if __name__ == "__main__":
    generate_synthetic_data()
