import pandas as pd

def check_bias():
    df = pd.read_csv('data/resume_screening.csv')
    
    print("⚖️ Selection Rates by Education Level:")
    metrics = df.groupby('Education_Level').agg(
        Total_Applicants=('Shortlisted', 'count'),
        Shortlisted_Count=('Shortlisted', 'sum'),
        Selection_Rate=('Shortlisted', 'mean')
    )
    print(metrics)
    
    max_rate = metrics['Selection_Rate'].max()
    min_rate = metrics['Selection_Rate'].min()
    disparate_impact = min_rate / max_rate
    print(f"\nMinimum vs Maximum Selection Ratio: {disparate_impact:.4f}")
    if disparate_impact < 0.8:
        print("⚠️ Potential algorithmic bias detected (fails 80% rule).")
    else:
        print("✅ Fair distribution across groups.")

if __name__ == "__main__":
    check_bias()
