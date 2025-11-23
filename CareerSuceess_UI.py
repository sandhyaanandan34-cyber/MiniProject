import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained model (Logistic Regression or SVM pipeline)
# Make sure you saved the pipeline with preprocessing included
model = joblib.load("logistic_regression_model.pkl")

model2 = joblib.load("svm_model.pkl")
 
log_reg  = model

#For Visualisation
#START: Visualisation Function
def visualise(log_reg, field_filter=None):
    coef = log_reg.named_steps['classifier'].coef_[0]

    numeric_features = ['Age','High_School_GPA','SAT_Score','University_GPA',
                        'Internships_Completed','Projects_Completed','Certifications',
                        'Soft_Skills_Score','Networking_Score','Work_Life_Balance']

    categorical_features = ['Gender','Study_Domain','Entrepreneurship']

    feature_names = (
        log_reg.named_steps['preprocessor']
            .transformers_[0][1]['scaler']
            .get_feature_names_out(numeric_features).tolist()
        +
        log_reg.named_steps['preprocessor']
            .transformers_[1][1]['encoder']
            .get_feature_names_out(categorical_features).tolist()
    )

    importance_df = pd.DataFrame({'feature': feature_names, 'coef': coef})

    if field_filter:
        importance_df = importance_df[importance_df['feature'].str.contains(field_filter)]

    groups = {
        'Academic Strength': ['High_School_GPA','SAT_Score','University_GPA'],
        'Practical Experience': ['Internships_Completed','Projects_Completed','Certifications'],
        'Social Capital': ['Soft_Skills_Score','Networking_Score'],
        'Career Outcomes': ['Work_Life_Balance','Gender','Field_Category','Entrepreneurship']
    }

    group_importance = {}
    for group, feats in groups.items():
        mask = importance_df['feature'].str.contains('|'.join(feats))
        group_importance[group] = importance_df.loc[mask, 'coef'].abs().sum()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(group_importance.keys(), group_importance.values())
    title = "Aggregated Feature Importance"
    if field_filter:
        title += f" (Field: {field_filter})"
    ax.set_title(title)
    ax.set_ylabel("Sum of Absolute Coefficients")

    return fig

#END: Visualisation Function

st.title("Career Success Prediction")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=60, value=22)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
high_school_gpa = st.number_input("High School GPA", min_value=0.0, max_value=4.0, value=3.5)
sat_score = st.number_input("SAT Score", min_value=800, max_value=1600, value=1200)
university_gpa = st.number_input("University GPA", min_value=0.0, max_value=4.0, value=3.2)

field_category = st.selectbox("Field of Study Category",
                              ["STEM (Computer Science, Engineering, Medicine, Nursing)", "Business & Economics (Business, Finance, Marketing)", "Arts & Social Science (Arts, Psychology, Education, Law)", "Unknown"])

internships = st.number_input("Internships Completed", min_value=0, max_value=10, value=2)
projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=5)
certifications = st.number_input("Certifications", min_value=0, max_value=10, value=1)

soft_skills = st.slider("Soft Skills Score", 0, 10, 7)
networking = st.slider("Networking Score", 0, 10, 6)

years_to_promotion = st.number_input("Years to Promotion", min_value=0, max_value=10, value=3)
work_life_balance = st.slider("Work-Life Balance", 0, 10, 7)
entrepreneurship = st.selectbox("Entrepreneurship", ["Yes", "No"])

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'High_School_GPA': [high_school_gpa],
    'SAT_Score': [sat_score],
    'University_GPA': [university_gpa],
    'Study_Domain': [field_category],
    'Internships_Completed': [internships],
    'Projects_Completed': [projects],
    'Certifications': [certifications],
    'Soft_Skills_Score': [soft_skills],
    'Networking_Score': [networking],
    'Years_to_Promotion': [years_to_promotion],
    'Work_Life_Balance': [work_life_balance],
    'Entrepreneurship': [entrepreneurship]
})


# Predict
if st.button("Predict Career Success"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of success (class 1)

    if prediction == 1:
        st.success(f"✅ LR Model:\n Predicted: Career Success\nProbability of Success: {probability:.2%}")
        fig = visualise(log_reg)
        st.pyplot(fig)
    else:
        st.error(f"❌ LR Model:\n Predicted: Not Yet Successful\nProbability of Success: {probability:.2%}")
        fig = visualise(log_reg )
        st.pyplot(fig)

  
    st.markdown(" Academic Strength - High_School_GPA, SAT_Score, University_GPA  \nPractical Experience - Internships_Completed, Projects_Completed, Certifications    \nSocial Capital - Soft_Skills_Score, Networking_Score  \nCareer Outcomes (current state) - Years_to_Promotion, Work_Life_Balance, Field_Category (STEM, Business & Economics, Arts & Social Science), Gender, Entrepreneurship")



