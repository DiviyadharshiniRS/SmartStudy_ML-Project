import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🎓 SmartStudy - Student Performance Predictor")

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Feature selection
X = df.drop(["StudentID", "GPA", "GradeClass"], axis=1)
y = df["GPA"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
# ---------------- UI INPUT ----------------

st.header("Enter Student Details")

age = st.number_input("Age", 15, 25, 17)

# Gender
gender_label = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender_label == "Male" else 0

#Ethnicity
ethnicity_label = st.selectbox(
    "Ethnicity",
    ["Group 0", "Group 1", "Group 2", "Group 3"]
)
ethnicity_map = {
    "Group 0": 0,
    "Group 1": 1,
    "Group 2": 2,
    "Group 3": 3
}
ethnicity = ethnicity_map[ethnicity_label]

# Parental Education
parent_edu_label = st.selectbox(
    "Parental Education",
    ["No Education", "School", "High School", "Undergraduate", "Postgraduate"]
)
parent_edu_map = {
    "No Education": 0,
    "School": 1,
    "High School": 2,
    "Undergraduate": 3,
    "Postgraduate": 4
}
parent_edu = parent_edu_map[parent_edu_label]

# Study Time
study_time = st.slider("Study Time Weekly (hours)", 0, 20, 10)

# Absences
absences = st.slider("Number of Absences", 0, 30, 5)

# Tutoring
tutoring_label = st.selectbox("Tutoring Support", ["No", "Yes"])
tutoring = 1 if tutoring_label == "Yes" else 0

# Parental Support (FIXED)
parent_support_label = st.selectbox(
    "Parental Support Level",
    ["None", "Low", "Medium", "High"],
    help="Higher support improves academic performance"
)
parent_support_map = {
    "None": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3
}
parent_support = parent_support_map[parent_support_label]

# Extracurricular
extra_label = st.selectbox("Participates in Extracurricular Activities", ["No", "Yes"])
extra = 1 if extra_label == "Yes" else 0

# Sports
sports_label = st.selectbox("Participates in Sports", ["No", "Yes"])
sports = 1 if sports_label == "Yes" else 0

# Music
music_label = st.selectbox("Participates in Music", ["No", "Yes"])
music = 1 if music_label == "Yes" else 0

# Volunteering
volunteer_label = st.selectbox("Participates in Volunteering", ["No", "Yes"])
volunteer = 1 if volunteer_label == "Yes" else 0

# Predict button
if st.button("Predict Performance"):

    new_student = pd.DataFrame([[
        age, gender, ethnicity, parent_edu,
        study_time, absences, tutoring,
        parent_support, extra, sports,
        music, volunteer
    ]], columns=X.columns)

    # scale
    new_scaled = scaler.transform(new_student)

    # predict
    predicted_gpa = model.predict(new_scaled)[0]

    # convert to 10 scale
    gpa_10 = (predicted_gpa / 4) * 10

    # performance
    if predicted_gpa >= 3.5:
        performance = "Excellent"
    elif predicted_gpa >= 3.0:
        performance = "Good"
    elif predicted_gpa >= 2.0:
        performance = "Average"
    else:
        performance = "At Risk"

    # risk score
    risk_score = 1 - (predicted_gpa / 4)

    st.subheader("📊 Prediction Result")
    st.write(f"GPA (4 scale): {round(predicted_gpa,2)}")
    st.write(f"GPA (10 scale): {round(gpa_10,2)}")
    st.write(f"Performance: {performance}")
    st.write(f"Risk Score: {round(risk_score,2)}")

    # ---------------- Suggestions ----------------
    st.subheader("💡 Suggestions")

    avg_study = df["StudyTimeWeekly"].mean()
    avg_absence = df["Absences"].mean()

    if study_time < avg_study:
        st.write("- Increase study time")

    if absences > avg_absence:
        st.write("- Reduce absences")

    if parent_support < 2:
        st.write("- Improve academic support")

    if study_time >= avg_study and absences <= avg_absence:
        st.write("- Good performance, keep it up!")