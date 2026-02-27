# Studentperformanceanalyizer
#Student Performance Analysis is a data analytics project that explores academic performance data to identify key factors influencing student scores and predict outcomes using machine learning techniques.
# Student Performance Analyzer
# Author: You
# Description: Analyze student academic performance using Data Analysis & ML

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. CREATE / LOAD DATASET
# -------------------------------

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7],
    "StudyHours": [5, 8, 2, 6, 3, 7, 4],
    "Attendance": [80, 90, 60, 85, 65, 88, 70],
    "PreviousMarks": [70, 85, 50, 78, 55, 82, 65],
    "Assignments": [75, 88, 55, 80, 60, 85, 68],
    "FinalScore": [72, 86, 52, 79, 58, 84, 66]
}

df = pd.DataFrame(data)
print("\nDataset Loaded Successfully\n")
print(df)

# -------------------------------
# 2. BASIC STATISTICAL ANALYSIS
# -------------------------------

print("\nStatistical Summary\n")
print(df.describe())

# -------------------------------
# 3. CORRELATION HEATMAP
# -------------------------------

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Between Performance Factors")
plt.show()

# -------------------------------
# 4. SCORE DISTRIBUTION
# -------------------------------

plt.figure()
plt.hist(df['FinalScore'], bins=5)
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.title("Student Performance Distribution")
plt.show()

# -------------------------------
# 5. ATTENDANCE VS FINAL SCORE
# -------------------------------

plt.figure()
plt.scatter(df['Attendance'], df['FinalScore'])
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.title("Attendance vs Final Score")
plt.show()

# -------------------------------
# 6. IDENTIFY AT-RISK STUDENTS
# -------------------------------

df['PerformanceLevel'] = df['FinalScore'].apply(
    lambda x: 'At Risk' if x < 60 else 'Good'
)

print("\nStudent Performance Levels\n")
print(df[['StudentID', 'FinalScore', 'PerformanceLevel']])

# -------------------------------
# 7. MACHINE LEARNING ANALYSIS
# -------------------------------

X = df[['StudyHours', 'Attendance', 'PreviousMarks', 'Assignments']]
y = df['FinalScore']

model = LinearRegression()
model.fit(X, y)

df['PredictedScore'] = model.predict(X)

print("\nActual vs Predicted Scores\n")
print(df[['StudentID', 'FinalScore', 'PredictedScore']])

# -------------------------------
# 8. ANALYSIS RESULT
# -------------------------------

print("\nKey Insights:")
print("- Higher attendance improves performance")
print("- Previous marks strongly affect final score")
print("- Low study hours lead to poor performance")

print("\nStudent Performance Analysis Completed Successfully!")
