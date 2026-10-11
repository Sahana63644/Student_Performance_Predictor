
# 🎓 Student Performance Predictor

A machine learning project that analyzes student academic data and predicts final mathematics grades using Linear Regression.

## 📌 Project Overview

This project helps explore student performance using study time, school absences, and previous grades. It includes an interactive Streamlit dashboard for visualizing data and generating grade predictions.

## 🎯 Objectives

- Analyze student academic performance.
- Explore the relationship between study time and final grades.
- Train a Linear Regression model.
- Evaluate predictions using MAE, MSE, and R².
- Build an interactive dashboard for student grade prediction.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Data Visualization
- Linear Regression

## 📊 Dataset

The project uses a student mathematics performance dataset containing 395 student records and 33 columns.

Selected features:
- `studytime` — weekly study-time category
- `absences` — number of school absences
- `G1` — first-period grade
- `G2` — second-period grade
- `G3` — final grade (prediction target)

## 🤖 Machine Learning Model

Algorithm: Linear Regression

The dataset is split into:
- 80% training data
- 20% testing data

Reported evaluation results:
- Mean Absolute Error (MAE): 1.31
- Mean Squared Error (MSE): 4.18
- R² Score: 0.796

These results are from the current train/test split and may vary if the dataset or model settings change.

## ✨ Features

- Student performance dashboard
- Average grade by study-time category
- Pass versus below-passing grade statistics
- Final grade distribution chart
- Individual student grade prediction
- Model evaluation metrics
- CSV downloads for the dataset and prediction reports

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Student_Performance_Predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## ⚠️ Disclaimer

Predictions are estimates based on historical data. They are not guaranteed academic outcomes and should not be used as the sole basis for student-related decisions.

## 👩‍💻 Author

Add your name and GitHub profile link here.
