
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "student_data.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Ensure data/student_data.csv exists."
        )
    return pd.read_csv(DATA_PATH, sep=";")


df = load_data()

# --------------------------------------------------
# 3. PREPARE DATA
# --------------------------------------------------
features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 4. TRAIN AND COMPARE THREE MODELS
# --------------------------------------------------
@st.cache_resource
def train_and_compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}
    results = []
    predictions_by_model = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        trained_models[name] = model
        predictions_by_model[name] = predicted

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predicted),
            "MSE": mean_squared_error(y_test, predicted),
            "R² Score": r2_score(y_test, predicted)
        })

    results_df = pd.DataFrame(results)

    # Lower MAE is used to select the displayed best model.
    best_model_name = results_df.loc[
        results_df["MAE"].idxmin(), "Model"
    ]

    return trained_models, results_df, predictions_by_model, best_model_name


trained_models, model_results, predictions_by_model, best_model_name = (
    train_and_compare_models(X_train, y_train, X_test, y_test)
)

# Keep Linear Regression as the prediction model so that
# existing predictions remain consistent with the earlier app.
model = trained_models["Linear Regression"]

linear_predictions = predictions_by_model["Linear Regression"]
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

# --------------------------------------------------
# 5. DASHBOARD HEADER
# --------------------------------------------------
st.title("🎓 Student Performance Predictor")

st.write(
    "Analyze student academic performance and predict final "
    "mathematics grades using machine learning."
)

st.caption(
    "Python | Pandas | Scikit-learn | Streamlit | Machine Learning"
)

st.divider()

# --------------------------------------------------
# 6. KEY PERFORMANCE INDICATORS
# --------------------------------------------------
total_students = len(df)
average_grade = df[target].mean()
pass_count = int((df[target] >= 10).sum())
fail_count = total_students - pass_count
pass_percentage = pass_count / total_students * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", total_students)
col2.metric("Average Final Grade", f"{average_grade:.2f}/20")
col3.metric("Pass Rate", f"{pass_percentage:.1f}%")
col4.metric("Linear Regression MAE", f"{linear_mae:.2f}")

st.divider()

# --------------------------------------------------
# 7. STUDENT PERFORMANCE INSIGHTS
# --------------------------------------------------
st.header("📊 Student Performance Insights")

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Average Grade by Study Time")

    study_chart = (
        df.groupby("studytime")[target]
        .mean()
        .reindex([1, 2, 3, 4])
        .reset_index()
    )

    study_chart["Study Time"] = study_chart["studytime"].map({
        1: "Below 2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: "Above 10 hours"
    })

    st.bar_chart(
        study_chart,
        x="Study Time",
        y=target,
        x_label="Weekly Study Time Category",
        y_label="Average Final Grade"
    )

with right_chart:
    st.subheader("Pass vs Below Passing Grade")

    result_chart = pd.DataFrame({
        "Result": ["Pass", "Below Passing Grade"],
        "Students": [pass_count, fail_count]
    })

    st.bar_chart(
        result_chart,
        x="Result",
        y="Students",
        x_label="Student Result",
        y_label="Number of Students"
    )

st.caption(
    "A grade of 10 or higher is counted as a pass for this project."
)

st.divider()

# --------------------------------------------------
# 8. FINAL GRADE DISTRIBUTION
# --------------------------------------------------
st.subheader("📈 Final Grade Distribution")

grade_distribution = (
    df[target]
    .value_counts()
    .sort_index()
    .rename_axis("Final Grade")
    .reset_index(name="Number of Students")
)

st.bar_chart(
    grade_distribution,
    x="Final Grade",
    y="Number of Students",
    x_label="Final Grade (0–20)",
    y_label="Number of Students"
)

# --------------------------------------------------
# 9. MODEL COMPARISON
# --------------------------------------------------
st.divider()
st.header("🤖 Machine Learning Model Comparison")

st.write(
    "Three regression models are trained and evaluated using "
    "the same training and testing records."
)

display_results = model_results.copy()
display_results["MAE"] = display_results["MAE"].round(3)
display_results["MSE"] = display_results["MSE"].round(3)
display_results["R² Score"] = display_results["R² Score"].round(3)

st.dataframe(
    display_results,
    hide_index=True,
    use_container_width=True
)

st.subheader("Compare MAE Scores")

st.bar_chart(
    model_results.set_index("Model")[["MAE"]],
    x_label="Model",
    y_label="Mean Absolute Error (lower is better)"
)

st.subheader("Compare R² Scores")

st.bar_chart(
    model_results.set_index("Model")[["R² Score"]],
    x_label="Model",
    y_label="R² Score (higher is generally better)"
)

st.info(
    f"Lowest-MAE model on this test split: **{best_model_name}**. "
    "This is a test-set comparison, not a guarantee of future performance."
)

with st.expander("What do the evaluation metrics mean?"):
    st.markdown(
        """
        - **MAE:** Average absolute difference between actual and predicted grades.
          Lower is better.
        - **MSE:** Average squared prediction error. Lower is better.
        - **R² Score:** Measures how much variation in the test grades is
          explained by the model. Higher is generally better; it can be negative.
        """
    )

# --------------------------------------------------
# 10. INDIVIDUAL STUDENT PREDICTION
# --------------------------------------------------
st.divider()
st.header("🎯 Predict an Individual Student's Grade")

st.write(
    "Enter the student's study time, absences, and previous grades."
)

input_col1, input_col2 = st.columns(2)

with input_col1:
    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda value: {
            1: "1 - Less than 2 hours",
            2: "2 - 2 to 5 hours",
            3: "3 - 5 to 10 hours",
            4: "4 - More than 10 hours"
        }[value]
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=0,
        step=1
    )

with input_col2:
    g1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

    g2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

prediction_model_name = st.selectbox(
    "Choose Prediction Model",
    options=list(trained_models.keys()),
    index=0
)

if st.button(
    "🔮 Predict Final Grade",
    type="primary",
    use_container_width=True
):
    input_data = pd.DataFrame(
        [[studytime, absences, g1, g2]],
        columns=features
    )

    selected_model = trained_models[prediction_model_name]
    raw_prediction = float(selected_model.predict(input_data)[0])

    predicted_grade = max(0.0, min(20.0, raw_prediction))
    predicted_percentage = predicted_grade / 20 * 100

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Predicted Final Grade",
        f"{predicted_grade:.2f} / 20"
    )

    result_col2.metric(
        "Predicted Percentage",
        f"{predicted_percentage:.1f}%"
    )

    if predicted_grade >= 10:
        st.success("✅ Predicted result: Pass")
    else:
        st.warning("⚠️ Predicted result: Below passing grade")

    if raw_prediction < 0 or raw_prediction > 20:
        st.caption(
            "The raw model output was outside 0–20 and has been "
            "bounded for display."
        )

    prediction_report = pd.DataFrame({
        "Study Time Category": [studytime],
        "Absences": [absences],
        "First Period Grade": [g1],
        "Second Period Grade": [g2],
        "Prediction Model": [prediction_model_name],
        "Predicted Final Grade": [round(predicted_grade, 2)],
        "Predicted Percentage": [round(predicted_percentage, 2)],
        "Predicted Result": [
            "Pass" if predicted_grade >= 10
            else "Below passing grade"
        ]
    })

    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=prediction_report.to_csv(index=False),
        file_name="student_prediction_report.csv",
        mime="text/csv"
    )

    st.caption(
        "Predictions are estimates based on historical data and are "
        "not guaranteed academic outcomes."
    )

# --------------------------------------------------
# 11. DOWNLOAD DATASET AND MODEL RESULTS
# --------------------------------------------------
st.divider()
st.subheader("📥 Download Project Reports")

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="Download Student Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="student_performance_dataset.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="Download Model Comparison (CSV)",
        data=model_results.to_csv(index=False),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Student Performance Analysis, Comparison and Prediction "
    "Using Statistical Machine Learning"
)
