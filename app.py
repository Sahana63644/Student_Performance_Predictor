import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

# Professional color palette
NAVY = "#0B1020"
PANEL = "#151C30"
PURPLE = "#8B5CF6"
TEAL = "#2DD4BF"
BLUE = "#60A5FA"
GREEN = "#34D399"
ORANGE = "#FBBF24"
RED = "#FB7185"
TEXT = "#F1F5F9"
MUTED = "#A7B0C0"

# ==================================================
# 2. CUSTOM CSS
# ==================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            135deg, {NAVY} 0%, #10182B 55%, #111827 100%
        );
        color: {TEXT};
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
        letter-spacing: -0.02em;
    }}

    [data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid #29344D;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: #29344D;
        border-radius: 14px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. PROFESSIONAL CHART STYLE
# ==================================================
def style_plot(fig, height=360):
    """Apply a consistent professional dark theme."""

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=TEXT,
            family="Arial, sans-serif",
            size=12,
        ),
        title_font=dict(
            size=17,
            color=TEXT,
        ),
        margin=dict(
            l=16,
            r=16,
            t=48,
            b=20,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#39445D",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(167,176,192,0.13)",
            zeroline=False,
            linecolor="#39445D",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
    )

    return fig


# ==================================================
# 4. LOAD DATASET
# ==================================================
DATA_PATH = Path(__file__).parent / "data" / "student_data.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Ensure data/student_data.csv exists."
        )

    data = pd.read_csv(DATA_PATH, sep=";")

    required_columns = {
        "studytime",
        "absences",
        "G1",
        "G2",
        "G3",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    return data


df = load_data()


# ==================================================
# 5. PREPARE DATA
# ==================================================
features = [
    "studytime",
    "absences",
    "G1",
    "G2",
]

target = "G3"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ==================================================
# 6. TRAIN MACHINE LEARNING MODELS
# ==================================================
@st.cache_resource
def train_and_compare_models(
    X_train,
    y_train,
    X_test,
    y_test,
):
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results = []
    predictions_by_model = {}

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test)

        trained_models[name] = estimator
        predictions_by_model[name] = predictions

        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(
                    y_test, predictions
                ),
                "MSE": mean_squared_error(
                    y_test, predictions
                ),
                "R² Score": r2_score(
                    y_test, predictions
                ),
            }
        )

    results_df = pd.DataFrame(results)

    best_model_name = results_df.loc[
        results_df["MAE"].idxmin(),
        "Model",
    ]

    return (
        trained_models,
        results_df,
        predictions_by_model,
        best_model_name,
    )


(
    trained_models,
    model_results,
    predictions_by_model,
    best_model_name,
) = train_and_compare_models(
    X_train,
    y_train,
    X_test,
    y_test,
)

# Keep Linear Regression as the default prediction model.
linear_predictions = predictions_by_model["Linear Regression"]

linear_mae = mean_absolute_error(
    y_test, linear_predictions
)

linear_mse = mean_squared_error(
    y_test, linear_predictions
)

linear_r2 = r2_score(
    y_test, linear_predictions
)


# ==================================================
# 7. DASHBOARD HEADER
# ==================================================
st.title("🎓 Student Performance Predictor")

st.write(
    "Analyze student academic performance and predict "
    "final mathematics grades using machine learning."
)

st.caption(
    "Python  •  Pandas  •  Scikit-learn  •  Plotly  "
    "•  Streamlit  •  Machine Learning"
)

st.divider()


# ==================================================
# 8. KEY PERFORMANCE INDICATORS
# ==================================================
total_students = len(df)

average_grade = df[target].mean()

pass_count = int(
    (df[target] >= 10).sum()
)

fail_count = total_students - pass_count

pass_percentage = (
    pass_count / total_students * 100
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    f"{total_students:,}",
)

col2.metric(
    "Average Final Grade",
    f"{average_grade:.2f}/20",
)

col3.metric(
    "Pass Rate",
    f"{pass_percentage:.1f}%",
)

col4.metric(
    "Linear Regression MAE",
    f"{linear_mae:.2f}",
)

st.divider()


# ==================================================
# 9. STUDENT PERFORMANCE INSIGHTS
# ==================================================
st.header("📊 Student Performance Insights")

left_chart, right_chart = st.columns(2)


# Average grade by study time
with left_chart:
    st.subheader("Average Grade by Study Time")

    study_chart = (
        df.groupby("studytime")[target]
        .mean()
        .reindex([1, 2, 3, 4])
        .reset_index()
    )

    study_labels = {
        1: "Below 2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: "Above 10 hours",
    }

    study_chart["Study Time"] = (
        study_chart["studytime"].map(study_labels)
    )

    fig_study = px.bar(
        study_chart,
        x="Study Time",
        y=target,
        color="Study Time",
        color_discrete_sequence=[
            PURPLE,
            BLUE,
            TEAL,
            GREEN,
        ],
        text=study_chart[target].map(
            lambda value: f"{value:.1f}"
        ),
    )

    fig_study.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_study.update_yaxes(
        title="Average Final Grade",
        range=[0, 20],
    )

    fig_study.update_xaxes(
        title="Weekly Study Time Category"
    )

    fig_study.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plot(fig_study),
        use_container_width=True,
    )


# Pass versus below passing grade
with right_chart:
    st.subheader("Pass vs Below Passing Grade")

    result_chart = pd.DataFrame(
        {
            "Result": [
                "Pass",
                "Below Passing Grade",
            ],
            "Students": [
                pass_count,
                fail_count,
            ],
        }
    )

    fig_pass = px.bar(
        result_chart,
        x="Result",
        y="Students",
        color="Result",
        color_discrete_map={
            "Pass": GREEN,
            "Below Passing Grade": RED,
        },
        text="Students",
    )

    fig_pass.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_pass.update_yaxes(
        title="Number of Students",
        rangemode="tozero",
    )

    fig_pass.update_xaxes(
        title="Student Result"
    )

    fig_pass.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plot(fig_pass),
        use_container_width=True,
    )


st.caption(
    "A grade of 10 or higher is counted as a pass for this project."
)

st.divider()


# ==================================================
# 10. FINAL GRADE DISTRIBUTION
# ==================================================
st.subheader("📈 Final Grade Distribution")

grade_distribution = (
    df[target]
    .value_counts()
    .sort_index()
    .rename_axis("Final Grade")
    .reset_index(name="Number of Students")
)

fig_distribution = px.bar(
    grade_distribution,
    x="Final Grade",
    y="Number of Students",
    color="Number of Students",
    color_continuous_scale=[
        "#312E81",
        PURPLE,
        TEAL,
    ],
    text="Number of Students",
)

fig_distribution.update_traces(
    textposition="outside",
    texttemplate="%{text}",
    cliponaxis=False,
)

fig_distribution.update_xaxes(
    title="Final Grade (0–20)",
    dtick=1,
)

fig_distribution.update_yaxes(
    title="Number of Students",
    rangemode="tozero",
)

fig_distribution.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(
    style_plot(fig_distribution, height=390),
    use_container_width=True,
)


# ==================================================
# 11. MACHINE LEARNING MODEL COMPARISON
# ==================================================
st.divider()

st.header("🤖 Machine Learning Model Comparison")

st.write(
    "Three regression models are trained and evaluated "
    "using the same training and testing records."
)

display_results = model_results.copy()

for metric in ["MAE", "MSE", "R² Score"]:
    display_results[metric] = (
        display_results[metric].round(3)
    )

st.dataframe(
    display_results,
    hide_index=True,
    use_container_width=True,
)

model_colors = {
    "Linear Regression": PURPLE,
    "Decision Tree": BLUE,
    "Random Forest": TEAL,
}

metric_col1, metric_col2 = st.columns(2)


# MAE comparison
with metric_col1:
    st.subheader("Compare MAE Scores")

    mae_fig = px.bar(
        model_results,
        x="Model",
        y="MAE",
        color="Model",
        color_discrete_map=model_colors,
        text=model_results["MAE"].map(
            lambda value: f"{value:.2f}"
        ),
    )

    mae_fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    mae_fig.update_yaxes(
        title="Mean Absolute Error (lower is better)",
        rangemode="tozero",
    )

    mae_fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plot(mae_fig),
        use_container_width=True,
    )


# R² comparison
with metric_col2:
    st.subheader("Compare R² Scores")

    r2_fig = px.bar(
        model_results,
        x="Model",
        y="R² Score",
        color="Model",
        color_discrete_map=model_colors,
        text=model_results["R² Score"].map(
            lambda value: f"{value:.3f}"
        ),
    )

    r2_fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    r2_fig.update_yaxes(
        title="R² Score (higher is generally better)"
    )

    r2_fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plot(r2_fig),
        use_container_width=True,
    )


st.info(
    f"Lowest-MAE model on this test split: "
    f"**{best_model_name}**. "
    "This is a test-set comparison, not a guarantee "
    "of future performance."
)

with st.expander("What do the evaluation metrics mean?"):
    st.markdown(
        """
        - **MAE:** Average absolute difference between
          actual and predicted grades. Lower is better.
        - **MSE:** Average squared prediction error.
          Lower is better.
        - **R² Score:** Measures how much variation in
          the test grades is explained by the model.
          Higher is generally better; it can be negative.
        """
    )


# ==================================================
# 12. INDIVIDUAL STUDENT PREDICTION
# ==================================================
st.divider()

st.header("🎯 Predict an Individual Student's Grade")

st.write(
    "Enter the student's study time, absences, "
    "and previous grades."
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
            4: "4 - More than 10 hours",
        }[value],
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )


with input_col2:
    g1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
    )

    g2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
    )


prediction_model_name = st.selectbox(
    "Choose Prediction Model",
    options=list(trained_models.keys()),
    index=0,
)


# ==================================================
# 13. PREDICT AND SHOW RESULT CHART
# ==================================================
if st.button(
    "🔮 Predict Final Grade",
    type="primary",
    use_container_width=True,
):
    input_data = pd.DataFrame(
        [[studytime, absences, g1, g2]],
        columns=features,
    )

    selected_model = trained_models[
        prediction_model_name
    ]

    raw_prediction = float(
        selected_model.predict(input_data)[0]
    )

    # Keep the displayed prediction within 0–20.
    predicted_grade = max(
        0.0,
        min(20.0, raw_prediction),
    )

    predicted_percentage = (
        predicted_grade / 20 * 100
    )

    st.subheader("✨ Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Predicted Final Grade",
        f"{predicted_grade:.2f} / 20",
    )

    result_col2.metric(
        "Predicted Percentage",
        f"{predicted_percentage:.1f}%",
    )

    result_col3.metric(
        "Prediction Model",
        prediction_model_name,
    )

    if predicted_grade >= 10:
        st.success("✅ Predicted result: Pass")
    else:
        st.warning(
            "⚠️ Predicted result: Below passing grade"
        )

    if raw_prediction < 0 or raw_prediction > 20:
        st.caption(
            "The raw model output was outside 0–20 "
            "and has been bounded for display."
        )

    # ----------------------------------------------
    # NEW: PROFESSIONAL PREDICTION CHART
    # ----------------------------------------------
    st.markdown("### 📊 Previous Grades vs Predicted Grade")

    prediction_chart_data = pd.DataFrame(
        {
            "Grade Type": [
                "First Period (G1)",
                "Second Period (G2)",
                "Predicted Final (G3)",
            ],
            "Grade": [
                float(g1),
                float(g2),
                predicted_grade,
            ],
        }
    )

    prediction_fig = go.Figure()

    prediction_fig.add_trace(
        go.Bar(
            x=prediction_chart_data["Grade Type"],
            y=prediction_chart_data["Grade"],
            marker=dict(
                color=[
                    BLUE,
                    PURPLE,
                    TEAL,
                ],
                line=dict(
                    color="#FFFFFF",
                    width=0,
                ),
            ),
            text=[
                f"{value:.1f}/20"
                for value in prediction_chart_data["Grade"]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Grade: %{y:.2f}/20"
                "<extra></extra>"
            ),
        )
    )

    # Passing threshold line
    prediction_fig.add_hline(
        y=10,
        line_dash="dash",
        line_color=ORANGE,
        line_width=2,
        annotation_text="Pass threshold: 10/20",
        annotation_position="top left",
        annotation_font_color=ORANGE,
    )

    prediction_fig.update_yaxes(
        title="Grade (0–20)",
        range=[0, 22],
        dtick=2,
    )

    prediction_fig.update_xaxes(
        title="Grade comparison"
    )

    prediction_fig.update_layout(
        showlegend=False,
    )

    st.plotly_chart(
        style_plot(prediction_fig, height=390),
        use_container_width=True,
    )

    st.caption(
        "This chart compares the student's two previous "
        "period grades with the model's estimated final "
        "grade. The dashed line marks the project's "
        "pass threshold."
    )

    # ----------------------------------------------
    # DOWNLOAD INDIVIDUAL PREDICTION REPORT
    # ----------------------------------------------
    prediction_report = pd.DataFrame(
        {
            "Study Time Category": [studytime],
            "Absences": [absences],
            "First Period Grade": [g1],
            "Second Period Grade": [g2],
            "Prediction Model": [
                prediction_model_name
            ],
            "Predicted Final Grade": [
                round(predicted_grade, 2)
            ],
            "Predicted Percentage": [
                round(predicted_percentage, 2)
            ],
            "Predicted Result": [
                "Pass"
                if predicted_grade >= 10
                else "Below passing grade"
            ],
        }
    )

    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=prediction_report.to_csv(index=False),
        file_name="student_prediction_report.csv",
        mime="text/csv",
    )

    st.caption(
        "Predictions are estimates based on historical data "
        "and are not guaranteed academic outcomes."
    )


# ==================================================
# 14. DOWNLOAD DATASET AND MODEL RESULTS
# ==================================================
st.divider()

st.subheader("📥 Download Project Reports")

download_col1, download_col2 = st.columns(2)


with download_col1:
    st.download_button(
        label="Download Student Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="student_performance_dataset.csv",
        mime="text/csv",
    )


with download_col2:
    st.download_button(
        label="Download Model Comparison (CSV)",
        data=model_results.to_csv(index=False),
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )


# ==================================================
# 15. FOOTER
# ==================================================
st.divider()

st.caption(
    "Student Performance Analysis, Comparison and "
    "Prediction Using Statistical Machine Learning"
)

st.caption(
    "Model predictions are educational estimates and "
    "should not be treated as guaranteed outcomes."
)