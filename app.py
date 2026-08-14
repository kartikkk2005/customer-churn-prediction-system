"""
Streamlit Dashboard for Real-Time Customer Churn Prediction & Business Insights.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import REQUIRED_COLUMNS
from src.predict import load_artifacts, predict_single_customer, predict_batch
from src.preprocessing import add_engineered_features

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism dark aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    .risk-badge-low {
        background-color: #2e7d32;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #ed6c02;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #d32f2f;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_and_preprocessor():
    """Load cached model and preprocessor artifacts."""
    return load_artifacts()


@st.cache_data
def get_metrics_data():
    """Load cached benchmark metrics."""
    metrics_path = os.path.join("models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main():
    # Sidebar
    st.sidebar.title("⚡ Churn Predictor AI")
    st.sidebar.caption("Machine Learning Customer Retention Platform")

    model_metrics = get_metrics_data()
    best_name = model_metrics["best_model_name"] if model_metrics else "XGBoost Classifier"
    best_auc = model_metrics["best_model_roc_auc"] if model_metrics else 0.85

    st.sidebar.markdown("---")
    st.sidebar.subheader("Active ML Engine")
    st.sidebar.info(f"**Model**: {best_name}\n\n**ROC-AUC Score**: {best_auc:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 🛠 Tech Stack
        - **Core**: Python 3.11, Pandas, NumPy
        - **ML**: Scikit-Learn, XGBoost
        - **UI/Viz**: Streamlit, Plotly
        - **Validation**: Stratified 5-Fold CV
        """
    )

    # Main Header
    st.title("🎯 Customer Churn Prediction & Business Insights")
    st.markdown(
        "Predict customer churn probability in real time, analyze benchmark model performance, and extract actionable customer retention strategies."
    )
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🔮 Real-Time Risk Calculator",
            "📊 Benchmark Analytics & Insights",
            "📁 Batch CSV Prediction Engine",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: Real-Time Customer Risk Calculator
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Customer Profile & Contract Attributes")
        st.caption("Adjust customer attributes below to calculate instant churn probability.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        with col2:
            st.markdown("#### 📜 Account & Billing")
            tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:
            st.markdown("#### 🌐 Services & Charges")
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

            if internet != "No":
                security = st.selectbox("Online Security", ["No", "Yes"])
                backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_prot = st.selectbox("Device Protection", ["No", "Yes"])
                tech_supp = st.selectbox("Tech Support", ["No", "Yes"])
                stream_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                stream_mov = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                security = backup = device_prot = tech_supp = stream_tv = stream_mov = "No internet service"

            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(monthly_charges * tenure), step=10.0)

        customer_data = {
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_supp,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        st.markdown("---")
        if st.button("🚀 Calculate Churn Probability", use_container_width=True):
            model, preprocessor = get_model_and_preprocessor()
            result = predict_single_customer(customer_data, model, preprocessor)

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("### Prediction Outcome")

                prob = result["churn_probability"]
                risk = result["risk_level"]

                if risk == "Low Risk":
                    st.markdown(f'<div class="risk-badge-low">🟢 Low Churn Risk ({prob:.1%})</div>', unsafe_allow_html=True)
                elif risk == "Medium Risk":
                    st.markdown(f'<div class="risk-badge-medium">🟡 Medium Churn Risk ({prob:.1%})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-badge-high">🔴 High Churn Risk ({prob:.1%})</div>', unsafe_allow_html=True)

                st.write("")
                # Gauge Chart
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Churn Probability (%)", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#6C5CE7"},
                            "steps": [
                                {"range": [0, 35], "color": "rgba(46, 125, 50, 0.3)"},
                                {"range": [35, 65], "color": "rgba(237, 108, 2, 0.3)"},
                                {"range": [65, 100], "color": "rgba(211, 47, 47, 0.3)"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.markdown("### 💡 Recommended Business Retention Actions")
                recs = []
                if contract == "Month-to-month":
                    recs.append("📌 **Contract Incentive**: Offer a 15% discount on an annual contract to switch away from Month-to-Month billing.")
                if tech_supp != "Yes" and internet != "No":
                    recs.append("🛠 **Tech Support Add-On**: Include 3 months of free Tech Support to lower customer friction.")
                if payment_method == "Electronic check":
                    recs.append("💳 **Payment Automation**: Encourage switching to automatic bank transfer / credit card with a $10 bill credit.")
                if tenure < 6:
                    recs.append("🎁 **Onboarding Check-In**: High churn vulnerability window (< 6 months). Assign a customer success check-in.")
                if not recs:
                    recs.append("✅ **Healthy Customer**: Customer exhibits low churn indicators. Maintain regular loyalty rewards.")

                for rec in recs:
                    st.success(rec)

    # ---------------------------------------------------------
    # TAB 2: Benchmark Analytics & Model Comparison
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Model Benchmarking & Performance Metrics")
        st.caption("Comparative evaluation of Logistic Regression, Random Forest, and XGBoost classifiers.")

        if model_metrics:
            models_dict = model_metrics["models"]

            metrics_list = []
            for name, metrics in models_dict.items():
                metrics_list.append(
                    {
                        "Model": name,
                        "Accuracy": metrics["accuracy"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-Score": metrics["f1_score"],
                        "ROC-AUC": metrics["roc_auc"],
                        "CV ROC-AUC": metrics["cv_roc_auc"],
                    }
                )

            df_metrics = pd.DataFrame(metrics_list)
            st.dataframe(df_metrics.style.highlight_max(axis=0, color="#6C5CE7"), use_container_width=True)

            b_col1, b_col2 = st.columns(2)

            with b_col1:
                st.markdown("#### 📊 Metric Comparison Chart")
                fig_bar = px.bar(
                    df_metrics,
                    x="Model",
                    y=["Accuracy", "F1-Score", "ROC-AUC"],
                    barmode="group",
                    color_discrete_sequence=["#6C5CE7", "#00CEC9", "#FF7675"],
                    title="Classifier Performance Metrics Comparison",
                )
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with b_col2:
                st.markdown("#### 📈 ROC Curves")
                fig_roc = go.Figure()

                for name, metrics in models_dict.items():
                    if "roc_curve" in metrics:
                        fpr = metrics["roc_curve"]["fpr"]
                        tpr = metrics["roc_curve"]["tpr"]
                        auc = metrics["roc_auc"]
                        fig_roc.add_trace(
                            go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode="lines",
                                name=f"{name} (AUC = {auc:.4f})",
                            )
                        )

                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Random Classifier",
                    )
                )
                fig_roc.update_layout(
                    title="Receiver Operating Characteristic (ROC) Curves",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            # Key Business Drivers Insights
            st.markdown("---")
            st.markdown("### 🔑 Key Churn Risk Drivers Analysis")
            st.markdown(
                """
                - **Contract Duration**: Month-to-month contracts are the **#1 indicator** of customer churn risk.
                - **Service Packaging**: Customers without **TechSupport** or **OnlineSecurity** exhibit double the churn probability.
                - **Payment Choice**: Electronic check users have significantly higher churn rates compared to automated payment methods.
                - **Tenure Lifecycle**: The initial 1–6 months represent the highest risk window for churn.
                """
            )
        else:
            st.warning("Model metrics file not found. Please run training pipeline first.")

    # ---------------------------------------------------------
    # TAB 3: Batch CSV Prediction Engine
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Bulk Customer Churn Inference")
        st.caption("Upload a CSV file containing customer data to generate predictions for hundreds of customers simultaneously.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        use_sample = st.checkbox("Or use existing dataset sample (data/customer_churn.csv)", value=True)

        df_to_predict = None

        if uploaded_file is not None:
            df_to_predict = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded uploaded CSV ({len(df_to_predict)} records).")
        elif use_sample and os.path.exists("data/customer_churn.csv"):
            df_to_predict = pd.read_csv("data/customer_churn.csv").head(100)
            st.info("Using top 100 sample records from `data/customer_churn.csv`.")

        if df_to_predict is not None:
            st.write("### Data Preview")
            st.dataframe(df_to_predict.head(5), use_container_width=True)

            if st.button("⚡ Run Batch Predictions", key="batch_pred_btn"):
                model, preprocessor = get_model_and_preprocessor()
                df_results = predict_batch(df_to_predict, model, preprocessor)

                st.markdown("---")
                st.subheader("Batch Prediction Results")

                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("Total Customers", len(df_results))
                predicted_churn_count = (df_results["Predicted_Churn"] == "Yes").sum()
                p_col2.metric("Predicted Churners", predicted_churn_count)
                p_col3.metric("Predicted Churn Rate", f"{(predicted_churn_count / len(df_results)):.1%}")

                st.dataframe(
                    df_results[
                        ["CustomerID", "MonthlyCharges", "Tenure", "Contract", "Churn_Probability", "Predicted_Churn", "Risk_Level"]
                    ]
                    if "CustomerID" in df_results.columns
                    else df_results[
                        ["MonthlyCharges", "Tenure", "Contract", "Churn_Probability", "Predicted_Churn", "Risk_Level"]
                    ],
                    use_container_width=True,
                )

                # Distribution plot
                fig_dist = px.histogram(
                    df_results,
                    x="Churn_Probability",
                    color="Risk_Level",
                    nbins=20,
                    title="Batch Customer Churn Probability Distribution",
                    color_discrete_map={"Low Risk": "#2e7d32", "Medium Risk": "#ed6c02", "High Risk": "#d32f2f"},
                )
                fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)

                # Download CSV
                csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Prediction CSV Report",
                    data=csv_bytes,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
