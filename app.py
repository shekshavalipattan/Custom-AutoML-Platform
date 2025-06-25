import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from io import StringIO

from utils.data_handler import DataHandler
from utils.model_trainer import ModelTrainer
from utils.model_evaluator import ModelEvaluator
from utils.model_explainer import ModelExplainer
from utils.visualizer import Visualizer

# Set page configuration
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve visibility and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5 !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #424242 !important;
        margin-bottom: 1rem !important;
    }
    .section-header {
        font-size: 1.5rem !important;
        color: #1E88E5 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #E0E0E0 !important;
    }
    .description {
        font-size: 1.1rem !important;
        color: #424242 !important;
        margin-bottom: 2rem !important;
        background-color: #F5F5F5 !important;
        padding: 1rem !important;
        border-radius: 5px !important;
        border-left: 3px solid #1E88E5 !important;
    }
    .stButton > button {
        background-color: #1E88E5 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        border-radius: 5px !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #1565C0 !important;
    }
    div[data-testid="stSidebarNav"] li {
        padding: 0.75rem 0 !important;
    }
    div[data-testid="stSidebarNav"] p {
        font-size: 1.1rem !important;
    }
    .stRadio > div {
        background-color: #F5F5F5 !important; 
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .success-box {
        background-color: #E8F5E9 !important;
        padding: 1rem !important;
        border-radius: 5px !important;
        border-left: 3px solid #43A047 !important;
        margin-bottom: 1rem !important;
    }
    .info-box {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 5px !important;
        border-left: 3px solid #1E88E5 !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Main header
st.markdown("<h1 class='main-header'>🤖 AutoML Platform</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='description'>This platform automatically processes your data, trains multiple models, and helps you select the best one. Similar to H2O AutoML, it handles the heavy lifting of machine learning so you can focus on insights.</div>",
    unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #4FC3F7;'>Navigation</h2>", unsafe_allow_html=True)

# Add some spacing and styling to the sidebar
st.sidebar.markdown("""
<style>
    [data-testid='stSidebar'] {
        padding: 1rem;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    div[role='radiogroup'] > div {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.1);
        transition: all 0.2s;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    div[role='radiogroup'] > div:hover {
        background-color: rgba(49, 120, 198, 0.3);
        border-color: rgba(49, 120, 198, 0.8);
    }
    div[role='radiogroup'] label {
        font-weight: 500;
        color: #ffffff !important;
    }

    /* Make sure text is visible in both light and dark modes */
    .st-emotion-cache-16txtl3 {
        color: rgba(250, 250, 250, 0.9) !important;
    }

    /* Navigation item styling with better contrast */
    .stRadio label {
        color: #ffffff !important;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

pages = ["Upload Data", "Train Models", "Evaluate Models", "Visualize Results", "Model Explainability", "Export Model"]
icons = ["📁", "🧠", "📊", "📈", "🔍", "💾"]

page = st.sidebar.radio("",
                        pages,
                        format_func=lambda x: f"{icons[pages.index(x)]} {x}")

# Add a sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #B0BEC5; font-size: 0.8rem;'>© 2025 AutoML Platform</div>",
                    unsafe_allow_html=True)

# Upload Data Page
if page == "Upload Data":
    st.markdown("<h2 class='sub-header'>📁 Upload Data</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Upload your dataset (CSV or Excel) to get started. The platform will automatically analyze your data 
    and prepare it for model training.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.session_state.data = data

            st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns")

            # Display data preview
            st.subheader("Data Preview:")
            st.dataframe(data.head())

            # Display basic statistics
            st.subheader("Basic Statistics:")
            st.write(data.describe())

            # Data info
            st.subheader("Data Types:")
            buffer = StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            # Check for missing values
            st.subheader("Missing Values:")
            missing_values = data.isnull().sum()
            missing_percent = (missing_values / len(data)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage': missing_percent
            })
            st.write(missing_df[missing_df['Missing Values'] > 0])

            # Data handler for further processing
            data_handler = DataHandler()

            # Identify potential target columns (numerical or categorical with few unique values)
            potential_targets = data_handler.identify_potential_targets(data)

            # Ask user to select target column and problem type
            st.subheader("Model Configuration:")
            target_column = st.selectbox("Select Target Column:", potential_targets)

            # Determine if classification or regression
            if target_column:
                unique_values = data[target_column].nunique()
                if unique_values < 10:  # Heuristic: if fewer than 10 unique values, likely classification
                    default_problem = "Classification"
                else:
                    default_problem = "Regression"

                problem_type = st.radio("Problem Type:", ["Classification", "Regression"],
                                        index=0 if default_problem == "Classification" else 1)

                # Save selections to session state
                st.session_state.target_column = target_column
                st.session_state.problem_type = problem_type

                st.success(f"Target column set to '{target_column}' for {problem_type.lower()} problem")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Train Models Page
elif page == "Train Models":
    st.markdown("<h2 class='sub-header'>🧠 Train Models</h2>", unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload data first.")
    elif st.session_state.target_column is None or st.session_state.problem_type is None:
        st.warning("Please select a target column and problem type on the Upload Data page.")
    else:
        st.write(
            f"Training models for **{st.session_state.problem_type}** problem with target: **{st.session_state.target_column}**")

        # Model configuration options
        st.subheader("Training Configuration:")

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

        with col2:
            time_limit = st.slider("Training Time Limit (seconds)", 30, 600, 120)
            include_preprocessing = st.checkbox("Automatic Data Preprocessing", value=True, key="preprocessing_option")

        model_options = {
            "Classification": {
                "LogisticRegression": st.checkbox("Logistic Regression", value=True, key="cls_logreg"),
                "RandomForest": st.checkbox("Random Forest", value=True, key="cls_rf"),
                "GradientBoosting": st.checkbox("Gradient Boosting", value=True, key="cls_gb"),
                "SVM": st.checkbox("Support Vector Machine", value=False, key="cls_svm"),
                "KNN": st.checkbox("K-Nearest Neighbors", value=False, key="cls_knn"),
                "DecisionTree": st.checkbox("Decision Tree", value=True, key="cls_dt"),
                "XGBoost": st.checkbox("XGBoost", value=True, key="cls_xgb")
            },
            "Regression": {
                "LinearRegression": st.checkbox("Linear Regression", value=True, key="reg_linreg"),
                "RandomForest": st.checkbox("Random Forest", value=True, key="reg_rf"),
                "GradientBoosting": st.checkbox("Gradient Boosting", value=True, key="reg_gb"),
                "SVR": st.checkbox("Support Vector Regression", value=False, key="reg_svr"),
                "KNN": st.checkbox("K-Nearest Neighbors", value=False, key="reg_knn"),
                "DecisionTree": st.checkbox("Decision Tree", value=True, key="reg_dt"),
                "XGBoost": st.checkbox("XGBoost", value=True, key="reg_xgb")
            }
        }

        selected_models = [model for model, selected in model_options[st.session_state.problem_type].items() if
                           selected]

        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            if st.button("Start Training"):
                with st.spinner("Training models - this may take a while..."):
                    # Initialize model trainer
                    trainer = ModelTrainer(
                        problem_type=st.session_state.problem_type,
                        cv_folds=cv_folds,
                        test_size=test_size,
                        time_limit=time_limit
                    )

                    # Prepare data
                    data_handler = DataHandler()
                    X, y, preprocessor = data_handler.preprocess_data(
                        st.session_state.data,
                        st.session_state.target_column,
                        include_preprocessing=include_preprocessing
                    )

                    # Store preprocessor
                    st.session_state.preprocessor = preprocessor

                    # Train models
                    model_results = trainer.train_models(X, y, selected_models)

                    # Store models and results
                    st.session_state.models = model_results["models"]
                    st.session_state.evaluation_results = model_results["evaluation"]

                    # Find best model
                    evaluator = ModelEvaluator(st.session_state.problem_type)
                    best_model_name = evaluator.get_best_model(model_results["evaluation"])
                    st.session_state.best_model = {
                        "name": best_model_name,
                        "model": model_results["models"][best_model_name]
                    }

                    # Generate feature importance
                    explainer = ModelExplainer()
                    feature_importance = explainer.get_feature_importance(
                        X,
                        st.session_state.models,
                        data_handler.get_feature_names(st.session_state.data, st.session_state.target_column)
                    )
                    st.session_state.feature_importance = feature_importance

                st.markdown(
                    "<div class='success-box'>✅ Training complete! Navigate to the Evaluate Models page to see results.</div>",
                    unsafe_allow_html=True)

# Evaluate Models Page
elif page == "Evaluate Models":
    st.markdown("<h2 class='sub-header'>📊 Evaluate Models</h2>", unsafe_allow_html=True)

    if st.session_state.evaluation_results is None:
        st.warning("Please train models first.")
    else:
        st.subheader("Model Performance Comparison")

        # Extract evaluation metrics
        eval_results = st.session_state.evaluation_results

        # Create a DataFrame for display
        results_df = pd.DataFrame(eval_results).T.reset_index()
        results_df.columns = ['Model'] + list(results_df.columns[1:])

        # Create two separate dataframes - one for normalized metrics and one for raw metrics
        norm_cols = ['Model'] + [col for col in results_df.columns[1:] if not col.startswith('raw_')]
        raw_cols = ['Model'] + [col for col in results_df.columns[1:] if col.startswith('raw_')]

        norm_df = results_df[norm_cols].copy()
        raw_df = results_df[raw_cols].copy()

        # Format normalized metrics as percentages for display
        for col in norm_df.columns:
            if col != 'Model':
                norm_df[col] = norm_df[col] * 100
                # Rename columns to show they are percentages
                new_col = col + ' (%)' if not col.endswith('(%)') else col
                norm_df.rename(columns={col: new_col}, inplace=True)

        # Clean up raw metrics column names by removing 'raw_' prefix
        for col in raw_df.columns:
            if col.startswith('raw_'):
                raw_df.rename(columns={col: col[4:]}, inplace=True)  # Remove 'raw_' prefix

        # Display normalized metrics with percentage formatting
        st.subheader("Normalized Performance Metrics (%)")
        st.write("Higher values indicate better performance. All metrics are normalized to a 0-100% scale.")
        st.dataframe(norm_df.style.format("{:.2f}%", subset=[col for col in norm_df.columns if col != 'Model'])
                     .highlight_max(subset=[col for col in norm_df.columns if col != 'Model'], axis=0))

        # Add option to view raw metrics
        with st.expander("View Raw Metrics (Actual Values)"):
            st.write("These are the actual metric values before normalization.")
            st.dataframe(raw_df.style.format("{:.4f}", subset=[col for col in raw_df.columns if col != 'Model']))

        # Highlight best model
        st.subheader("Best Model")
        best_model_name = st.session_state.best_model["name"]
        st.success(f"The best performing model is: **{best_model_name}**")

        # Show best model metrics
        best_metrics = eval_results[best_model_name].copy()

        # Format metrics in two sections - normalized (percentage) and raw values
        formatted_metrics = {"Normalized Metrics (%)": {}, "Raw Metrics": {}}

        for key, value in best_metrics.items():
            if key.startswith('raw_'):
                # Format raw values with appropriate precision
                if isinstance(value, float):
                    # Strip 'raw_' prefix for display
                    metric_name = key[4:]  # Remove 'raw_' prefix
                    formatted_metrics["Raw Metrics"][metric_name] = round(value, 4)
            else:
                # Convert normalized scores to percentages
                formatted_metrics["Normalized Metrics (%)"][key] = round(value * 100, 2)

        # Display the metrics in an expandable section
        with st.expander("Detailed Metrics for Best Model", expanded=True):
            # Show normalized metrics
            st.subheader("Normalized Metrics (higher is better)")
            st.write("These metrics are normalized to a 0-100% scale for easy comparison.")
            st.json(formatted_metrics["Normalized Metrics (%)"])

            # Show raw metrics
            st.subheader("Raw Metrics")
            st.write("These are the actual metric values before normalization.")
            st.json(formatted_metrics["Raw Metrics"])

        # Display detailed metrics for selected model
        st.subheader("Detailed Model Evaluation")
        selected_model = st.selectbox("Select Model for Detailed Analysis:", list(eval_results.keys()))

        if selected_model:
            st.write(f"### {selected_model} Evaluation")

            # Initialize evaluator
            evaluator = ModelEvaluator(st.session_state.problem_type)

            # Get data
            data_handler = DataHandler()
            X, y, _ = data_handler.preprocess_data(
                st.session_state.data,
                st.session_state.target_column,
                preprocessor=st.session_state.preprocessor
            )

            # Show confusion matrix for classification
            if st.session_state.problem_type == "Classification":
                st.write("#### Confusion Matrix")
                confusion_matrix = evaluator.plot_confusion_matrix(
                    st.session_state.models[selected_model],
                    X,
                    y
                )
                st.pyplot(confusion_matrix)

                st.write("#### ROC Curve")
                roc_curve = evaluator.plot_roc_curve(
                    st.session_state.models[selected_model],
                    X,
                    y
                )
                st.pyplot(roc_curve)

            # Show residual plots for regression
            else:
                st.write("#### Residual Plot")
                residual_plot = evaluator.plot_residuals(
                    st.session_state.models[selected_model],
                    X,
                    y
                )
                st.pyplot(residual_plot)

                st.write("#### Prediction vs Actual")
                pred_vs_actual = evaluator.plot_prediction_vs_actual(
                    st.session_state.models[selected_model],
                    X,
                    y
                )
                st.pyplot(pred_vs_actual)

# Visualize Results Page
elif page == "Visualize Results":
    st.markdown("<h2 class='sub-header'>📈 Visualize Results</h2>", unsafe_allow_html=True)

    if st.session_state.evaluation_results is None:
        st.warning("Please train models first.")
    else:
        visualizer = Visualizer()

        # Metrics comparison
        st.subheader("Model Metrics Comparison")

        # Display performance metrics
        metrics_fig = visualizer.plot_metrics_comparison(st.session_state.evaluation_results,
                                                         st.session_state.problem_type)
        st.plotly_chart(metrics_fig)

        # Feature Importance
        if st.session_state.feature_importance is not None:
            st.subheader("Feature Importance")

            # Select model for feature importance
            model_names = list(st.session_state.feature_importance.keys())
            selected_model = st.selectbox("Select Model:", model_names)

            if selected_model:
                feature_imp_fig = visualizer.plot_feature_importance(
                    st.session_state.feature_importance[selected_model]
                )
                st.plotly_chart(feature_imp_fig)

        # Get data for additional visualizations
        if st.session_state.data is not None:
            st.subheader("Data Distribution")

            # Select columns for visualization
            numeric_columns = st.session_state.data.select_dtypes(include=np.number).columns.tolist()
            selected_columns = st.multiselect("Select columns to visualize:", numeric_columns)

            if selected_columns:
                # Distribution plot
                dist_fig = visualizer.plot_distributions(st.session_state.data, selected_columns)
                st.plotly_chart(dist_fig)

                # Correlation plot
                st.subheader("Feature Correlation")
                corr_fig = visualizer.plot_correlation(st.session_state.data[selected_columns])
                st.plotly_chart(corr_fig)

# Model Explainability Page
elif page == "Model Explainability":
    st.markdown("<h2 class='sub-header'>🔍 Model Explainability</h2>", unsafe_allow_html=True)

    if st.session_state.evaluation_results is None or st.session_state.models == {}:
        st.warning("Please train models first.")
    else:
        st.subheader("Understanding Model Predictions")

        explainer = ModelExplainer()

        # Select model
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model to Explain:", model_names)

        if selected_model:
            # Get data
            data_handler = DataHandler()
            X, y, _ = data_handler.preprocess_data(
                st.session_state.data,
                st.session_state.target_column,
                preprocessor=st.session_state.preprocessor
            )

            feature_names = data_handler.get_feature_names(st.session_state.data, st.session_state.target_column)

            # Partial Dependence Plots
            st.write("#### Partial Dependence Plots")
            st.write(
                "These plots show how the prediction changes when a feature varies, with all other features held constant.")

            # Select feature for PDP
            selected_feature = st.selectbox("Select feature for analysis:", feature_names)

            if selected_feature:
                feature_idx = feature_names.index(selected_feature)

                pdp_fig = explainer.plot_partial_dependence(
                    st.session_state.models[selected_model],
                    X,
                    feature_idx,
                    feature_name=selected_feature
                )
                st.pyplot(pdp_fig)

            # Sample-based explanation
            st.write("#### Sample Explanation")
            st.write("Explain prediction for a specific data point")

            # Select a sample to explain
            sample_index = st.number_input("Select a row index from your data:",
                                           min_value=0,
                                           max_value=len(X) - 1 if len(X) > 0 else 0,
                                           value=0)

            if st.button("Explain this sample"):
                try:
                    explanation = explainer.explain_prediction(
                        st.session_state.models[selected_model],
                        X[sample_index:sample_index + 1],
                        feature_names
                    )

                    # Display the sample data
                    st.write("Sample data:")
                    sample_df = pd.DataFrame([dict(zip(feature_names, X[sample_index]))])
                    st.dataframe(sample_df)

                    # Display the explanation
                    exp_fig = explainer.plot_explanation(explanation, feature_names)
                    st.pyplot(exp_fig)

                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")

# Export Model Page
elif page == "Export Model":
    st.markdown("<h2 class='sub-header'>💾 Export Model</h2>", unsafe_allow_html=True)

    if st.session_state.evaluation_results is None or st.session_state.models == {}:
        st.warning("Please train models first.")
    else:
        st.subheader("Export Trained Model")

        # Select model to export
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model to Export:", model_names)

        if selected_model:
            model_to_export = st.session_state.models[selected_model]

            # Export options
            export_format = st.radio("Export Format:", ["Joblib", "Pickle"])

            if st.button("Export Model"):
                with st.spinner("Preparing model for export..."):
                    # Create an export package with model and preprocessor
                    export_package = {
                        'model': model_to_export,
                        'preprocessor': st.session_state.preprocessor,
                        'target_column': st.session_state.target_column,
                        'problem_type': st.session_state.problem_type,
                        'export_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'feature_names': DataHandler().get_feature_names(
                            st.session_state.data,
                            st.session_state.target_column
                        )
                    }

                    # Export the package
                    if export_format == "Joblib":
                        import joblib
                        from io import BytesIO

                        buffer = BytesIO()
                        joblib.dump(export_package, buffer)
                        model_bytes = buffer.getvalue()
                        filename = f"{selected_model.lower().replace(' ', '_')}_model.joblib"
                    else:
                        import pickle

                        model_bytes = pickle.dumps(export_package)
                        filename = f"{selected_model.lower().replace(' ', '_')}_model.pkl"

                # Provide download link
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=filename,
                    mime="application/octet-stream"
                )

                # Usage instructions
                st.subheader("How to use the exported model")
                st.code(f"""
# Code to load and use the exported model
import joblib  # or import pickle

# Load the model package
model_package = joblib.load('{filename}')  # or pickle.load(open('{filename}', 'rb'))

# Extract components
model = model_package['model']
preprocessor = model_package['preprocessor']
feature_names = model_package['feature_names']

# Preprocess new data (assuming df is your new data)
# Make sure it has the same features as your training data
X_new = preprocessor.transform(df)

# Make predictions
predictions = model.predict(X_new)
                """, language="python")

# Show footer
st.markdown("---")
st.markdown("### AutoML Platform | Made with Streamlit")
