import streamlit as st
import modeva
from modeva.utils.authenticate import authenticate
from modeva import Dataset
from modeva.models import MoLGBMRegressor
from modeva import TestSuite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # noqa: F401


st.set_page_config(page_title="QuCreate Streamlit Lab - Bike Share Predictive Modeler", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab - Bike Share Predictive Modeler")
st.divider()

# Authentication (as per the notebook - include token directly for self-sufficiency)
try:
    authenticate(token='eaaa4301-b140-484c-8e93-f9f633c8bacb')
    st.success("Modeva Authentication Successful!", icon="✅")
except Exception as e:
    st.error(f"Modeva Authentication Failed: {e}", icon="🚨")
    st.stop()

st.header("Bike Share Demand Prediction", divider='blue')

st.markdown("""
    **Objective:** To predict bike rental demand using the Bikesharing dataset. 
    This application demonstrates data preprocessing, exploratory data analysis, model training, and evaluation using the Modeva library and Streamlit.
    """)

# Initialize DataSet
ds = Dataset()

# Load BikeSharing dataset
try:
    ds.load("BikeSharing")
    st.success("BikeSharing dataset loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Error loading BikeSharing dataset: {e}", icon="🚨")
    st.stop()

st.subheader("1. Dataset Overview", divider='gray')

with st.expander("Show Raw Data and Summary"):
    st.write("### Raw Data Preview")
    st.dataframe(ds.raw_data.head())
    st.write("### Dataset Summary")
    data_summary = ds.summary()
    st.json(data_summary.table["summary"])
    st.write("### Numerical Feature Summary")
    st.dataframe(data_summary.table["numerical"])
    st.write("### Categorical Feature Summary")
    st.dataframe(data_summary.table["categorical"])
    st.markdown("""
        **Dataset Description:**
        This section displays the raw data and descriptive statistics of the BikeSharing dataset. 
        - **Raw Data Preview:** Shows the first few rows of the loaded dataset.
        - **Dataset Summary:** Provides a high-level overview, including data dimensions and feature types.
        - **Numerical Feature Summary:**  Presents statistics for numerical columns like mean, median, standard deviation, etc.
        - **Categorical Feature Summary:** Shows counts and unique values for categorical columns.
        """)

st.subheader("2. Data Preprocessing", divider='gray')

with st.expander("Configure Data Preprocessing"):
    st.write("### Feature Selection")
    all_features = ds.raw_data.columns.tolist()
    default_active_features = [f for f in all_features if f not in ['instant', 'dteday', 'cnt', 'casual', 'registered']] #Exclude target and id-like columns by default
    selected_inactive_features = st.multiselect(
        "Select features to exclude from the model:",
        options=all_features,
        default=[f for f in all_features if f not in default_active_features]
    )

    ds.set_inactive_features(features=selected_inactive_features)
    active_features_exp = ", ".join(ds.active_features) if ds.active_features else "All features are active."
    st.write(f"Active features: {active_features_exp}")
    st.write(f"Inactive features: {', '.join(ds.inactive_features)}")

    st.write("### Preprocessing Steps")
    preprocess_steps_expander = st.expander("View Preprocessing Steps Code", expanded=False)
    with preprocess_steps_expander:
        st.code("""
        ds.reset_preprocess() # Reset any previous preprocessing
        ds.impute_missing() # Handle missing values (default strategy)
        ds.scale_numerical(method="minmax") # Scale numerical features to [0, 1]
        ds.encode_categorical(features=("season", "weathersit", "holiday", "workingday"), method="ordinal") # Encode categorical features ordinally
        ds.bin_numerical(features=("atemp", ), bins=10, method="uniform") # Bin 'atemp' feature
        ds.preprocess() # Execute all defined preprocessing steps
        """, language='python')

    if st.button("Apply Preprocessing"):
        with st.spinner("Applying preprocessing steps..."):
            ds.reset_preprocess()
            ds.impute_missing()
            ds.scale_numerical(method="minmax")
            ds.encode_categorical(features=("season", "weathersit", "holiday", "workingday"), method="ordinal")
            ds.bin_numerical(features=("atemp", ), bins=10, method="uniform")
            ds.preprocess()
        st.success("Preprocessing applied successfully!", icon="✅")

    if st.checkbox("Show Preprocessed Data"):
        st.write("### Preprocessed Data Preview")
        st.dataframe(ds.data.head())
        st.markdown("""
            **Data Preprocessing Explanation:**
            - **Feature Selection:** Users can choose to exclude features that might not be relevant for the model.
            - **Preprocessing Steps:** The following steps are applied to prepare the data:
                - **Reset Preprocessing:** Clears any previously applied preprocessing steps.
                - **Impute Missing Values:** Fills in any missing data points using a default strategy (e.g., mean for numerical, mode for categorical).
                - **Scale Numerical Features:** Scales numerical features using Min-Max scaling to ensure all features contribute equally.
                - **Encode Categorical Features:** Converts categorical features into numerical representations using ordinal encoding.
                - **Bin Numerical Features:** Discretizes the 'atemp' (feeling temperature) feature into bins.
                - **Execute Preprocessing:** Applies all the defined preprocessing steps to the dataset.
            """)


st.subheader("3. Exploratory Data Analysis (EDA)", divider='gray')

with st.expander("Explore Data Visualizations"):
    st.write("### 1D EDA: Distribution of Bike Rentals (cnt)")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(ds.eda_1d(feature="cnt", plot_type="density").plot(figsize=(6,4)))
        st.markdown("Density plot showing the distribution of bike rental counts.")
    with col2:
        st.pyplot(ds.eda_1d(feature="cnt", plot_type="histogram").plot(figsize=(6,4)))
        st.markdown("Histogram showing the frequency distribution of bike rental counts.")

    st.write("### 2D EDA: Bike Rentals vs. Hour of the Day")
    st.pyplot(ds.eda_2d(feature_x="hr", feature_y="cnt", feature_color="yr", sample_size=300).plot(figsize=(8,5)))
    st.markdown("""
        Scatter plot of bike rentals (cnt) against the hour of the day (hr), colored by year (yr). 
        This visualization helps understand the hourly trends in bike rentals across different years.
        """)

    st.write("### 2D EDA: Bike Rentals vs. Season")
    st.pyplot(ds.eda_2d(feature_x="season", feature_y="workingday").plot(figsize=(8,5)))
    st.markdown("""
        Scatter plot of season vs workingday.
        This visualization helps understand the distribution of working days across different seasons.
        """)

    st.write("### 2D EDA: Bike Rentals (cnt) by Season")
    st.pyplot(ds.eda_2d(feature_x="season", feature_y="cnt").plot(figsize=(8,5)))
    st.markdown("Box plot showing the distribution of bike rentals for each season.")

    st.write("### 3D EDA: Bike Rentals vs. Hour and Temperature")
    st.pyplot(ds.eda_3d(feature_x="hr", feature_y="atemp", feature_z="cnt", feature_color="yr", sample_size=300).plot(figsize=(8,6)))
    st.markdown("""
        3D scatter plot of bike rentals (cnt) against hour (hr) and temperature (atemp), colored by year (yr).
        Visualizes the combined effect of hour and temperature on bike rentals.
        """)

    st.write("### Correlation Heatmap")
    eda_correlation_features = ['hr', 'season', 'workingday', 'weathersit', 'windspeed', 'hum', 'cnt']
    st.pyplot(ds.eda_correlation(features=eda_correlation_features).plot(figsize=(8,8)))
    st.markdown("""
        Heatmap showing the correlation matrix between selected features. 
        Helps identify features that are highly correlated with each other or with the target variable.
        """)

    st.write("### PCA - Dimension Reduction (First 5 Components)")
    pca_features = ['hr', 'season', 'workingday', 'weathersit', 'windspeed', 'hum', 'cnt']
    st.pyplot(ds.eda_pca(features=pca_features, n_components=5).plot(figsize=(8,6)))
    st.markdown("""
        PCA (Principal Component Analysis) plot showing the first 5 principal components. 
        Used for dimensionality reduction and visualizing data in a lower-dimensional space.
        """)
    st.markdown("""
        **Exploratory Data Analysis (EDA) Explanation:**
        This section provides various visualizations to explore the BikeSharing dataset. 
        - **1D EDA:** Shows the distribution of the target variable 'cnt' (bike rental count) using density plots and histograms.
        - **2D & 3D EDA:**  Visualizes relationships between pairs and triplets of features, helping to understand patterns and correlations.
        - **Correlation Heatmap:** Displays the correlation matrix to identify feature relationships.
        - **PCA:** Applies Principal Component Analysis for dimensionality reduction and visualization.
        """)


st.subheader("4. Model Training and Evaluation", divider='gray')

with st.expander("Train and Evaluate Predictive Model"):
    st.write("### Model Selection and Parameters")
    model_name = st.selectbox("Select Model", ["LGBMRegressor"], index=0, help="Currently only LGBMRegressor is implemented.")

    model_params_expander = st.expander("Adjust Model Parameters", expanded=False)
    with model_params_expander:
        n_estimators = st.slider("Number of Estimators (n_estimators)", min_value=50, max_value=500, value=100, step=50, help="The number of boosting stages to be run.")
        max_depth = st.slider("Maximum Depth (max_depth)", min_value=2, max_value=10, value=3, step=1, help="Maximum depth of the individual regression estimators.")
        learning_rate = st.slider("Learning Rate (learning_rate)", min_value=0.01, max_value=0.3, value=0.1, step=0.01, format="%.2f", help="Boosting learning rate.")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_name == "LGBMRegressor":
                model = MoLGBMRegressor(name="LGBM", n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, verbose=-1, random_state=42) # Added learning_rate and random_state
                ds.set_target(feature="cnt")
                ds.set_task_type('Regression')
                ds.set_random_split(test_ratio=0.2, random_state=42) # Consistent random_state for reproducibility
                model.fit(ds.train_x, ds.train_y.ravel())
                st.session_state['trained_model'] = model
                st.session_state['test_suite'] = TestSuite(ds, model)
                st.success("Model trained successfully!", icon="✅")

    if 'trained_model' in st.session_state:
        st.write("### Model Evaluation")
        test_suite = st.session_state['test_suite']

        st.write("#### Performance Metrics")
        accuracy_table_result = test_suite.diagnose_accuracy_table(train_dataset="train", test_dataset="test")
        st.dataframe(accuracy_table_result.table)
        st.markdown("""
            **Accuracy Table:**
            Displays key performance metrics such as Mean Squared Error (MSE) and R-squared (R2) for both the training and test datasets. 
            - **MSE (Mean Squared Error):**  Measures the average squared difference between the predicted and actual values. Lower values are better.
            - **R-squared (R2):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher values (closer to 1) are better.
            """)

        st.write("#### Feature Importance")
        feature_importance_result = test_suite.interpret_fi()
        fig_fi = feature_importance_result.plot(figsize=(8, 5))
        st.pyplot(fig_fi)
        st.markdown("""
            **Feature Importance Plot:**
            Visualizes the importance of each feature in the model's predictions. 
            Features with higher importance values have a greater influence on the model's output.
            """)

        st.write("#### Residual Analysis")
        residual_analysis_result = test_suite.diagnose_residual_analysis(features="hr", dataset="test")
        fig_residual = residual_analysis_result.plot(figsize=(8, 5))
        st.pyplot(fig_residual)
        st.markdown("""
            **Residual Analysis Plot:**
            Analyzes the residuals (the differences between predicted and actual values) to check model assumptions and identify potential issues like non-linearity or heteroscedasticity.
            """)

        st.write("#### Reliability Plot")
        reliability_result = test_suite.diagnose_reliability(train_dataset="test", test_dataset="test", test_size=0.5, random_state=42)
        fig_reliability = reliability_result.plot(figsize=(8, 5))
        st.pyplot(fig_reliability)
        st.markdown("""
            **Reliability Plot:**
            Evaluates the reliability of the model's predictions. 
            It helps to understand if the model's predicted probabilities or intervals are well-calibrated.
            """)

        st.write("#### Robustness - Noise Sensitivity (MAE)")
        robustness_result = test_suite.diagnose_robustness(dataset="test", perturb_features=None, metric="MAE", noise_levels=(0.1, 0.2, 0.3))
        fig_robustness = robustness_result.plot(figsize=(8, 5))
        st.pyplot(fig_robustness)
        st.markdown("""
            **Robustness Plot:**
            Assesses the model's sensitivity to noise in the input features. 
            It shows how the model's performance degrades as noise levels increase.
            """)

        st.markdown("""
            **Model Training and Evaluation Explanation:**
            This section allows users to train and evaluate a predictive model. 
            - **Model Selection:** Users can select a model (currently LGBMRegressor).
            - **Model Parameters:** Adjustable parameters for the selected model to fine-tune training.
            - **Model Training:** Trains the selected model on the preprocessed dataset.
            - **Model Evaluation:** Evaluates the trained model using various metrics and visualizations, including:
                - Accuracy Table (MSE, R2)
                - Feature Importance
                - Residual Analysis
                - Reliability
                - Robustness
            """)


st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
