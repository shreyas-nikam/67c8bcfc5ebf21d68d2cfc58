# QuLab Bike Share Predictive Modeler

## Description

This Streamlit application, powered by the Modeva library, is designed to predict bike rental demand using the Bikesharing dataset. It provides a user-friendly interface to explore the dataset, perform data preprocessing, conduct exploratory data analysis (EDA), train a predictive model, and evaluate its performance. This application serves as an educational tool to demonstrate the end-to-end process of building a predictive model, from data loading to model diagnostics.

Key features include:

- **Dataset Overview:** Explore raw data, summary statistics, and feature descriptions of the BikeSharing dataset.
- **Data Preprocessing:** Configure and apply preprocessing steps such as feature selection, missing value imputation, numerical scaling, categorical encoding, and numerical binning.
- **Exploratory Data Analysis (EDA):** Visualize data distributions, relationships between features, correlations, and perform dimensionality reduction using PCA.
- **Model Training and Evaluation:** Train a LightGBM Regressor model with adjustable parameters and evaluate its performance using various metrics and diagnostic plots, including accuracy tables, feature importance, residual analysis, reliability plots, and robustness analysis.

## Installation

To run this Streamlit application, you need to have Python installed on your system. Follow these steps to set up your environment:

1. **Clone the repository (if applicable):**
   ```bash
   git clone [repository_url] # Replace [repository_url] with the actual repository URL if available
   cd [repository_directory] # Navigate to the project directory
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install the required Python packages:**
   ```bash
   pip install streamlit modeva pandas numpy matplotlib seaborn
   ```
   **Note:** `modeva` library requires authentication.

4. **Modeva Authentication:**
   This application uses the `modeva` library, which requires authentication to access its functionalities. You will need a Modeva token. For demonstration purposes, the provided code includes a placeholder token. For actual use, you should obtain your own Modeva token.

   * **Obtain a Modeva Token (if you don't have one):** Please refer to the Modeva documentation or contact QuantUniversity to acquire a valid API token for Modeva.
   * **Replace Placeholder Token:**  In the Streamlit application code (`your_app_name.py`), locate the `authenticate()` function call and replace the placeholder token with your actual Modeva token:
     ```python
     authenticate(token='YOUR_MODEVA_TOKEN') # Replace YOUR_MODEVA_TOKEN with your actual token
     ```
     For quick testing, the provided code includes a hardcoded token: `'eaaa4301-b140-484c-8e93-f9f633c8bacb'`. **It is highly recommended to use your own token for extended use and proper access.**

## Usage

1. **Run the Streamlit application:**
   Navigate to the directory containing the Streamlit application file (e.g., `app.py`) in your terminal and run the following command:
   ```bash
   streamlit run your_app_name.py # Replace your_app_name.py with the actual filename of your Streamlit application
   ```

2. **Access the application in your browser:**
   Streamlit will provide a local URL (usually `http://localhost:8501`) in your terminal. Open this URL in your web browser to access the Bike Share Predictive Modeler application.

3. **Explore the application sections:**

   - **Dataset Overview:**  Review the raw data, dataset summary, and feature statistics. Use the "Show Raw Data and Summary" expander to view this information.

   - **Data Preprocessing:**
     - **Feature Selection:**  Choose features to exclude from the model using the "Select features to exclude from the model" multiselect dropdown within the "Configure Data Preprocessing" expander.
     - **Preprocessing Steps:** Review the default preprocessing steps applied by expanding "View Preprocessing Steps Code".
     - **Apply Preprocessing:** Click the "Apply Preprocessing" button to execute the defined preprocessing steps.
     - **Show Preprocessed Data:** Check the "Show Preprocessed Data" checkbox to preview the preprocessed dataset.

   - **Exploratory Data Analysis (EDA):**
     - Expand the "Explore Data Visualizations" section to view various EDA plots.
     - Explore 1D and 2D distributions, correlation heatmaps, and PCA plots to understand the dataset characteristics and feature relationships.

   - **Model Training and Evaluation:**
     - **Model Selection:** Select "LGBMRegressor" from the "Select Model" dropdown (currently, only LGBMRegressor is available).
     - **Model Parameters:** Adjust model hyperparameters like "Number of Estimators", "Maximum Depth", and "Learning Rate" within the "Adjust Model Parameters" expander.
     - **Train Model:** Click the "Train Model" button to train the selected model using the preprocessed data and chosen parameters.
     - **Model Evaluation:** After training, the application will display evaluation metrics (Accuracy Table), Feature Importance plot, Residual Analysis plot, Reliability plot, and Robustness plot. Review these to assess the model's performance and diagnostics.

4. **Interact with the application:**
   Experiment with different preprocessing configurations, EDA visualizations, and model parameters to gain insights into bike share demand prediction and model behavior.

## Credits

Developed by QuantUniversity.

- **Modeva Library:**  This application leverages the Modeva library for dataset handling, preprocessing, model building, and evaluation.
- **Streamlit:** The application's user interface is built using Streamlit.
- **BikeSharing Dataset:**  The application uses the publicly available BikeSharing dataset.

## License

© 2025 QuantUniversity. All Rights Reserved.

This demonstration is for educational purposes only. For full legal documentation, please visit [link to legal documentation - if available, otherwise remove this part]. Reproduction of this demonstration requires prior written consent from QuantUniversity.
