# Bike Share Predictive Modeler - Technical Specifications

## Overview
The Bike Share Predictive Modeler is a single-page Streamlit application that enables users to train and evaluate predictive models using the Bikesharing dataset from modeva. This application empowers users to understand the process of transforming raw data into interactive visualizations and predictive insights. Users will learn to preprocess data, select features, and evaluate model performance, all within an intuitive, interactive interface.

## Learning Outcomes
- Gain a clear understanding of the key insights derived from the uploaded document.
- Learn how to transform raw data into interactive visualizations using Streamlit.
- Understand the process of data preprocessing and exploration.
- Develop an intuitive, user-friendly application that explains the underlying data concepts.

## Dataset Details
- **Source**: Use the Bikesharing dataset from modeva as shown in the PDF.

## Functional Specifications

### Data Preprocessing and Exploration
- Load the Bikesharing dataset using modeva data handlers.
- Display a preview of the data with options to filter and sort columns.
- Perform preprocessing steps, including feature scaling, encoding, and handling missing values.
- Allow users to select features for modeling using a multi-select widget. Leverage `ds.set_inactive_features` to toggle feature visibility.

### Model Training
- Implement the LGBMRegressor model from `modeva.models` to predict the bike rental count ('cnt').
- Allow users to adjust model parameters using sliders and input boxes, providing real-time feedback on parameter values.
- Initiate model training on the dataset with support for different feature subsets based on user selection.

### Model Evaluation
- After training, evaluate the model's performance and display key metrics including Mean Squared Error (MSE) and R-squared (R2).
- Integrate `TestSuite` from `modeva` to present additional metrics such as reliability and robustness.

### Visualizations
- **Interactive Charts**: Use Streamlit components to create dynamic line charts, bar graphs, and scatter plots showing trends and correlations.
- **Annotations & Tooltips**: Enhance charts with annotations and tool tips to explain data insights and highlight key data points. Include inline explanations of metrics like MSE and R2.
- Display feature importance visually to help users understand which inputs affect the model predictions the most.

### User Interaction
- **Input Forms and Widgets**: Include forms and widgets for interacting with the data and model settings, observing real-time changes in predictions and metrics.
- **Real-time Updates**: Ensure that interactions with parameters, feature selection, and data configurations trigger immediate updates to visualizations and model outputs.

## Documentation and Help
- Provide inline help using tool tips and Streamlit's `st.sidebar.markdown` or `st.markdown` functions to guide users through steps in data exploration, feature selection, model training, and evaluation.
- Offer explanations for each section of the app, ensuring users understand both how to use the app and the significance of the underlying data science concepts.

## Additional Details
- Ensure the design is intuitive, allowing users to effortlessly flow through data exploration, model training, and evaluation without prior machine learning knowledge.
- Seamlessly integrate educational elements, helping users learn the impact of data and parameters on predictive modeling.

## References
- Refer to Streamlit documentation for creating interactive elements: https://docs.streamlit.io/
- Leverage modeva library documentation to ensure correct usage of LGBMRegressor and TestSuite.