import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as nn
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
import base64
import io
from matplotlib.backends.backend_pdf import PdfPages
import base64 # Already imported, but good to note
from streamlit_pdf_viewer import pdf_viewer # Import the new component

st.set_page_config(layout="wide")
st.title("Air Quality Index (AQI) Dashboard")
st.markdown("---")

# Remove "Documents" tab
tabs = st.tabs(["Home","Feature Importance", "EDA", "PCA", "Classification Models", "Regression Models"])
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

import os # Import os module for path joining

# Function to display PDF using streamlit-pdf-viewer
def show_pdf(file_path):
    """Displays a PDF file from a local path using streamlit-pdf-viewer."""
    try:
        # Ensure the path is correct relative to the script
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, file_path)

        with open(abs_file_path, "rb") as f:
            pdf_bytes = f.read()

        # Use the pdf_viewer component
        pdf_viewer(input=pdf_bytes, width=700) # Pass bytes directly

    except FileNotFoundError:
        st.error(f"Error: File not found. Tried path: {abs_file_path}")
    except Exception as e:
        st.error(f"Error displaying PDF with component: {e}")


# --- Home Tab Content (Displayed Immediately) ---
with tabs[0]:
    st.subheader(" We created the dataset synthetically by implementing the research paper as shown below:")
    st.markdown("Document Viewer")
    # Display the specific PDF file from the local path
    pdf_path = "s10661-023-11646-3.pdf"
    show_pdf(pdf_path)

    # Display the image
    st.subheader("Formula Used:")
    try:
        st.image("Screenshot 2025-04-20 at 7.28.17 PM.png", caption="Formula Image")
    except Exception as e:
        st.error(f"Could not load image: {e}")


    st.markdown("---") # Add a separator
    st.subheader("Dataset Preview (Upload CSV to see data)")
    # Placeholder or message until CSV is uploaded
    if uploaded_file is None:
        st.info("Please upload a CSV file using the sidebar to see the dataset preview and other analyses.")

# --- Content Dependent on CSV Upload ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    import pandas as pd
    from scipy.spatial import cKDTree

    # Load datasets
    df1 = pd.read_csv('AQI_2.csv', encoding='latin-1')

    # Rename df2 columns to avoid conflicts
    df2=df.copy()
    for i in df2.columns:
        if i in df2.columns:
            df2.rename(columns={i: i+"_matched"}, inplace=True)
    
    # Match using PM2.5
    tree = cKDTree(df2[['PM2.5_matched']].values)
    distances, indices = tree.query(df1[['PM2.5']].values, k=1)

    # Match rows from df2
    matched_df2 = df2.iloc[indices].reset_index(drop=True)

    # Concatenate
    merged_df = pd.concat([df1.reset_index(drop=True), matched_df2], axis=1)

    # Identify and drop matched columns that duplicate those in df1
    # (e.g., PM2.5 and PM2.5_matched are same info)
    cols_to_drop = []
    for col in df1.columns:
        matched_col = col + '_matched'
        if matched_col in merged_df.columns:
            cols_to_drop.append(matched_col)

    # Drop the duplicate matched columns
    merged_df = merged_df.drop(columns=cols_to_drop)
    

    
    

    breakpoints_pm25 = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.4, 301, 500)
    ]

    def calculate_aqi(concentration, breakpoints):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo)
        return None

    def categorize_aqi(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

    df["AQI_PM2.5"] = df["PM2.5"].apply(lambda x: calculate_aqi(x, breakpoints_pm25))
    df["AQI_Category"] = df["AQI_PM2.5"].apply(categorize_aqi)

    # --- Data Cleaning and Preparation ---
    def remove_outliers_iqr(df, y):
        """Removes rows with outliers based on IQR for numeric columns."""
        df_copy = df.copy()

        # Scale features
        scaler = MinMaxScaler()
        X_columns = df_copy.columns
        X_scaled = scaler.fit_transform(df_copy)
        X = pd.DataFrame(X_scaled, columns=X_columns, index=df_copy.index)

        # Compute IQR bounds for all columns
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create a boolean mask for rows *without* any outliers
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)

        # Apply the mask to both X and y, making sure they share the same index
        X_filtered = X[mask]
        y_filtered = y[mask]
        return X_filtered , y_filtered
    
    def create_polynomials(df_cleaned):
        poly_features_input_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temp', 'Humidity', 'WS', 'Benzene', 'Toluene', 'NO']
# Ensure only columns present in the dataframe are selected
        poly_features_input_cols = [col for col in poly_features_input_cols if col in df_cleaned.columns]

        if not poly_features_input_cols:
            print("Skipping Polynomial Features: No suitable input columns found.")
        else:
            print(f"Input columns for Polynomial Features: {poly_features_input_cols}")

            # Select the data and handle potential NaNs before applying PolynomialFeatures
            poly_data = df_cleaned[poly_features_input_cols].copy()
            # Option: Impute NaNs if necessary, e.g., poly_data.fillna(poly_data.mean(), inplace=True)
            # For now, let's see how many NaNs we have
            initial_nan_count = poly_data.isnull().sum().sum()
            if initial_nan_count > 0:
                print(f"Warning: Input data for Polynomial Features contains {initial_nan_count} NaN values. Consider imputation.")
                # Simple mean imputation for demonstration:
                print("Applying mean imputation before generating polynomial features.")
                poly_data.fillna(poly_data.mean(), inplace=True)


            # Initialize PolynomialFeatures (degree=2 includes interactions and squares)
            # include_bias=False avoids adding a column of ones
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

            # Fit and transform the data
            poly_features = poly.fit_transform(poly_data)

            # Get the names of the new features
            poly_feature_names = poly.get_feature_names_out(poly_features_input_cols)

            # Create a new DataFrame with these features
            df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=poly_data.index)
            print(f"Generated {df_poly.shape[1]} polynomial features.")

            # Add these new features back to the original DataFrame
            # Avoid adding duplicate columns if some poly features match existing ones (though unlikely with include_bias=False)
            cols_to_add = [col for col in df_poly.columns if col not in df_cleaned.columns]
            df = pd.concat([df_cleaned, df_poly[cols_to_add]], axis=1)

            print(f"Added {len(cols_to_add)} new polynomial features to the DataFrame.")
            # Display some of the new feature names
            print("Example new feature names:", list(cols_to_add[:5]) + list(cols_to_add[-5:]))
            return df


    # Apply outlier removal and drop rows where AQI couldn't be calculated or is NaN
    
    df_clean = df.dropna()
    df_clean["pollutionRatio"] = df_clean["Benzene"] / df_clean["Toluene"]
    df_clean["NO*WS"] = df_clean["NO"] * df_clean["WS"]
    # Check if data is empty after cleaning
    if df_clean.empty:
        st.warning("No data remaining after cleaning and removing NaN AQI values. Cannot proceed with analysis.")
        st.stop() # Stop the script execution if no data left\
    numeric_cols = df_clean.select_dtypes(include='number').columns
    X = df_clean[numeric_cols]
    y = df_clean['AQI_Category'] # Target is the categorical AQI
    X = create_polynomials(X)
    # Ensure y has the same index as X before passing to remove_outliers_iqr
    y = y.loc[X.index]
    X, y = remove_outliers_iqr(X.copy(), y)
    # Check again after selecting X and y
    if X.empty or y.empty:
        st.warning("Feature set (X) or target (y) is empty after processing. Cannot proceed with modeling.")
        st.stop() # Stop execution
    
    # Convert y to Pandas Series to preserve index
    y = pd.Series(y, index=X.index)
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Convert y to Pandas Series after Label Encoding
    y = pd.Series(y, index=X.index)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # --- End Data Cleaning and Preparation ---

    X=X[['PM2.5','AQI_PM2.5',
        'PM10^2',
        'PM10' ,
        'PM10 NO2',
        'PM10 SO2' ,
        'PM10 NO' ,
        'PM10 Benzene' ,
        'PM10 Toluene' ,
        'PM10 WS']]


    # HOME (Display DataFrame Preview only after upload)
    with tabs[0]:
        # The PDF is already displayed outside the if block.
        # Now display the dataframe head since the file is uploaded.
        st.subheader("Dataset 1")
        st.dataframe(df.head())
        st.subheader("Dataset 2")
        st.dataframe(df1.head())
        st.subheader("Merged Dataset:")
        st.dataframe(merged_df.head())
        st.subheader("Dataset description")
        st.dataframe(merged_df.describe())

    # FEATURE IMPORTANCE
    with tabs[1]:
        st.subheader("Feature Importance using Random Forest (Based on Full Cleaned Data)")
        # Ensure y has the same index as X before fitting
        st.dataframe(X.head())
        st.markdown(type(y))
        model_fi = RandomForestClassifier(random_state=42) # Use a fixed random state for reproducibility
        model_fi.fit(X, y) # Fit on the entire cleaned, processed dataset (X, y)

        importance_df = pd.DataFrame({
            "Feature": X.columns, # Get feature names from X
            "Importance": model_fi.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))
        
        # Get valid columns in X after processing
        valid_cols_x = X.columns

        # Create a list of columns to keep, ensuring they exist in X
        cols_to_keep = [col for col in ['PM2.5','AQI_PM2.5', 
        'PM10^2',
        'PM10' ,
        'PM10 NO2',
        'PM10 SO2' ,
        'PM10 NO' ,
        'PM10 Benzene' , 
        'PM10 Toluene' ,
        'PM10 WS'] if col in valid_cols_x]

        # Subset X with the valid columns
        X = X[cols_to_keep]

        plt.figure(figsize=(14, 12))

    # EDA (Remains inside the if block)
    with tabs[2]:
        st.subheader("Histogram Grid")
        valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
        fig, axes = plt.subplots(2, 5, figsize=(30, 20))
        axes = axes.flatten()
        num_plots = min(len(valid_cols), 25)
        for idx in range(num_plots):
            col = valid_cols[idx]
            sns.histplot(X[col], kde=True, ax=axes[idx])
            axes[idx].set_title(col)
        st.pyplot(fig)

        st.subheader("Bivariate Scatter Plots")
        # fig_bi, axes_bi = plt.subplots(2, 5, figsize=(30, 10))
        # for idx, col in enumerate(valid_cols[:10]):
        #     sns.scatterplot(x=col, y="AQI_PM2.5", data=df_clean, ax=axes_bi.flatten()[idx])
        #     axes_bi.flatten()[idx].set_title(f"{col} vs AQI_PM2.5")
        # st.pyplot(fig_bi)
        st.subheader("Boxplots by AQI Category (Using Cleaned Data)")
        fig_box, axes_box = plt.subplots(1, 3, figsize=(24, 6))
        # Use df_clean for boxplots
        fig_box, axes = plt.subplots(2, 5, figsize=(30, 20))
        axes = axes.flatten()

        num_plots = min(len(valid_cols), 25)
        for idx in range(num_plots):
            col = valid_cols[idx]
            sns.boxplot(x=y, y=X[col], ax=axes[idx])
            axes[idx].set_title(col)
        st.pyplot(fig_box)

        st.subheader("Pairplot (Using Cleaned Data)")
        # Use df_clean for pairplot, select relevant columns
        pairplot_cols = ["PM2.5", "SO2", "NO2", "WS", "AQI_PM2.5", "AQI_Category"]
        # Ensure all selected columns exist in df_clean before plotting
        valid_pairplot_cols = [col for col in pairplot_cols if col in df_clean.columns]
        if len(valid_pairplot_cols) > 1: # Need at least two columns for pairplot
             pairplot_fig = sns.pairplot(df_clean[valid_pairplot_cols], hue="AQI_Category" if "AQI_Category" in valid_pairplot_cols else None)
             st.pyplot(pairplot_fig)
        else:
             st.warning("Not enough valid columns found in cleaned data for pairplot.")


        st.subheader("Correlation Heatmap (Using Cleaned Data)")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_clean.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        # st.subheader("Download Cleaned Data")
        # csv = df_clean.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode()).decode()
        # href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Dataset</a>'
        # st.markdown(href, unsafe_allow_html=True)

    

    # PCA TAB
    with tabs[3]:
        st.subheader("PCA")
        pca = PCA(n_components=2)
        y_clean = y[X.index]
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X.index)
        pca_df["AQI_Category"] = y_clean
        st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="AQI_Category", title="PCA"))

    # MODEL METRICS TAB
    with tabs[4]:
        st.subheader("Classifier Models")
        
        models = {
            "Naive Bayes" : GaussianNB(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        preds = {}
        confusion_matrices = {}
        metrics_summary = []

        for name, model in models.items():
            st.markdown(f"### {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[name] = (y_test, y_pred)

            # Evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Save for comparison
            metrics_summary.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": precision,
                "F1 Score": f1
            })

            # Confusion matrix
            conf_df = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
            confusion_matrices[name] = conf_df

            st.write("**Confusion Matrix:**")
            st.dataframe(conf_df)

            st.write(f"**Accuracy:** {acc:.2f}")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")
            st.markdown("---")

        # Comparison table
        st.subheader("Model Comparison Metrics")
        metrics_df = pd.DataFrame(metrics_summary).sort_values(by="F1 Score", ascending=False)
        st.dataframe(metrics_df.style.format({"Accuracy": "{:.2f}", "Precision": "{:.2f}", "F1 Score": "{:.2f}"}))


# Add "Regression Models" to your tabs list
    with tabs[5]:
        st.subheader("Regression Models for PM2.5 Prediction")
        df_clean = X.copy()
        # Separate features and target for regression
        if 'PM2.5' in df_clean.columns:
            # Create regression dataset
            y_reg = df_clean['PM2.5']
            
            # Use base columns from df_clean instead of trying to use X's columns which have polynomial features
            X_reg_base = df_clean.select_dtypes(include=np.number).drop(['PM2.5', 'AQI_PM2.5'], axis=1, errors='ignore')
            
            # Create polynomial features for regression model
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_reg_poly = poly.fit_transform(X_reg_base)
            poly_feature_names = poly.get_feature_names_out(X_reg_base.columns)
            
            # Create DataFrame with polynomial features
            X_reg = pd.DataFrame(X_reg_poly, columns=poly_feature_names, index=X_reg_base.index)
            
            # Check if dataset is valid
            if X_reg.empty:
                st.warning("Not enough features for regression modeling after filtering.")
            else:
                # Display feature set info
                st.write(f"Using {X_reg.shape[1]} features for regression (including polynomial features)")
                
                # Split the data
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                    X_reg, y_reg, test_size=0.3, random_state=42)
                
                # Feature scaling for neural network
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_reg)
                X_test_scaled = scaler.transform(X_test_reg)
                
                # Define regression models
                reg_models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(random_state=42),
                    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
                }
                
                # Import and create Neural Network model
                from sklearn.neural_network import MLPRegressor
                
                # Add Neural Network to the models
                nn_model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                    verbose=False
                )
                
                # Add neural network to the model dictionary
                reg_models["Neural Network"] = nn_model
                
                # Create a DataFrame to store results
                results_df = pd.DataFrame(columns=["Model", "RMSE", "MAE", "R²"])
                
                for name, model in reg_models.items():
                    # Create a subheader for each model
                    st.markdown(f"### {name}")
                    
                    # Fit and predict (use scaled data for Neural Network)
                    if name == "Neural Network":
                        model.fit(X_train_scaled, y_train_reg)
                        y_pred_reg = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train_reg, y_train_reg)
                        y_pred_reg = model.predict(X_test_reg)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
                    mae = mean_absolute_error(y_test_reg, y_pred_reg)
                    r2 = r2_score(y_test_reg, y_pred_reg)
                    
                    # Display metrics
                    st.write(f"**RMSE:** {rmse:.2f}")
                    st.write(f"**MAE:** {mae:.2f}")
                    st.write(f"**R² Score:** {r2:.2f}")
                    
                    # Add to results DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame({
                        "Model": [name],
                        "RMSE": [rmse],
                        "MAE": [mae],
                        "R²": [r2]
                    })], ignore_index=True)
                    
                    # Create scatter plot of actual vs predicted values
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test_reg, y_pred_reg, alpha=0.5)
                    ax.plot([y_test_reg.min(), y_test_reg.max()], 
                            [y_test_reg.min(), y_test_reg.max()], 
                            'k--', lw=2)
                    ax.set_xlabel('Actual PM2.5')
                    ax.set_ylabel('Predicted PM2.5')
                    ax.set_title(f'{name}: Actual vs Predicted PM2.5')
                    st.pyplot(fig)
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        # Get feature importance
                        importances = model.feature_importances_
                        
                        # Handle potential issues with too many features
                        # Just take top 20 features if there are more than 20
                        if len(importances) > 20:
                            indices = np.argsort(importances)[-20:]
                            top_features = X_reg.columns[indices]
                            top_importances = importances[indices]
                            
                            imp_df = pd.DataFrame({
                                'Feature': top_features,
                                'Importance': top_importances
                            }).sort_values('Importance', ascending=False)
                        else:
                            imp_df = pd.DataFrame({
                                'Feature': X_reg.columns,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                        
                        st.write("**Top Feature Importance:**")
                        st.dataframe(imp_df)
                        
                        # Bar chart for feature importance
                        st.bar_chart(imp_df.set_index('Feature'))
                    
                    # For Neural Network, show loss curve
                    if name == "Neural Network" and hasattr(model, 'loss_curve_'):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(model.loss_curve_)
                        ax.set_xlabel('Iterations')
                        ax.set_ylabel('Loss')
                        ax.set_title('Neural Network Training Loss Curve')
                        st.pyplot(fig)
                        
                        if hasattr(model, 'validation_scores_'):
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(model.validation_scores_)
                            ax.set_xlabel('Iterations')
                            ax.set_ylabel('Validation Score')
                            ax.set_title('Neural Network Validation Score')
                            st.pyplot(fig)
                    
                    st.markdown("---")  # Add separator between models
                
                # Model comparison
                st.subheader("Regression Model Comparison")
                
                # Sort by RMSE (lower is better)
                results_df_sorted = results_df.sort_values('RMSE')
                st.dataframe(results_df_sorted)
                
                # Create comparison bar charts
                fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # RMSE comparison (lower is better)
                sns.barplot(x='Model', y='RMSE', data=results_df_sorted, ax=axes[0])
                axes[0].set_title('RMSE Comparison (Lower is Better)')
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
                
                # MAE comparison (lower is better)
                sns.barplot(x='Model', y='MAE', data=results_df_sorted, ax=axes[1])
                axes[1].set_title('MAE Comparison (Lower is Better)')
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
                
                # R² comparison (higher is better)
                sns.barplot(x='Model', y='R²', data=results_df_sorted.sort_values('R²', ascending=False), ax=axes[2])
                axes[2].set_title('R² Comparison (Higher is Better)')
                axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig_comp)
        else:
            st.error("PM2.5 column not found in the dataset. Cannot perform regression modeling.")