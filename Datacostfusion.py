
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.optimize import linprog
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Datacost Fusion System")

# --- Global Variables & Helper Functions ---

# WARNING: Hardcoded file paths. Replace with relative paths or st.file_uploader.
DEFAULT_IMAGE_PATH = '3653777-hd_1280_720_30fps.mp4'
AD_BUDGET_SALES_CSV = "Advertising Budget and Sales.csv"
ADVERTISING_CSV = "advertising.csv"
ADVERTISING_DATA_CSV = "Advertising_Data.csv"
CAMPAIGN_RESULTS_CSV = "Advertising.campaigns.result.csv"
# WARNING: Hardcoded video paths. Replace with accessible URLs or relative paths if videos are local.
ICE_CREAM_VIDEO_PATH = "13525415_1080_1920_30fps.mp4"
FOOD_VIDEO_PATH = "6183107-hd_1920_1080_30fps.mp4"


def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        st.error(f"Error: Image file not found at {image_path}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV {file_path}: {e}")
        return None

def rename_duplicates(df):
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique()
    for dup in duplicates:
        count = 1
        col_indices = cols[cols == dup].index
        new_names = [dup] + [f"{dup}_{i+1}" for i in range(len(col_indices) - 1)]
        for i, idx in enumerate(col_indices):
            cols[idx] = new_names[i]
    df.columns = cols
    return df

def cluster_and_fuse_data(df1, df2, df3, num_clusters):
     # Basic check if dataframes are loaded
    if df1 is None or df2 is None or df3 is None:
        st.error("One or more dataframes failed to load. Cannot perform fusion.")
        return None

    # Handle potential differing columns by aligning before concat
    # A simple approach is to concatenate and let imputation handle NaNs
    # A more robust approach might involve selecting common columns or specific merging logic
    try:
        # Ensure consistent index if needed, or reset index
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        df3 = df3.reset_index(drop=True)

        # Concatenate along columns (axis=1) - assuming rows correspond or it's okay to have NaNs where they don't
        merged_df = pd.concat([df1, df2, df3], axis=1)

        # Rename duplicate columns *before* imputation/scaling if names matter
        merged_df = rename_duplicates(merged_df)

        # Identify numeric columns for imputation and scaling
        numeric_cols = merged_df.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            st.error("No numeric columns found in the merged data for clustering.")
            return merged_df # Return merged df without cluster if no numeric cols

        merged_df_numeric = merged_df[numeric_cols]

        # Impute missing values ONLY in numeric columns
        imputer = SimpleImputer(strategy='mean')
        merged_df_imputed_numeric = pd.DataFrame(imputer.fit_transform(merged_df_numeric), columns=numeric_cols, index=merged_df.index)

        # Scale the numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(merged_df_imputed_numeric)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # Explicitly set n_init
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Add the cluster labels back to the original merged dataframe (or imputed numeric one)
        # It's often better to add it back to the full dataframe including non-numeric cols
        merged_df['Cluster'] = cluster_labels

        return merged_df

    except ValueError as ve:
        st.error(f"ValueError during fusion/clustering: {ve}. Check DataFrame shapes and content.")
        # Attempt to return the raw merged dataframe before the error if possible
        try:
            raw_merged = pd.concat([df1, df2, df3], axis=1)
            raw_merged = rename_duplicates(raw_merged)
            return raw_merged
        except: # If even concat fails, return None
             return None
    except Exception as e:
        st.error(f"An unexpected error occurred during fusion/clustering: {e}")
        return None


# --- Page Functions ---

def home_page():
    """Displays the main welcome page."""
    st.title("Welcome to Datacost Fusion System")
    st.write("Welcome to the Datacost Fusion System! This platform enables seamless integration and analysis of diverse data sources to drive insights and decision-making.")
    st.write("Please explore the various features and functionalities available using the sidebar navigation.")

    image = load_image(DEFAULT_IMAGE_PATH)
    if image:
        st.image(image, use_container_width=True)  # Updated here


def retention_calculator_page():
    """Page 1: Campaign Customer Retention Calculator."""
    st.title('Campaign Customer Retention Calculator')
    st.sidebar.header('Input Parameters') # Keep inputs in sidebar for consistency
    campaign_cost = st.sidebar.number_input('Campaign Cost ($)', min_value=0.0, value=10000.0, key='rc_cost')
    initial_customers = st.sidebar.number_input('Initial Number of Customers', min_value=0, value=1000, key='rc_initial')
    new_customers = st.sidebar.number_input('New Customers Acquired', min_value=0, value=500, key='rc_new')
    retained_customers = st.sidebar.number_input('Retained Customers from Previous Campaign', min_value=0, value=800, key='rc_retained')

    st.header('Campaign Results')
    total_customers = initial_customers + new_customers
    if total_customers > 0:
        retention_rate = retained_customers / total_customers
    else:
        retention_rate = 0

    # Placeholder - Allow user input for revenue per customer or use a default
    revenue_per_customer = st.number_input('Average Revenue Per Retained Customer ($)', min_value=0.0, value=100.0, step=10.0, key='rc_revenue_per')
    revenue_from_retained_customers = retained_customers * revenue_per_customer

    if campaign_cost > 0:
        roi = (revenue_from_retained_customers - campaign_cost) / campaign_cost * 100
    else:
        roi = float('inf') if revenue_from_retained_customers > 0 else 0 # Handle zero cost case

    col1, col2 = st.columns(2)
    col1.metric("Retention Rate", f"{retention_rate:.2%}")
    col2.metric("Revenue from Retained Customers", f"${revenue_from_retained_customers:,.2f}")

    if campaign_cost > 0:
      st.metric("Return on Investment (ROI)", f"{roi:.2f}%")
    else:
      st.info("ROI cannot be calculated with zero campaign cost.")

    st.markdown("---")
    st.write("**Note:** Ensure the 'Retained Customers' number reflects customers retained *due to* or *during* the specific campaign being analyzed for accurate ROI.")


def data_fusion_page():
    """Page 2: Data Fusion and Clustering Analysis."""
    st.title('Data Fusion and Clustering Analysis')

    st.info(f"""
    Attempting to load data from:
    1.  `{AD_BUDGET_SALES_CSV}`
    2.  `{ADVERTISING_CSV}`
    3.  `{ADVERTISING_DATA_CSV}`
    Ensure these files exist or modify the paths.
    """)

    df1 = load_csv(AD_BUDGET_SALES_CSV)
    df2 = load_csv(ADVERTISING_CSV)
    df3 = load_csv(ADVERTISING_DATA_CSV)

    if df1 is None or df2 is None or df3 is None:
        st.error("Could not load all necessary data files. Cannot proceed with fusion.")
        return # Stop execution for this page

    # --- Data Fusion ---
    st.header("Fused Data")
    # Simple concatenation (adjust logic if specific merging is needed)
    try:
        df_fused = pd.concat([df1, df2, df3], ignore_index=True) # Use ignore_index if indices are not meaningful
        df_fused = rename_duplicates(df_fused.copy()) # Ensure renaming happens on a copy
    except Exception as e:
        st.error(f"Error during data concatenation: {e}")
        return

    # Allow the user to select columns to display
    all_columns = df_fused.columns.tolist()
    selected_columns = st.multiselect('Select columns to display:', all_columns, default=all_columns[:min(5, len(all_columns))], key='df_select_cols')

    if selected_columns:
        st.dataframe(df_fused[selected_columns])
    else:
        st.dataframe(df_fused) # Show all if none selected

    # Allow the user to download fused data
    csv = df_fused.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Fused Data as CSV",
        data=csv,
        file_name='fused_data.csv',
        mime='text/csv',
        key='df_download'
    )

    st.markdown("---")

    # --- Clustering ---
    st.header("Clustering Analysis")
    num_clusters = st.slider("Choose the number of clusters (K):", min_value=2, max_value=10, value=3, key='df_n_clusters')

    # Reload original data for clustering function if needed (or pass df_fused if appropriate)
    # cluster_and_fuse_data expects separate DFs, let's reload to be safe according to its original design
    df1_c = load_csv(AD_BUDGET_SALES_CSV)
    df2_c = load_csv(ADVERTISING_CSV)
    df3_c = load_csv(ADVERTISING_DATA_CSV)
    clustered_data = cluster_and_fuse_data(df1_c, df2_c, df3_c, num_clusters)

    if clustered_data is not None:
        st.write("Data with Cluster Labels:")
        st.dataframe(clustered_data)

        # Display cluster distribution
        if 'Cluster' in clustered_data.columns:
            st.subheader("Cluster Distribution")
            cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
        else:
            st.warning("Cluster column not found in the result. Cannot show distribution.")
    else:
        st.warning("Clustering could not be performed.")

    st.markdown("---")

    # --- Visualizations on Fused Data ---
    st.header("Visualizations on Fused Data")

    numeric_cols_fused = df_fused.select_dtypes(include=np.number).columns
    if len(numeric_cols_fused) > 1:
        st.subheader("Correlation Heatmap")
        try:
            corr = df_fused[numeric_cols_fused].corr()
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax_heatmap)
            ax_heatmap.set_title("Heatmap of Numeric Column Correlations")
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap) # Close the plot
        except Exception as e:
            st.error(f"Could not generate heatmap: {e}")
    else:
        st.info("Not enough numeric columns to generate a correlation heatmap.")


    if not df_fused.empty:
        st.subheader("Histogram")
        hist_column = st.selectbox("Select a column for the histogram:", df_fused.columns, key='df_hist_col')
        if hist_column:
            try:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                # Use seaborn's histplot which handles numeric/categorical data
                sns.histplot(df_fused[hist_column], bins=30, kde=True, ax=ax_hist)
                ax_hist.set_title(f'Distribution of {hist_column}')
                ax_hist.set_xlabel(hist_column)
                ax_hist.set_ylabel('Frequency')
                st.pyplot(fig_hist)
                plt.close(fig_hist) # Close the plot
            except Exception as e:
                st.error(f"Could not generate histogram for {hist_column}: {e}")

def budget_pie_chart_page():
    """Page 3: Pie Chart for Budget Distribution and Estimation."""
    st.title('Advertising Budget Analysis')

    st.info(f"Attempting to load data from: `{AD_BUDGET_SALES_CSV}`")
    df = load_csv(AD_BUDGET_SALES_CSV)

    if df is None:
        st.error("Could not load data file. Cannot proceed.")
        return

    st.subheader('Loaded Dataset Sample:')
    st.dataframe(df.head())

    budget_cols = [col for col in df.columns if 'Budget' in col or 'Sales' in col]
    if not budget_cols:
        st.warning("No columns containing 'Budget' or 'Sales' found in the dataset.")
        return

    st.subheader('Budget Distribution Pie Chart')
    selected_budget_type = st.selectbox("Select the budget/sales type for the pie chart:", budget_cols, key='bpc_select')

    if selected_budget_type:
        try:
            # Ensure the column is numeric
            df[selected_budget_type] = pd.to_numeric(df[selected_budget_type], errors='coerce')
            df.dropna(subset=[selected_budget_type], inplace=True) # Drop rows where the selected column is NaN

            total_selected = df[selected_budget_type].sum()

            # Calculate total budget (sum of all budget columns)
            all_budget_cols = [col for col in budget_cols if 'Budget' in col]
            df_numeric_budgets = df[all_budget_cols].apply(pd.to_numeric, errors='coerce')
            total_all_budgets = df_numeric_budgets.sum().sum() # Sum across columns, then sum the totals

            if total_all_budgets > 0:
                # Data for pie chart: selected vs. other budgets
                pie_data = [total_selected, total_all_budgets - total_selected]
                labels = [selected_budget_type, 'Other Budget Types']

                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=90)
                ax_pie.axis('equal')
                ax_pie.set_title(f'Proportion of {selected_budget_type} vs Other Budgets')
                st.pyplot(fig_pie)
                plt.close(fig_pie) # Close the plot
            else:
                st.warning("Total budget is zero or negative. Cannot generate pie chart.")

        except KeyError:
            st.error(f"Column '{selected_budget_type}' not found or contains non-numeric data.")
        except Exception as e:
            st.error(f"An error occurred generating the pie chart: {e}")

    # Simplified estimation logic (original code had undefined LinearRegression)
    # This part needs clarification on the goal. Let's provide basic info.
    st.subheader("Budget Information")
    sales_col = 'Sales ($)' # Assuming this column name exists
    tv_budget_col = 'TV Ad Budget ($)'

    if sales_col in df.columns:
        total_sales = pd.to_numeric(df[sales_col], errors='coerce').sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
    else:
        st.warning(f"Column '{sales_col}' not found.")

    if tv_budget_col in df.columns:
        total_tv_budget = pd.to_numeric(df[tv_budget_col], errors='coerce').sum()
        st.metric("Total TV Ad Budget", f"${total_tv_budget:,.2f}")
    else:
        st.warning(f"Column '{tv_budget_col}' not found.")

    # Placeholder for original estimation - requires model definition/training
    st.subheader("TV Ad Budget Estimation (Placeholder)")
    st.info("The original code included a placeholder for TV Ad Budget prediction using Linear Regression. "
            "To implement this, define features (e.g., Radio Ad Budget, Newspaper Ad Budget, Sales), "
            "train a LinearRegression model, and then use model.predict().")
    # Example structure (needs actual implementation):
    # try:
    #     features = ['Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)']
    #     target = 'TV Ad Budget ($)'
    #     X = df[features].apply(pd.to_numeric, errors='coerce').fillna(df[features].mean())
    #     y = df[target].apply(pd.to_numeric, errors='coerce').fillna(df[target].mean())
    #     model = LinearRegression()
    #     model.fit(X, y)
    #     # Example prediction input
    #     radio_input = st.number_input("Radio Ad Budget ($) for prediction:", value=100000)
    #     news_input = st.number_input("Newspaper Ad Budget ($) for prediction:", value=50000)
    #     sales_input = st.number_input("Sales ($) for prediction:", value=3000000)
    #     prediction = model.predict([[radio_input, news_input, sales_input]])
    #     st.write(f"Estimated TV Ad Budget ($) based on inputs: ${prediction[0]:.2f}")
    # except Exception as e:
    #     st.error(f"Could not perform estimation: {e}. Check if required columns exist and are numeric.")


def advertising_optimization_page():
    """Page 4: Advertising Cost Prediction and Optimization (using fused data)."""
    st.title('Advertising Optimization')

    st.info(f"""
    Attempting to load data from:
    1.  `{AD_BUDGET_SALES_CSV}`
    2.  `{ADVERTISING_CSV}`
    3.  `{ADVERTISING_DATA_CSV}`
    Ensure these files exist or modify the paths. This page uses fused data.
    """)

    df1 = load_csv(AD_BUDGET_SALES_CSV)
    df2 = load_csv(ADVERTISING_CSV)
    df3 = load_csv(ADVERTISING_DATA_CSV)

    if df1 is None or df2 is None or df3 is None:
        st.error("Could not load all necessary data files. Cannot proceed.")
        return

    # --- Data Fusion and Clustering ---
    st.header("Fused Data for Optimization")
    num_clusters_opt = st.slider("Choose the number of clusters for context:", key="opt_clusters", min_value=2, max_value=10, value=3)
    fused_data_opt = cluster_and_fuse_data(df1.copy(), df2.copy(), df3.copy(), num_clusters_opt) # Use copies

    if fused_data_opt is None:
        st.error("Data fusion failed. Cannot proceed with optimization.")
        return

    st.dataframe(fused_data_opt.head())

    # --- Placeholder Prediction/Optimization ---
    # Original code had conflicting predict/optimize functions and targets.
    # Let's clarify the goal: Predict 'Sales' based on ad spends, then 'optimize' cost.

    st.subheader("Predict Sales based on Ad Spend")
    # Define features (ad spends) and target (sales) - Adjust column names as per your fused data
    ad_spend_cols = [col for col in fused_data_opt.columns if 'Ad' in col or 'Budget' in col or 'spend' in col]
    # Try to find a Sales column
    sales_col_opt = next((col for col in fused_data_opt.columns if 'Sales' in col or 'revenue' in col), None)

    if not ad_spend_cols or sales_col_opt is None:
        st.warning(f"Could not automatically identify Ad Spend columns ({ad_spend_cols}) or Sales/Revenue column ({sales_col_opt}) in the fused data. Prediction/Optimization might be inaccurate or fail.")
        # Allow manual selection as fallback
        ad_spend_cols = st.multiselect("Manually select Ad Spend columns (Features):", fused_data_opt.columns, key='opt_manual_features')
        sales_col_opt = st.selectbox("Manually select Sales/Revenue column (Target):", fused_data_opt.columns, key='opt_manual_target')
        if not ad_spend_cols or sales_col_opt is None:
            st.error("Feature and Target columns must be selected to proceed.")
            return

    try:
        # Prepare data - handle potential non-numeric and NaNs
        X_opt = fused_data_opt[ad_spend_cols].apply(pd.to_numeric, errors='coerce')
        y_opt = fused_data_opt[sales_col_opt].apply(pd.to_numeric, errors='coerce')

        # Impute missing values
        imputer_X = SimpleImputer(strategy='mean')
        X_opt_imputed = imputer_X.fit_transform(X_opt)
        imputer_y = SimpleImputer(strategy='mean')
        y_opt_imputed = imputer_y.fit_transform(y_opt.values.reshape(-1, 1)).ravel()

        if len(X_opt_imputed) < 2 or len(y_opt_imputed) < 2:
             st.error("Not enough data after cleaning to train a model.")
             return

        X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt_imputed, y_opt_imputed, test_size=0.2, random_state=42)

        model_opt = LinearRegression()
        model_opt.fit(X_train_opt, y_train_opt)

        predictions_opt = model_opt.predict(X_test_opt)
        st.write("Sample Sales Predictions (Test Set):", predictions_opt[:5])
        st.write(f"Model R-squared (Test Set): {model_opt.score(X_test_opt, y_test_opt):.3f}")

    except Exception as e:
        st.error(f"Error during Sales Prediction model training/prediction: {e}")
        return # Stop if model fails


    st.subheader("Placeholder Cost Optimization")
    # The original optimization logic was very basic (fixed 10% reduction).
    # Real optimization would depend heavily on the prediction model and business goals.
    st.info("Optimization logic below is a simple placeholder (e.g., 10% cost reduction). "
            "Replace with actual optimization strategy based on model insights and objectives.")

    # Identify a primary cost column for the example - User should verify this
    cost_col_to_optimize = st.selectbox(
        "Select the primary Ad Cost column for placeholder optimization:",
        ad_spend_cols, # Use identified or selected ad spend columns
        key='opt_cost_col'
    )

    if cost_col_to_optimize:
        try:
            # Apply optimization to the original fused dataframe
            optimized_data = fused_data_opt.copy()
            # Ensure column is numeric before applying math
            optimized_data[cost_col_to_optimize] = pd.to_numeric(optimized_data[cost_col_to_optimize], errors='coerce')
            # Apply placeholder optimization (e.g., 10% reduction)
            optimized_data[f'Optimized_{cost_col_to_optimize}'] = optimized_data[cost_col_to_optimize] * 0.9
            st.write("Data with Placeholder Optimized Cost Column:")
            st.dataframe(optimized_data.head())
        except Exception as e:
            st.error(f"Error during placeholder optimization: {e}")
    else:
        st.warning("No cost column selected for optimization.")

    st.markdown("---")
    st.header("Optimization Algorithms (Conceptual)")
    st.info("The following demonstrates the UI for selecting optimization algorithms but uses placeholder logic or simple examples.")
    optimization_algorithms_section() # Call the sub-function


def optimization_algorithms_section():
    """Part of Page 4: Demonstrates Optimization Algorithm selection UI."""
    # Note: This section uses placeholder implementations or simple Scipy linprog

    algorithm = st.selectbox("Select Conceptual Optimization Algorithm",
                             ["Linear Programming (LP) Example",
                              "Mixed-Integer Linear Programming (MILP) - Placeholder",
                              "Simulated Annealing - Placeholder"],
                             key='opt_algo_select')

    if algorithm == "Linear Programming (LP) Example":
        st.write("### Linear Programming Example (Maximize Revenue within Budget)")
        st.write("This example uses `scipy.optimize.linprog` to find an optimal allocation.")

        budget_lp = st.number_input("Enter total budget for LP:", min_value=0.0, value=10000.0, step=100.0, format="%.2f", key='lp_budget')
        num_products_lp = st.number_input("Enter the number of products/channels for LP:", min_value=1, value=3, step=1, format="%d", key='lp_num_prod')

        st.write("Enter the cost and expected revenue for each:")
        costs_lp = []
        revenues_lp = []
        for i in range(num_products_lp):
            col1, col2 = st.columns(2)
            with col1:
                cost_i = st.number_input(f"Cost {i+1}:", min_value=0.0, value=float(np.random.randint(500, 2000)), step=50.0, format="%.2f", key=f'lp_cost_{i}')
                costs_lp.append(cost_i)
            with col2:
                revenue_i = st.number_input(f"Revenue {i+1}:", min_value=0.0, value=float(np.random.randint(cost_i, cost_i*5)), step=50.0, format="%.2f", key=f'lp_rev_{i}')
                revenues_lp.append(revenue_i)

        if st.button("Run Linear Programming Optimization", key='lp_run'):
            if not costs_lp or not revenues_lp:
                st.warning("Please enter cost and revenue for all items.")
                return

            # Objective function coefficients (negative revenue for maximization)
            c = [-rev for rev in revenues_lp]
            # Constraint matrix (budget constraint: sum(cost_i * x_i) <= budget_lp)
            # Assuming x_i represents the *amount* spent on channel i (needs bounds adjustment)
            # OR x_i represents allocation proportion (0 to 1), then constraint is sum(cost_i * x_i) <= budget_lp
            # Let's assume x_i is the *number of units* or *level of investment* (needs better definition)
            # Simple approach: Assume x_i is proportion of budget. A = [[1, 1, ...]], b = [1]. Cost constraint: sum(cost_i*x_i*TotalBudget) <= TotalBudget (not linear in x_i)

            # Let's redefine: x_i = amount spent on channel i. Maximize sum(revenue_i/cost_i * x_i) subject to sum(x_i) <= budget_lp
            if any(c <= 0 for c in costs_lp):
                 st.warning("Costs must be positive for ROI calculation.")
                 return
            c_roi = [-rev / cost for rev, cost in zip(revenues_lp, costs_lp)] # Maximize ROI * amount spent
            A = [np.ones(num_products_lp)] # Constraint: sum(x_i) <= budget
            b = [budget_lp]
            # Bounds: amount spent must be non-negative
            bounds = [(0, None) for _ in range(num_products_lp)] # Can spend any non-negative amount up to budget

            try:
                result = linprog(c_roi, A_ub=A, b_ub=b, bounds=bounds, method='highs') # Use highs as default

                if result.success:
                    st.write("### LP Optimization Result")
                    st.write(f"Status: {result.message}")
                    # The result.fun is the negative of the maximized objective function value
                    # It represents maximized sum(ROI * amount_spent). Total Revenue needs recalculation.
                    allocated_amounts = result.x
                    total_revenue_optimized = sum(rev / cost * amount for rev, cost, amount in zip(revenues_lp, costs_lp, allocated_amounts))
                    st.write(f"Maximized Objective Value (Sum of ROI*Amount Spent): {-result.fun:.2f}")
                    st.write(f"Estimated Total Revenue: ${total_revenue_optimized:.2f}")
                    st.write(f"Total Budget Spent: ${sum(allocated_amounts):,.2f}")

                    st.write("Allocation (Amount Spent per Channel):")
                    for i, x_val in enumerate(allocated_amounts):
                        st.write(f"Channel {i+1}: ${x_val:.2f}")
                else:
                    st.error(f"Linear Programming optimization failed: {result.message}")
            except Exception as e:
                st.error(f"Error during Linear Programming: {e}")


    elif algorithm == "Mixed-Integer Linear Programming (MILP) - Placeholder":
        st.write("### Mixed-Integer Linear Programming (MILP)")
        st.info("MILP allows for integer constraints (e.g., choosing whether to run a campaign or not). "
                "Requires specialized solvers (like PuLP, Gurobi, CPLEX). This is a placeholder.")
        # Placeholder function call
        # mixed_integer_linear_programming(budget, prices, revenues)

    elif algorithm == "Simulated Annealing - Placeholder":
        st.write("### Simulated Annealing")
        st.info("Simulated Annealing is a probabilistic technique for finding the global optimum of a function. "
                "Useful for complex, non-linear optimization problems. This is a placeholder.")
        # Placeholder function call
        # simulated_annealing(budget, prices, revenues)

# --- Add other page functions similarly ---

def price_optimization_page():
    st.title("Simple Price Optimization (Demand Prediction)")
    st.info("This tool uses a simple Linear Regression model based on user-provided price/demand pairs to estimate demand at an average price. Assumes a linear relationship.")

    num_products = st.number_input("Enter the number of Price/Demand data points:", min_value=2, value=5, step=1, key='po_num')

    product_data = []
    st.write("Enter historical Price and corresponding Demand:")
    for i in range(num_products):
        st.write(f"--- Data Point {i + 1} ---")
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input(f"Price {i + 1}:", min_value=0.01, value=float(10 + i*2), step=0.50, key=f'po_price_{i}')
        with col2:
            demand = st.number_input(f"Demand {i + 1}:", min_value=0, value=int(100 - i*10), step=1, key=f'po_demand_{i}')
        product_data.append({'Price': price, 'Demand': demand})

    if len(product_data) >= 2:
        df_po = pd.DataFrame(product_data)
        st.write("Entered Data:")
        st.dataframe(df_po)

        try:
            # Train linear regression model (Demand = m * Price + c)
            X_po = df_po[['Price']]
            y_po = df_po['Demand']
            model_po = LinearRegression()
            model_po.fit(X_po, y_po)

            st.write(f"Model Fit: Demand ≈ {model_po.coef_[0]:.2f} * Price + {model_po.intercept_:.2f}")

            # Predict demand at the average price point
            avg_price = df_po['Price'].mean()
            predicted_demand_at_avg = model_po.predict([[avg_price]])[0]

            st.success(f"Estimated Demand at Average Price (${avg_price:.2f}): {predicted_demand_at_avg:.2f} units")

            # Allow user to predict demand for a custom price
            custom_price = st.number_input("Enter a custom price to predict demand:", min_value=0.01, value=avg_price, step=0.50, key='po_custom_price')
            predicted_demand_custom = model_po.predict([[custom_price]])[0]
            st.write(f"Estimated Demand at Price ${custom_price:.2f}: {predicted_demand_custom:.2f} units")

        except Exception as e:
            st.error(f"Could not train model or predict: {e}")
    else:
        st.warning("Please enter at least 2 data points.")


def budget_allocation_tool_page():
    st.title("Simple Budget Allocation Tool")
    st.info("This tool uses sample data for demonstration.")

    # Sample data (can be replaced with uploaded data or inputs)
    customer_data = {
        'Customer': ['Customer A', 'Customer B', 'Customer C'],
        'Segment': ['Segment 1', 'Segment 2', 'Segment 1'],
        'Budget_Allocation': [5000, 7000, 6000]
    }
    product_data = {
        'Product': ['Pen', 'Pencil', 'Notebook'],
        'Unit_Cost': [1, 0.5, 2]
    }
    customer_df = pd.DataFrame(customer_data)
    product_df = pd.DataFrame(product_data)

    st.write("## Budget Allocation")

    selected_customer = st.selectbox("Select Customer", customer_df['Customer'].unique(), key='ba_customer')
    customer_info = customer_df[customer_df['Customer'] == selected_customer].iloc[0]
    selected_segment = customer_info['Segment']
    total_budget = customer_info['Budget_Allocation']

    st.write(f"### Customer: {selected_customer}, Segment: {selected_segment}")
    st.metric("Total Budget Allocated", f"${total_budget:,.2f}")

    st.write("### Enter Product Quantities:")
    quantities = {}
    total_cost = 0
    for index, row in product_df.iterrows():
        quantity = st.number_input(f"Quantity for {row['Product']} (Cost: ${row['Unit_Cost']}/unit)",
                                     min_value=0, step=1, key=f"ba_qty_{row['Product']}")
        quantities[row['Product']] = quantity
        total_cost += quantity * row['Unit_Cost']

    st.write("---")
    st.metric("Calculated Total Cost", f"${total_cost:,.2f}")

    # Check if total cost exceeds allocated budget
    if total_cost > total_budget:
        st.error(f"Total cost (${total_cost:,.2f}) exceeds allocated budget (${total_budget:,.2f})!")
    elif total_cost > 0:
        st.success("Total cost is within the allocated budget.")
        st.progress(total_cost / total_budget)
    else:
         st.info("Enter quantities to calculate cost.")


def customer_segmentation_page():
    st.title("Customer Segmentation with Advertisement (Demo)")
    st.info("This page uses sample data and Linear Regression for a simplified 'purchase amount prediction' based on segments. Real segmentation often uses clustering (like K-Means).")

    # Sample customer data
    customer_data = {
        'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Age': [35, 45, 30, 50, 40, 22, 55, 38],
        'Income': [50000, 60000, 40000, 70000, 55000, 35000, 80000, 62000],
        'Category': ['Electronics', 'Books', 'Electronics', 'Fashion', 'Advertisement', 'Books', 'Fashion', 'Electronics'],
        'Purchase_Amount': [100, 120, 90, 130, 110, 80, 150, 115]
    }
    df_cs = pd.DataFrame(customer_data)

    st.write("### Sample Customer Data")
    st.dataframe(df_cs)

    try:
        # Encoding categorical variable
        label_encoder = LabelEncoder()
        df_cs['Category_Encoded'] = label_encoder.fit_transform(df_cs['Category'])
        categories = label_encoder.classes_ # Get category names

        # Select features and target variable
        X_cs = df_cs[['Age', 'Income', 'Category_Encoded']]
        y_cs = df_cs['Purchase_Amount']

        # Ensure enough data for split
        if len(df_cs) < 5:
             st.warning("Not enough data for train/test split. Using all data for training.")
             X_train_cs, X_test_cs, y_train_cs, y_test_cs = X_cs, X_cs, y_cs, y_cs # Use all data
        else:
             X_train_cs, X_test_cs, y_train_cs, y_test_cs = train_test_split(X_cs, y_cs, test_size=0.2, random_state=42)


        # Train the linear regression model
        model_cs = LinearRegression()
        model_cs.fit(X_train_cs, y_train_cs)

        # Evaluate model (optional display)
        score = model_cs.score(X_test_cs, y_test_cs)
        # st.write(f"Model R-squared Score: {score:.3f}")

        # Display customer segmentation form
        st.write("### Predict Purchase Amount for a New Customer Profile")
        age_input = st.number_input("Enter Age:", min_value=18, max_value=100, value=35, key='cs_age')
        income_input = st.number_input("Enter Income:", min_value=10000, max_value=200000, value=50000, step=1000, key='cs_income')
        category_input = st.selectbox("Select Category:", categories, key='cs_category')

        # Convert category to encoded value
        category_encoded_input = label_encoder.transform([category_input])[0]

        # Make prediction for customer segment
        predicted_purchase_amount = model_cs.predict([[age_input, income_input, category_encoded_input]])
        st.success(f"Predicted Purchase Amount for this profile: ${predicted_purchase_amount[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during segmentation/prediction: {e}")


def product_quality_analysis_page():
    st.title('Product Quality Analysis Dashboard (Demo)')
    st.info("This dashboard uses randomly generated sample data.")

    # Simulate a dataset
    @st.cache_data # Cache the generated data
    def create_quality_dataset(num_records=100):
        np.random.seed(42) # Ensure reproducibility within a session run
        data = {
            'ProductID': range(1, num_records + 1),
            'Rating': np.random.randint(1, 6, size=num_records), # 1 to 5 stars
            'Review': np.random.choice(['Excellent', 'Good', 'Average', 'Poor', 'Terrible'], num_records, p=[0.3, 0.4, 0.15, 0.1, 0.05]),
            'Sales': np.random.randint(50, 1000, size=num_records),
            'Returns': np.random.randint(0, 50, size=num_records) # Added returns metric
        }
        df = pd.DataFrame(data)
        # Convert textual review to numeric score for modeling (example mapping)
        review_map = {'Excellent': 5, 'Good': 4, 'Average': 3, 'Poor': 2, 'Terrible': 1}
        df['Review_Score'] = df['Review'].map(review_map)
        return df

    # Allow regeneration, maybe clear cache? (Simple button for now)
    if st.sidebar.button('Regenerate Sample Data', key='pq_regen'):
        st.cache_data.clear() # Clear cache before regenerating

    df_pq = create_quality_dataset()

    st.sidebar.header('Dataset Options')
    if st.sidebar.checkbox('Show Sample Data', key='pq_show_data'):
        st.write("### Sample Product Quality Data")
        st.dataframe(df_pq)

    # --- Model Training: Predict Rating ---
    st.header("Predict Product Rating")
    st.write("Training a RandomForest model to predict 'Rating' based on other features.")

    try:
        # Define features and target
        # Using Review_Score, Sales, Returns to predict Rating
        features_pq = ['Review_Score', 'Sales', 'Returns']
        target_pq = 'Rating'
        X_pq = df_pq[features_pq]
        y_pq = df_pq[target_pq]

        # Basic validation
        if X_pq.isnull().values.any() or y_pq.isnull().values.any():
             st.warning("Data contains missing values. Imputing with mean.")
             imputer_X_pq = SimpleImputer(strategy='mean')
             X_pq = pd.DataFrame(imputer_X_pq.fit_transform(X_pq), columns=features_pq)
             # Impute y if necessary (less common for target)
             y_pq = y_pq.fillna(y_pq.mean())


        if len(df_pq) < 10:
             st.warning("Very small dataset, model results may not be reliable.")
             # Avoid train/test split if too small
             X_train_pq, X_test_pq, y_train_pq, y_test_pq = X_pq, X_pq, y_pq, y_pq
        else:
            X_train_pq, X_test_pq, y_train_pq, y_test_pq = train_test_split(X_pq, y_pq, test_size=0.2, random_state=42)


        # Model pipeline with scaling
        model_pq = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model_pq.fit(X_train_pq, y_train_pq)

        # --- Prediction and Evaluation ---
        y_pred_pq = model_pq.predict(X_test_pq)
        rmse_pq = np.sqrt(mean_squared_error(y_test_pq, y_pred_pq))

        st.write(f"### Model Evaluation")
        st.metric("Root Mean Square Error (RMSE)", f"{rmse_pq:.3f}")
        # Note: RMSE interpretation depends on the scale of 'Rating' (1-5). Lower is better.

        # --- Feature Importance ---
        st.write("### Feature Importances (Predicting Rating)")
        # Access the regressor step in the pipeline
        regressor_step = model_pq.named_steps['regressor']
        if hasattr(regressor_step, 'feature_importances_'):
            importances = regressor_step.feature_importances_
            features = X_pq.columns
            df_importances = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

            # Use Altair for better interactive charts
            chart_imp = alt.Chart(df_importances).mark_bar().encode(
                x='Importance:Q',
                y=alt.Y('Feature:N', sort='-x') # Sort bars by importance
            ).properties(
                title='Feature Importance for Predicting Product Rating'
            )
            st.altair_chart(chart_imp, use_container_width=True)
        else:
            st.info("Could not retrieve feature importances from the model.")

    except Exception as e:
        st.error(f"An error occurred during model training or evaluation: {e}")


    # --- Insights ---
    st.write("""
    ## Insights for Product Improvement
    - **Review Feature Importances:** Identify which factors (Review Score, Sales, Returns) most significantly influence the predicted product Rating according to the model.
    - **Focus Efforts:** Use this information to prioritize areas for improvement. If 'Returns' is highly important and negative, focus on reducing returns. If 'Review Score' is key, focus on quality and addressing feedback.
    - **Monitor Trends:** Regularly rerun this analysis with updated data to track changes and the impact of improvements.
    """)

def immersive_analytics_page():
    st.title("Immersive Analytics & Interactive Simulations (Demo)")
    st.info("This page uses randomly generated sample advertisement data.")

    # Sample data generation function
    @st.cache_data
    def generate_ad_data(num_records=100):
        np.random.seed(42)
        ad_data = pd.DataFrame({
            'Ad_ID': range(1, num_records + 1),
            'Clicks': np.random.randint(100, 1000, num_records),
            'Views': np.random.randint(1000, 10000, num_records),
            'Conversions': np.random.randint(5, 100, num_records),
            'Cost': np.random.uniform(50, 500, num_records),
            'Platform': np.random.choice(['Google', 'Facebook', 'TikTok', 'LinkedIn'], num_records)
        })
        # Calculate derived metrics
        ad_data['CTR'] = (ad_data['Clicks'] / ad_data['Views']).fillna(0)
        ad_data['CPC'] = (ad_data['Cost'] / ad_data['Clicks']).fillna(0).replace(np.inf, 0)
        ad_data['CPA'] = (ad_data['Cost'] / ad_data['Conversions']).fillna(0).replace(np.inf, 0) # Cost Per Acquisition/Conversion
        ad_data['Revenue'] = ad_data['Conversions'] * np.random.uniform(20, 100, num_records) # Simulate revenue per conversion
        ad_data['ROI'] = ((ad_data['Revenue'] - ad_data['Cost']) / ad_data['Cost']).fillna(0).replace(np.inf, 0)
        return ad_data

    # --- Page Selection ---
    page_mode = st.radio("Select Mode:", ["Immersive Analytics (3D Plot)", "Interactive Simulations (Filtering)"], key='ia_mode', horizontal=True)

    df_ad = generate_ad_data()

    if st.checkbox("Show Sample Ad Data", key='ia_show_data'):
         st.dataframe(df_ad.head())

    if page_mode == "Immersive Analytics (3D Plot)":
        st.header("3D Ad Performance Visualization")
        st.write("Visualizing Clicks, Views, and Revenue, colored by ROI.")

        # Use Plotly Express for interactive 3D scatter plot
        try:
            fig_3d = px.scatter_3d(df_ad,
                                   x='Clicks',
                                   y='Views',
                                   z='Revenue',
                                   color='ROI', # Color points by Return on Investment
                                   size='Cost', # Size points by Cost
                                   hover_data=['Ad_ID', 'Platform', 'CTR', 'CPA'], # Show more info on hover
                                   color_continuous_scale=px.colors.sequential.Viridis, # Color scale
                                   title='3D Ad Performance (Size=Cost, Color=ROI)')
            fig_3d.update_traces(marker=dict(sizemin=3)) # Ensure minimum marker size
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate 3D plot: {e}")

    elif page_mode == "Interactive Simulations (Filtering)":
        st.header("Interactive Ad Data Filtering")
        st.write("Use the sliders and selectors below to filter the ad data.")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_roi = st.slider("Minimum ROI Threshold:", min_value=float(df_ad['ROI'].min()), max_value=float(df_ad['ROI'].max()), value=float(df_ad['ROI'].quantile(0.1)), step=0.1, key='ia_roi_slider')
            max_cpa = st.slider("Maximum Cost Per Acquisition (CPA):", min_value=float(df_ad['CPA'].min()), max_value=float(df_ad['CPA'].quantile(0.95)), value=float(df_ad['CPA'].quantile(0.8)), step=1.0, key='ia_cpa_slider') # Use quantiles for reasonable max
        with col2:
            platforms = st.multiselect("Select Platforms:", options=df_ad['Platform'].unique(), default=df_ad['Platform'].unique(), key='ia_platform_multi')

        # Apply filters
        filtered_ads = df_ad[
            (df_ad['ROI'] >= min_roi) &
            (df_ad['CPA'] <= max_cpa) &
            (df_ad['Platform'].isin(platforms))
        ]

        # Display filtered advertisements
        st.write(f"### Filtered Advertisements ({len(filtered_ads)} matching criteria)")
        st.dataframe(filtered_ads)

        # Summary of filtered data
        if not filtered_ads.empty:
            st.write("#### Summary of Filtered Ads:")
            total_cost_filtered = filtered_ads['Cost'].sum()
            total_revenue_filtered = filtered_ads['Revenue'].sum()
            avg_roi_filtered = filtered_ads['ROI'].mean()
            st.metric("Total Cost", f"${total_cost_filtered:,.2f}")
            st.metric("Total Revenue", f"${total_revenue_filtered:,.2f}")
            st.metric("Average ROI", f"{avg_roi_filtered:.2f}")


def prediction_chart_page():
    st.title('5-Year Prediction Multiline Chart (Demo)')
    st.info(f"""
    Attempting to load historical data from: `{CAMPAIGN_RESULTS_CSV}`.
    **Note:** The prediction logic in the original code was purely random. This version maintains that placeholder logic.
    Replace `generate_predictions` with a proper forecasting model (e.g., ARIMA, Prophet) for real predictions.
    """)

    df_hist = load_csv(CAMPAIGN_RESULTS_CSV)

    if df_hist is None:
        st.error("Could not load historical data file. Cannot proceed.")
        return

    try:
        # Data Preprocessing
        if 'date' not in df_hist.columns:
            st.error("Dataset must contain a 'date' column.")
            return
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.sort_values('date')

        # Identify brand/metric columns (assuming all columns except 'date' are brands/metrics)
        brands = df_hist.columns.drop('date')
        # Ensure brand columns are numeric
        for brand in brands:
            df_hist[brand] = pd.to_numeric(df_hist[brand], errors='coerce')
        # Consider dropping columns that are not numeric after conversion? Or impute?
        # df_hist = df_hist.dropna(subset=brands) # Option: drop rows with NaNs in brand cols

        st.subheader('Historical Dataset Sample')
        st.dataframe(df_hist.head())

        # --- Placeholder Prediction Generation ---
        def generate_random_predictions(historical_data, start_date, brand, years=5):
            """Generates purely random walk predictions."""
            last_value = historical_data[brand].iloc[-1]
            if pd.isna(last_value): # Handle case where last value is NaN
                 last_value = historical_data[brand].mean() # Use mean as a fallback start
                 if pd.isna(last_value): last_value = 0 # Use 0 if mean is also NaN

            future_dates = pd.date_range(start=start_date, periods=365 * years, freq='D')
            # Simple random walk: value = last_value + cumulative sum of random steps
            # Adjust scale of random step based on historical data std dev for *slight* realism
            std_dev = historical_data[brand].std()
            if pd.isna(std_dev) or std_dev == 0: std_dev = abs(last_value * 0.05) # Use 5% of last value if std is NaN/0
            random_steps = np.random.normal(loc=0, scale=std_dev * 0.1, size=len(future_dates)) # Smaller steps
            predicted_values = last_value + np.cumsum(random_steps)
            # Ensure non-negative predictions if applicable
            predicted_values[predicted_values < 0] = 0

            predictions = pd.DataFrame({
                'date': future_dates,
                'brand': brand,
                'value': predicted_values,
                'type': 'Predicted' # Add type identifier
            })
            return predictions

        # Input start date for predictions
        # Default to day after last date in data
        default_start_date = (df_hist['date'].max() + timedelta(days=1)).date()
        start_date_input = st.date_input('Select the start date for predictions', default_start_date, key='pred_start_date')

        # Generate predictions for each brand
        predictions_list = []
        for brand in brands:
            if not df_hist[brand].isnull().all(): # Only predict if there's some data
                predictions = generate_random_predictions(df_hist, start_date_input, brand, years=5)
                predictions_list.append(predictions)
            else:
                 st.warning(f"Skipping prediction for '{brand}' as it contains only missing values.")


        if not predictions_list:
            st.error("No predictions could be generated for any brand.")
            return

        predictions_df = pd.concat(predictions_list, ignore_index=True)

        # Prepare historical data for plotting
        historical_melted = df_hist.melt(id_vars=['date'], value_vars=brands, var_name='brand', value_name='value')
        historical_melted['type'] = 'Historical'

        # Combine historical and predicted data
        combined_df = pd.concat([historical_melted, predictions_df], ignore_index=True)
        combined_df = combined_df.dropna(subset=['value']) # Drop rows where value is NaN

        # --- Visualization ---
        st.subheader('Historical and Predicted Data')

        # Multiline chart using Altair
        line_chart = alt.Chart(combined_df).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(title='Date')),
            y=alt.Y('value:Q', axis=alt.Axis(title='Value')),
            color='brand:N',
            strokeDash=alt.StrokeDash('type:N', legend=alt.Legend(title="Data Type")), # Differentiate historical/predicted
            tooltip=['date:T', 'brand:N', 'value:Q', 'type:N']
        ).properties(
            title='Historical vs. Predicted Data (5 Years)',
            width=800, # Adjust width as needed
            # height=400 # Let height adjust or set explicitly
        ).interactive() # Allow zooming and panning

        st.altair_chart(line_chart, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e) # Print traceback for debugging


def campaign_creation_page():
    st.title('Campaign Success Filter (Based on New Users)')
    st.info(f"Loads data from `{CAMPAIGN_RESULTS_CSV}` and filters campaigns based on a 'new_users' threshold.")

    df_cc = load_csv(CAMPAIGN_RESULTS_CSV)
    if df_cc is None: return

    st.subheader('Loaded Campaign Dataset Sample:')
    st.dataframe(df_cc.head())

    # Check if 'new_users' column exists
    if 'new_users' not in df_cc.columns:
        st.error("The dataset must contain a column named 'new_users'. Please check the CSV file.")
        # Allow user to select the column if the name differs
        alt_user_col = st.selectbox("If 'new_users' column has a different name, select it here:", [''] + df_cc.columns.tolist(), key='cc_alt_col')
        if not alt_user_col:
            return
        user_col = alt_user_col
    else:
        user_col = 'new_users' # Default name

    try:
        # Ensure the column is numeric
        df_cc[user_col] = pd.to_numeric(df_cc[user_col], errors='coerce')
        # Drop rows where the user column is NaN, as they can't be compared
        df_cc_cleaned = df_cc.dropna(subset=[user_col])

        # Set a threshold for successful new users
        min_users = int(df_cc_cleaned[user_col].min()) if not df_cc_cleaned.empty else 0
        max_users = int(df_cc_cleaned[user_col].max()) if not df_cc_cleaned.empty else 1000
        default_threshold = int(df_cc_cleaned[user_col].median()) if not df_cc_cleaned.empty else 500

        new_users_threshold = st.number_input(
            f"Enter the minimum '{user_col}' threshold for a 'Successful Campaign':",
            min_value=min_users,
            max_value=max_users,
            value=default_threshold,
            step=max(1, (max_users - min_users) // 100), # Dynamic step size
            key='cc_threshold'
        )

        # Filter based on the threshold
        successful_campaigns = df_cc_cleaned[df_cc_cleaned[user_col] > new_users_threshold]

        # Display the successful campaigns
        st.subheader(f'Campaigns with > {new_users_threshold} {user_col}:')
        if successful_campaigns.empty:
            st.warning("No campaigns met the specified threshold.")
        else:
            st.dataframe(successful_campaigns)
            st.metric("Number of Successful Campaigns", len(successful_campaigns))

    except KeyError:
        st.error(f"Column '{user_col}' not found after selection. Please check.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def campaign_forecasting_budget_page():
    st.title('Campaign Forecasting Budget (Demo)')
    st.info("This page demonstrates a *placeholder* budget forecast. The calculation is based on simple multiplication and random noise, not a predictive model.")

    # Input: Start date of the campaign
    start_date = st.date_input('Select the start date of the campaign', datetime.today().date(), key='cfb_start_date')
    num_days = st.number_input('Number of days in campaign forecast:', min_value=1, max_value=90, value=14, key='cfb_days')

    # Parameters for placeholder calculation
    initial_users = st.number_input('Estimated Initial Users Targeted:', min_value=0, value=100, key='cfb_init_users')
    cost_per_new_user_target = st.number_input('Estimated Cost Per New User Targeted ($):', min_value=0.0, value=5.0, step=0.5, key='cfb_cost_per')
    daily_growth_factor = st.slider('Assumed Daily User Growth Factor:', 0.9, 1.5, 1.05, 0.01, key='cfb_growth')
    noise_level = st.number_input('Random Noise Level ($):', min_value=0.0, value=100.0, step=10.0, key='cfb_noise')


    # Placeholder forecast logic
    campaign_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    forecasted_budgets = []
    current_users = initial_users
    for i in range(num_days):
        # Simple growth model + cost per user + noise
        daily_budget = current_users * cost_per_new_user_target + np.random.normal(scale=noise_level)
        # Ensure budget is non-negative
        forecasted_budgets.append(max(0, daily_budget))
        # Update user count for next day
        current_users *= daily_growth_factor


    # Display campaign details
    st.subheader('Campaign Period:')
    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {campaign_dates[-1]}")

    # Display forecasted budget for each day
    st.subheader('Forecasted Daily Budget:')
    df_forecast = pd.DataFrame({
        'Date': campaign_dates,
        'Forecasted Budget ($)': forecasted_budgets
    })
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date']).dt.strftime('%Y-%m-%d')

    st.dataframe(df_forecast)

    # Visualization
    st.subheader("Budget Forecast Visualization")
    chart = alt.Chart(df_forecast).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Forecasted Budget ($):Q', title='Forecasted Budget ($)'),
        tooltip=['Date:T', alt.Tooltip('Forecasted Budget ($):Q', format=',.2f')]
    ).properties(
        title=f'{num_days}-Day Budget Forecast'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.metric("Total Forecasted Budget", f"${sum(forecasted_budgets):,.2f}")


def sales_analysis_production_page():
    st.title('Sales Analysis and Production Suggestion')
    st.info(f"Loads data from `{CAMPAIGN_RESULTS_CSV}` and provides suggestions based on 'Sales' thresholds.")

    df_sap = load_csv(CAMPAIGN_RESULTS_CSV)
    if df_sap is None: return

    st.subheader('Loaded Campaign Dataset Sample:')
    st.dataframe(df_sap.head())

    # Check if 'Sales' column exists
    sales_col = 'Sales' # Default name
    if sales_col not in df_sap.columns:
        st.error(f"Dataset must contain a column named '{sales_col}'. Please check the CSV file.")
        # Allow user to select the column if the name differs
        alt_sales_col = st.selectbox("If 'Sales' column has a different name, select it here:", [''] + df_sap.columns.tolist(), key='sap_alt_col')
        if not alt_sales_col:
            return
        sales_col = alt_sales_col

    try:
        # Ensure the column is numeric
        df_sap[sales_col] = pd.to_numeric(df_sap[sales_col], errors='coerce')
        df_sap_cleaned = df_sap.dropna(subset=[sales_col])

        if df_sap_cleaned.empty:
             st.warning(f"No valid numeric data found in the '{sales_col}' column.")
             return

        # Set thresholds for low and high sales (use quantiles for better defaults)
        min_sales = df_sap_cleaned[sales_col].min()
        max_sales = df_sap_cleaned[sales_col].max()
        q25_sales = df_sap_cleaned[sales_col].quantile(0.25)
        q75_sales = df_sap_cleaned[sales_col].quantile(0.75)

        st.subheader("Define Sales Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            low_sales_threshold = st.number_input(
                f"Threshold for 'Low' {sales_col}:",
                min_value=min_sales, max_value=max_sales, value=q25_sales,
                key='sap_low_thresh', format="%f" # Use float format for sales
            )
        with col2:
             high_sales_threshold = st.number_input(
                f"Threshold for 'High' {sales_col}:",
                min_value=min_sales, max_value=max_sales, value=q75_sales,
                key='sap_high_thresh', format="%f"
            )

        if low_sales_threshold >= high_sales_threshold:
            st.warning("Low sales threshold should be less than high sales threshold.")
            return

        # Filter based on thresholds
        low_sales_data = df_sap_cleaned[df_sap_cleaned[sales_col] < low_sales_threshold]
        high_sales_data = df_sap_cleaned[df_sap_cleaned[sales_col] > high_sales_threshold]
        medium_sales_data = df_sap_cleaned[
            (df_sap_cleaned[sales_col] >= low_sales_threshold) &
            (df_sap_cleaned[sales_col] <= high_sales_threshold)
        ]

        total_records = len(df_sap_cleaned)
        low_sales_percentage = (len(low_sales_data) / total_records) * 100 if total_records else 0
        high_sales_percentage = (len(high_sales_data) / total_records) * 100 if total_records else 0
        medium_sales_percentage = (len(medium_sales_data) / total_records) * 100 if total_records else 0


        # Suggestion logic
        st.subheader('Production Suggestion Based on Sales Analysis:')
        suggestion = ""
        suggestion_type = "info" # "success", "warning", "error"

        # Prioritize high sales message
        if high_sales_percentage > 25: # Using 25% as an example trigger point
            suggestion = (f"Sales performance is strong ({high_sales_percentage:.1f}% of records show high sales). "
                          "Consider increasing production or inventory to meet potential demand and capitalize on success.")
            suggestion_type = "success"
        elif low_sales_percentage > 30: # Using 30% as example trigger
             suggestion = (f"A significant portion of records ({low_sales_percentage:.1f}%) show low sales. "
                           "Review marketing strategies, product positioning, or pricing. Analyze low-performing campaigns/products.")
             suggestion_type = "warning"
        else:
             suggestion = ("Sales distribution appears relatively balanced or within expected ranges based on current thresholds. "
                           f"({low_sales_percentage:.1f}% Low, {medium_sales_percentage:.1f}% Medium, {high_sales_percentage:.1f}% High). "
                           "Continue monitoring performance.")
             suggestion_type = "info"

        if suggestion_type == "success": st.success(suggestion)
        elif suggestion_type == "warning": st.warning(suggestion)
        else: st.info(suggestion)

        # Display counts
        st.write("---")
        st.write("**Data Counts by Sales Category:**")
        st.metric(f"Low Sales (< {low_sales_threshold:,.2f})", len(low_sales_data))
        st.metric(f"Medium Sales (>= {low_sales_threshold:,.2f} and <= {high_sales_threshold:,.2f})", len(medium_sales_data))
        st.metric(f"High Sales (> {high_sales_threshold:,.2f})", len(high_sales_data))


    except KeyError:
        st.error(f"Column '{sales_col}' not found after selection. Please check.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def product_differentiation_page():
    st.title('Product Differentiation Based on Customer Purchases (Demo)')
    st.info("This page uses randomly generated sample purchase data.")

    # Generate sample data
    @st.cache_data
    def generate_purchase_data(num_records=1000, num_customers=300):
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = (end_date - start_date).days
        dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(num_records)]
        products = ['Laptop', 'Mouse', 'Keyboard', 'Webcam', 'Monitor', 'Headset']
        data = {
            'date': dates,
            'product': np.random.choice(products, num_records, p=[0.2, 0.15, 0.15, 0.1, 0.25, 0.15]), # Example probabilities
            'customer_id': np.random.randint(1, num_customers + 1, num_records),
            'quantity': np.random.randint(1, 4, num_records)
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    df_pd = generate_purchase_data()

    if st.checkbox("Show Sample Purchase Data", key='pd_show_data'):
        st.subheader('Sample Purchase Dataset')
        st.dataframe(df_pd.head())

    # Select a date to filter the dataset
    min_date = df_pd['date'].min().date()
    max_date = df_pd['date'].max().date()
    selected_date = st.date_input('Select a date to view purchases:', value=min_date, min_value=min_date, max_value=max_date, key='pd_date_select')

    # Filter the dataset based on the selected date
    # Convert selected_date back to datetime for comparison
    filtered_df = df_pd[df_pd['date'].dt.date == selected_date]

    st.subheader(f'Analysis for {selected_date}')

    if not filtered_df.empty:
        st.write(f"**{len(filtered_df)}** purchase records found on this date.")
        if st.checkbox(f'Show purchase records for {selected_date}', key='pd_show_filtered'):
             st.dataframe(filtered_df)

        # Calculate purchase counts for each product on that day
        # Consider quantity? Let's do count of transactions first, then sum of quantity.
        product_transaction_counts = filtered_df['product'].value_counts().reset_index()
        product_transaction_counts.columns = ['product', 'transaction_count']

        product_quantity_sum = filtered_df.groupby('product')['quantity'].sum().reset_index()
        product_quantity_sum.columns = ['product', 'total_quantity_sold']

        # Merge the counts
        product_summary = pd.merge(product_transaction_counts, product_quantity_sum, on='product', how='outer').fillna(0)

        st.subheader('Product Purchase Summary')
        st.dataframe(product_summary)

        # --- Visualizations ---
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Transactions per Product")
            bar_chart = alt.Chart(product_summary).mark_bar().encode(
                x=alt.X('product:N', title='Product'),
                y=alt.Y('transaction_count:Q', title='Number of Transactions'),
                color='product:N',
                tooltip=['product', 'transaction_count']
            ).properties(
                # title=f'Transactions per Product on {selected_date}' # Title can be redundant
            ).interactive()
            st.altair_chart(bar_chart, use_container_width=True)

        with col2:
            st.write("#### Total Quantity Sold per Product")
            pie_chart_data = product_summary[['product', 'total_quantity_sold']]
            pie_chart = alt.Chart(pie_chart_data).mark_arc(outerRadius=120).encode(
                 theta=alt.Theta(field='total_quantity_sold', type='quantitative', stack=True),
                 color=alt.Color(field='product', type='nominal'),
                 tooltip=['product', 'total_quantity_sold']
             ).properties(
                # title=f'Proportion of Quantity Sold on {selected_date}' # Title can be redundant
             )
            st.altair_chart(pie_chart, use_container_width=True)


    else:
        st.warning(f'No purchases found for {selected_date}')


def ad_budget_estimation_page():
    st.title('Advertising Budget Estimation Based on Sales Occasion (Demo)')
    st.info("Uses sample data linking occasions, sales, and ad budgets.")

    # Sample sales and advertising data
    @st.cache_data
    def generate_occasion_data():
        np.random.seed(42)
        data = {
            'occasion': ['New Year', 'Valentine\'s Day', 'Easter', 'Summer Sale', 'Back to School', 'Halloween', 'Black Friday', 'Christmas'],
            'sales': np.random.randint(5000, 30000, 8),
            'ad_budget': np.random.randint(500, 5000, 8)
        }
        df = pd.DataFrame(data)
        # Ensure budget is somewhat correlated with sales for realism
        df['ad_budget'] = (df['sales'] * np.random.uniform(0.05, 0.15, 8)).astype(int)
        return df

    df_abe = generate_occasion_data()

    st.subheader('Sample Sales and Advertising Data by Occasion')
    st.dataframe(df_abe)

    # Calculate the highest sales occasion and its budget
    max_sales_idx = df_abe['sales'].idxmax()
    max_sales_occasion_info = df_abe.loc[max_sales_idx]

    highest_sales_occasion = max_sales_occasion_info['occasion']
    highest_sales = max_sales_occasion_info['sales']
    corresponding_budget = max_sales_occasion_info['ad_budget']

    st.subheader('Occasion with Highest Sales')
    st.metric("Occasion", highest_sales_occasion)
    st.metric("Highest Sales Value", f"${highest_sales:,.0f}")
    st.metric("Ad Budget for this Occasion", f"${corresponding_budget:,.0f}")

    # --- Visualization ---
    st.subheader('Sales and Advertising Budget by Occasion')
    try:
        # Melt data for easier plotting with Altair
        df_melted = df_abe.melt(id_vars=['occasion'], value_vars=['sales', 'ad_budget'], var_name='Metric', value_name='Value')

        # Create grouped bar chart
        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('occasion:N', title='Occasion', sort='-y'), # Sort occasions by value
            y=alt.Y('Value:Q', title='Amount ($)'),
            color='Metric:N',
            tooltip=['occasion', 'Metric', alt.Tooltip('Value:Q', format=',.0f')],
            xOffset='Metric:N' # Group bars side-by-side
        ).properties(
            title='Sales vs. Ad Budget by Occasion'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating chart: {e}")


    # --- Budget Adjustment ---
    st.subheader('Adjust Advertising Budget for Highest Sales Occasion')
    st.write(f"Use the slider to apply a multiplier to the budget for **{highest_sales_occasion}** (${corresponding_budget:,.0f}).")
    multiplier = st.slider('Select Budget Multiplier:', min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='abe_multiplier')
    adjusted_budget = corresponding_budget * multiplier
    st.metric(f"Adjusted Ad Budget for {highest_sales_occasion}", f"${adjusted_budget:,.0f}")

    # --- Custom Estimation ---
    st.subheader('Estimate Budget for a Custom Sales Target')
    selected_occasion_est = st.selectbox('Select an occasion to base estimation on:', df_abe['occasion'], key='abe_select_occasion')
    custom_sales_target = st.number_input('Enter desired sales target for this occasion:', min_value=0, value=int(df_abe.loc[df_abe['occasion']==selected_occasion_est, 'sales'].iloc[0]), step=1000, key='abe_custom_sales')

    # Simple ratio-based estimation
    selected_occasion_data = df_abe[df_abe['occasion'] == selected_occasion_est].iloc[0]
    if selected_occasion_data['sales'] > 0:
        ad_budget_ratio = selected_occasion_data['ad_budget'] / selected_occasion_data['sales']
        estimated_custom_budget = custom_sales_target * ad_budget_ratio
        st.success(f"Estimated Ad Budget for ${custom_sales_target:,.0f} sales on {selected_occasion_est}: **${estimated_custom_budget:,.0f}** (based on historical ratio)")
    else:
        st.warning(f"Cannot estimate budget for {selected_occasion_est} as historical sales are zero.")

# --- Add stubs for the remaining page functions ---

def cost_analysis_page():
    st.title("Cost Analysis for Decision-Making (Demo)")
    st.info("Uses sample data for different cost types.")

    # Sample data
    @st.cache_data
    def generate_cost_data():
        np.random.seed(42)
        data = {
            'product': ['Product A', 'Product B', 'Product C', 'Product D'],
            'current_cost': np.random.randint(100, 500, 4),
            'incremental_cost': np.random.randint(20, 80, 4), # Increased range
            'opportunity_cost': np.random.randint(10, 50, 4), # Increased range
            'sunk_cost': np.random.randint(50, 200, 4) # Increased range
        }
        df = pd.DataFrame(data)
        # Differential Cost: Change in total cost between alternatives.
        # Example: Cost of new process - Cost of old process. Here, let's assume it's Current vs (Current + Incremental - Opportunity)
        # This definition can vary greatly based on the specific decision!
        # Simple definition for demo: Incremental cost associated with a change.
        df['differential_cost'] = df['incremental_cost'] # Simplistic view for demo
        # Relevant Cost: Costs that differ between alternatives and are future-oriented.
        # Typically includes incremental and opportunity costs. Excludes sunk costs.
        df['relevant_cost'] = df['incremental_cost'] + df['opportunity_cost']
        return df

    df_ca = generate_cost_data()

    st.subheader('Sample Product Cost Data')
    st.dataframe(df_ca)

    st.subheader('Analyze Costs for a Specific Product')
    selected_product = st.selectbox('Select a product:', df_ca['product'], key='ca_select_prod')

    # Retrieve the cost data
    product_data = df_ca[df_ca['product'] == selected_product].iloc[0]

    # Display the costs with definitions
    st.write(f"#### Cost Data for: {selected_product}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Cost", f"${product_data['current_cost']}")
        st.caption("Baseline cost of the product/process.")
        st.metric("Incremental Cost", f"${product_data['incremental_cost']}")
        st.caption("Additional cost incurred if a change is made.")
        st.metric("Relevant Cost", f"${product_data['relevant_cost']}")
        st.caption("Future costs differing between alternatives (e.g., Incremental + Opportunity). Key for decisions.")
    with col2:
        st.metric("Opportunity Cost", f"${product_data['opportunity_cost']}")
        st.caption("Value of the next best alternative foregone.")
        st.metric("Sunk Cost", f"${product_data['sunk_cost']}")
        st.caption("Past costs already incurred and cannot be recovered (irrelevant for future decisions).")
        st.metric("Differential Cost", f"${product_data['differential_cost']}")
        st.caption("Difference in total cost between two alternatives (using simplified demo definition: Incremental).")


    # --- Visualization ---
    st.subheader('Relevant vs. Irrelevant Cost Visualization')
    # Focus on relevant costs for decision making vs sunk cost
    cost_viz_data = pd.DataFrame({
        'Cost Type': ['Incremental Cost', 'Opportunity Cost', 'Sunk Cost'],
        'Amount': [product_data['incremental_cost'], product_data['opportunity_cost'], product_data['sunk_cost']],
        'Relevance': ['Relevant', 'Relevant', 'Irrelevant']
    })

    chart_rel = alt.Chart(cost_viz_data).mark_bar().encode(
        x=alt.X('Amount:Q', title='Cost Amount ($)'),
        y=alt.Y('Cost Type:N', title='Cost Type', sort='-x'),
        color='Relevance:N',
        tooltip=['Cost Type', 'Amount', 'Relevance']
    ).properties(
        title=f'Relevant vs. Irrelevant Costs for {selected_product}'
    ).interactive()
    st.altair_chart(chart_rel, use_container_width=True)

    # --- Custom Analysis ---
    with st.expander("Run Custom Cost Scenario"):
        st.subheader('Custom Cost Scenario Input')
        st.write("Enter hypothetical costs to calculate relevant and differential costs.")
        custom_current_cost = st.number_input('Custom Current Cost ($):', min_value=0, value=int(product_data['current_cost']), key='ca_cust_curr')
        custom_incremental_cost = st.number_input('Custom Incremental Cost ($):', min_value=0, value=int(product_data['incremental_cost']), key='ca_cust_inc')
        custom_opportunity_cost = st.number_input('Custom Opportunity Cost ($):', min_value=0, value=int(product_data['opportunity_cost']), key='ca_cust_opp')
        custom_sunk_cost = st.number_input('Custom Sunk Cost ($) (Informational only):', min_value=0, value=int(product_data['sunk_cost']), key='ca_cust_sunk')

        # Recalculate based on definitions used above
        custom_differential_cost = custom_incremental_cost # Demo definition
        custom_relevant_cost = custom_incremental_cost + custom_opportunity_cost

        st.write("---")
        st.metric("Calculated Custom Relevant Cost", f"${custom_relevant_cost}")
        st.metric("Calculated Custom Differential Cost", f"${custom_differential_cost}")
        st.caption("Remember: Sunk costs are irrelevant for the decision.")


def cost_optimization_algorithms_page():
    st.title('Cost Optimization Algorithms (Demo)')
    st.info("Uses sample data and simple LP for Budget Allocation, and basic sorting for ROI Maximization.")

    # Sample data for Budget Allocation
    @st.cache_data
    def generate_channel_data():
        np.random.seed(42)
        channels = ['SEO', 'PPC', 'Email', 'Social Media', 'Affiliate', 'Display']
        data = {
            'Channel': channels,
            'Expected Return': np.random.randint(5000, 25000, len(channels)),
            'Cost': np.random.randint(1000, 5000, len(channels))
        }
        df = pd.DataFrame(data)
        # Ensure realistic costs relative to return
        df['Cost'] = (df['Expected Return'] * np.random.uniform(0.1, 0.4, len(channels))).astype(int)
        return df

    df_coa = generate_channel_data()

    # --- Budget Allocation Model (Linear Programming) ---
    st.header('Budget Allocation Model (Maximize Return using LP)')

    total_budget_coa = st.number_input('Enter total budget:', min_value=0, value=10000, step=1000, key='coa_budget')
    min_return_constraint = st.number_input('Optional: Minimum required total return:', min_value=0, value=0, step=5000, key='coa_min_return')

    st.write("#### Channel Performance Data (Sample)")
    st.dataframe(df_coa)

    if st.button("Run Budget Allocation Optimization", key='coa_run_lp'):
        # Objective function: Maximize sum(Expected Return_i * x_i) where x_i is proportion of budget (0 to 1)
        # c = -np.array(df_coa['Expected Return']) # Maximize total return

        # Alternative: Maximize total ROI. ROI = Return/Cost. Maximize sum(ROI_i * amount_spent_i)
        # Let x_i be the *amount* spent on channel i.
        if (df_coa['Cost'] <= 0).any():
             st.warning("Cannot calculate ROI for channels with zero or negative cost.")
             return

        df_coa['ROI'] = df_coa['Expected Return'] / df_coa['Cost']
        c = -np.array(df_coa['ROI']) # Maximize sum(ROI_i * x_i)

        # Constraints:
        # 1. sum(x_i) <= total_budget_coa (Total spent <= Budget)
        # 2. sum(Expected_Return_i / Cost_i * x_i) >= min_return_constraint (If specified)
        #    (This constraint assumes return scales linearly with spend, which might not be true)
        # Let's stick to budget constraint only for simplicity here.
        constraints_A = [np.ones(len(df_coa))] # Sum of amounts spent
        constraints_b = [total_budget_coa] # Must be <= total budget

        # Bounds for each variable (amount spent must be non-negative)
        bounds = [(0, None) for _ in range(len(df_coa))] # Can spend $0 or more on each

        try:
            result = linprog(c, A_ub=constraints_A, b_ub=constraints_b, bounds=bounds, method='highs')

            if result.success:
                allocated_amounts = result.x
                df_coa['Allocated Budget'] = allocated_amounts
                df_coa['Resulting Return'] = df_coa['ROI'] * df_coa['Allocated Budget']

                st.subheader('Optimized Budget Allocation')
                st.dataframe(df_coa[['Channel', 'Cost', 'Expected Return', 'ROI', 'Allocated Budget', 'Resulting Return']].round(2))

                total_allocated = df_coa['Allocated Budget'].sum()
                total_return = df_coa['Resulting Return'].sum()
                st.metric("Total Budget Allocated", f"${total_allocated:,.2f}")
                st.metric("Total Estimated Return", f"${total_return:,.2f}")

                # --- Visualization ---
                st.subheader('Budget Allocation Pie Chart')
                # Filter out channels with zero allocation for cleaner pie chart
                plot_data = df_coa[df_coa['Allocated Budget'] > 0.01] # Avoid tiny slices
                if not plot_data.empty:
                    fig_pie, ax_pie = plt.subplots()
                    ax_pie.pie(plot_data['Allocated Budget'], labels=plot_data['Channel'], autopct='%1.1f%%', startangle=90)
                    ax_pie.axis('equal')
                    ax_pie.set_title("Optimized Budget Allocation by Channel")
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)
                else:
                    st.info("LP resulted in zero allocation or failed.")

            else:
                st.error(f"Linear Programming optimization failed: {result.message}")
        except Exception as e:
            st.error(f"Error during Linear Programming: {e}")


    # --- ROI Maximization (Greedy Approach) ---
    st.header('ROI Maximization (Activity Prioritization)')
    st.info("This section demonstrates a simple greedy approach: sorting activities by ROI and selecting them until the budget is exhausted.")

    # Example data for marketing activities
    # Let's reuse channel data, treating each as an 'activity'
    activities_df = df_coa[['Channel', 'Expected Return', 'Cost']].copy()
    if (activities_df['Cost'] <= 0).any():
        st.warning("Some activities have zero or negative cost. ROI cannot be calculated reliably.")
        activities_df = activities_df[activities_df['Cost'] > 0] # Filter them out

    if not activities_df.empty:
        activities_df['ROI'] = activities_df['Expected Return'] / activities_df['Cost']
        activities_df = activities_df.sort_values(by='ROI', ascending=False).reset_index(drop=True)

        st.subheader('Activities Sorted by ROI (Highest First)')
        st.dataframe(activities_df)

        # Simulate greedy selection based on budget
        budget_roi = st.number_input('Enter budget for ROI prioritization:', min_value=0, value=total_budget_coa, step=1000, key='coa_roi_budget')
        selected_activities = []
        remaining_budget = budget_roi
        for index, row in activities_df.iterrows():
            if remaining_budget >= row['Cost']:
                selected_activities.append(row['Channel'])
                remaining_budget -= row['Cost']
            # else: # Optional: break if an activity costs more than remaining budget
            #     break

        st.subheader('Selected Activities based on Greedy ROI Prioritization')
        if selected_activities:
            st.write(f"**Selected:** {', '.join(selected_activities)}")
            selected_df = activities_df[activities_df['Channel'].isin(selected_activities)]
            total_cost_selected = selected_df['Cost'].sum()
            total_return_selected = selected_df['Expected Return'].sum()
            st.metric("Total Cost of Selected Activities", f"${total_cost_selected:,.2f}")
            st.metric("Total Return from Selected Activities", f"${total_return_selected:,.2f}")
            st.metric("Remaining Budget", f"${remaining_budget:,.2f}")
        else:
            st.info("No activities selected based on the budget and ROI ranking.")
    else:
        st.warning("No valid activities for ROI prioritization.")


def nested_model_page():
    st.title('Nested Model for Predicting Ad Clicks (Demo)')
    st.info(f"""
    Loads data from `{CAMPAIGN_RESULTS_CSV}`.
    Predicts the *next* day's 'clicks' based on the *previous* day's 'spend'.
    This is a time-series-like prediction using a simple Linear Regression.
    """)

    df_nm = load_csv(CAMPAIGN_RESULTS_CSV)
    if df_nm is None: return

    # --- Data Preprocessing ---
    required_cols = ['spend', 'clicks']
    if not all(col in df_nm.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {required_cols}. Please check the CSV.")
        # Allow selection?
        sel_spend = st.selectbox("Select 'Spend' column:", [''] + df_nm.columns.tolist(), key='nm_sel_spend')
        sel_clicks = st.selectbox("Select 'Clicks' column:", [''] + df_nm.columns.tolist(), key='nm_sel_clicks')
        if not sel_spend or not sel_clicks: return
        spend_col, clicks_col = sel_spend, sel_clicks
    else:
        spend_col, clicks_col = 'spend', 'clicks'

    try:
        df_nm[spend_col] = pd.to_numeric(df_nm[spend_col], errors='coerce')
        df_nm[clicks_col] = pd.to_numeric(df_nm[clicks_col], errors='coerce')

        # Create lagged features: Previous day's spend predicts today's clicks
        df_nm['previous_spend'] = df_nm[spend_col].shift(1)
        df_nm_cleaned = df_nm.dropna(subset=['previous_spend', clicks_col]) # Drop rows with NaNs created by shift or original NaNs

        if df_nm_cleaned.empty or len(df_nm_cleaned) < 2:
            st.error("Not enough valid data after creating lagged features to train a model.")
            return

        X_nm = df_nm_cleaned[['previous_spend']] # Feature: previous day's spend
        y_nm = df_nm_cleaned[clicks_col]       # Target: today's clicks

        # Split the data
        if len(df_nm_cleaned) < 10:
            st.warning("Small dataset, using all data for training.")
            X_train_nm, X_test_nm, y_train_nm, y_test_nm = X_nm, X_nm, y_nm, y_nm
        else:
            X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_nm, y_nm, test_size=0.2, random_state=42, shuffle=False) # Don't shuffle time-based data

        # --- Model Training ---
        model_nm = LinearRegression()
        model_nm.fit(X_train_nm, y_train_nm)

        # --- Evaluation ---
        predictions_nm = model_nm.predict(X_test_nm)
        mse_nm = mean_squared_error(y_test_nm, predictions_nm)
        r2_nm = r2_score(y_test_nm, predictions_nm)

        st.subheader('Model Evaluation (Test Set)')
        st.metric('Mean Squared Error (MSE)', f"{mse_nm:.2f}")
        st.metric('R-squared (R2)', f"{r2_nm:.3f}")

        # --- Prediction for Next Day ---
        st.subheader('Predict Clicks for Tomorrow')
        # Get the very last known spend value from the original data
        last_spend_value = df_nm[spend_col].iloc[-1]
        if pd.isna(last_spend_value):
            st.warning("The last spend value is missing. Prediction might be inaccurate. Using mean spend instead.")
            last_spend_value = df_nm[spend_col].mean() # Use mean as fallback

        if pd.isna(last_spend_value):
             st.error("Cannot predict next day's clicks as no valid spend data is available.")
        else:
            next_day_prediction = model_nm.predict([[last_spend_value]])
            st.metric(f'Predicted Clicks for tomorrow (based on last spend of {last_spend_value:.2f})', f"{next_day_prediction[0]:.2f}")


        # --- Visualization ---
        st.subheader('Model Predictions vs. Actual Values (Test Set)')
        results_df = pd.DataFrame({'Actual Clicks': y_test_nm, 'Predicted Clicks': predictions_nm}, index=y_test_nm.index)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results_df.index, results_df['Actual Clicks'], color='black', marker='.', linestyle='-', label='Actual Clicks')
        ax.plot(results_df.index, results_df['Predicted Clicks'], color='blue', marker='.', linestyle='--', label='Predicted Clicks')
        ax.set_title('Nested Model: Actual vs. Predicted Clicks')
        ax.set_xlabel('Data Point Index (Test Set)')
        ax.set_ylabel('Number of Clicks')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    except KeyError as ke:
        st.error(f"Missing column: {ke}. Please ensure '{spend_col}' and '{clicks_col}' exist.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)


def proportion_actual_predicted_page():
    st.title('Daily Proportion of Actual vs. Predicted Values (Demo)')
    st.info("Uses randomly generated sample data for actual vs. predicted values over a campaign duration.")

    # Generate sample data
    np.random.seed(42)
    max_campaign_days = 30
    y_test_sample = np.random.randint(50, 200, max_campaign_days) # Sample actual values
    predictions_sample = y_test_sample + np.random.normal(0, 20, max_campaign_days) # Sample predictions with some noise
    predictions_sample = np.clip(predictions_sample, 0, None) # Ensure predictions are non-negative

    # Slider to select number of days
    num_days_viz = st.slider('Select number of campaign days to visualize:', min_value=1, max_value=max_campaign_days, value=min(14, max_campaign_days), key='prop_days_slider')

    # Data for the selected duration
    y_test_viz = y_test_sample[:num_days_viz]
    predictions_viz = predictions_sample[:num_days_viz]

    st.subheader(f'Daily Actual vs. Predicted Proportions (First {num_days_viz} Days)')

    cols = st.columns(min(num_days_viz, 5)) # Show up to 5 pies per row

    total_actual_viz = np.sum(y_test_viz)
    total_predicted_viz = np.sum(predictions_viz)

    if total_actual_viz <= 0 and total_predicted_viz <= 0:
        st.warning("Total actual and predicted values are zero. Cannot generate proportions.")
        return

    labels = ['Actual', 'Predicted']
    colors = ['skyblue', 'lightcoral']

    for i in range(num_days_viz):
        daily_actual = y_test_viz[i]
        daily_predicted = predictions_viz[i]

        # Proportions based on daily total (Actual + Predicted for that day)
        daily_total = daily_actual + daily_predicted
        if daily_total > 0:
            sizes = [daily_actual / daily_total, daily_predicted / daily_total]
        else:
            sizes = [0, 0] # Avoid division by zero

        # Plotting in columns
        with cols[i % len(cols)]:
             fig_pie, ax_pie = plt.subplots(figsize=(3, 3)) # Smaller figsize
             ax_pie.pie(sizes, labels=labels if i==0 else None, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
             ax_pie.set_title(f'Day {i + 1}', fontsize=10)
             st.pyplot(fig_pie)
             plt.close(fig_pie) # Close plot

    # Add overall comparison
    st.subheader("Overall Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Actual Value", f"{total_actual_viz:.2f}")
    with col2:
        st.metric("Total Predicted Value", f"{total_predicted_viz:.2f}")


def actual_predicted_line_page():
    st.title('Actual vs. Predicted Values Over Campaign (Line Chart Demo)')
    st.info("Uses randomly generated sample data.")

    # Generate sample data
    np.random.seed(42)
    max_campaign_days_line = 30
    y_test_line = np.random.randint(100, 200, max_campaign_days_line) + np.linspace(0, 50, max_campaign_days_line) # Add a slight upward trend
    predictions_line = y_test_line + np.random.normal(0, 15, max_campaign_days_line)
    predictions_line = np.clip(predictions_line, 0, None)
    days_line = np.arange(1, max_campaign_days_line + 1)

    # Slider for number of days
    reduced_days_line = st.slider(
        'Select number of campaign days to visualize:',
        min_value=1, max_value=max_campaign_days_line,
        value=min(14, max_campaign_days_line),
        key='ap_line_days'
    )

    # Filter data
    days_reduced = days_line[:reduced_days_line]
    y_test_reduced = y_test_line[:reduced_days_line]
    predictions_reduced = predictions_line[:reduced_days_line]

    # Create DataFrame for Altair
    df_plot = pd.DataFrame({
        'Day': days_reduced,
        'Actual': y_test_reduced,
        'Predicted': predictions_reduced
    })
    df_melted = df_plot.melt(id_vars=['Day'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Value')

    # Plot using Altair
    st.subheader(f'Actual vs. Predicted Values ({reduced_days_line}-Day Campaign)')
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('Day:Q', axis=alt.Axis(title='Day of Campaign')),
        y=alt.Y('Value:Q', axis=alt.Axis(title='Metric Value')),
        color='Type:N',
        tooltip=['Day', 'Type', alt.Tooltip('Value:Q', format=',.2f')]
    ).properties(
        title=f'Actual vs. Predicted Values Over a {reduced_days_line}-Day Campaign'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def cumulative_actual_predicted_page():
    st.title('Cumulative Actual vs. Predicted Values (Bar Chart Demo)')
    st.info("Uses randomly generated sample data.")

     # Generate sample data (consistent with previous page for linkage if needed)
    np.random.seed(42)
    max_campaign_days_cum = 30
    y_test_cum = np.random.randint(100, 200, max_campaign_days_cum) + np.linspace(0, 50, max_campaign_days_cum)
    predictions_cum = y_test_cum + np.random.normal(0, 15, max_campaign_days_cum)
    predictions_cum = np.clip(predictions_cum, 0, None)
    days_cum = np.arange(1, max_campaign_days_cum + 1)

    # Slider for number of days
    selected_days_cum = st.slider(
        "Select the number of days to visualize:",
        min_value=1, max_value=max_campaign_days_cum,
        value=min(14, max_campaign_days_cum),
        key='cum_days_slider'
    )

    # Filter data and calculate cumulative sums
    y_test_selected = y_test_cum[:selected_days_cum]
    predictions_selected = predictions_cum[:selected_days_cum]
    days_selected = days_cum[:selected_days_cum]

    cumulative_actual = np.cumsum(y_test_selected)
    cumulative_predicted = np.cumsum(predictions_selected)

    # Create DataFrame for Altair
    df_cum_plot = pd.DataFrame({
        'Day': days_selected,
        'Cumulative Actual': cumulative_actual,
        'Cumulative Predicted': cumulative_predicted
    })
    df_cum_melted = df_cum_plot.melt(
        id_vars=['Day'],
        value_vars=['Cumulative Actual', 'Cumulative Predicted'],
        var_name='Type', value_name='Cumulative Value'
    )

    # Plot using Altair grouped bar chart
    st.subheader(f'Cumulative Actual vs. Predicted Values ({selected_days_cum}-Day Campaign)')
    chart = alt.Chart(df_cum_melted).mark_bar().encode(
        x=alt.X('Day:O', axis=alt.Axis(title='Day of Campaign')), # Ordinal scale for days
        y=alt.Y('Cumulative Value:Q', axis=alt.Axis(title='Cumulative Value')),
        color='Type:N',
        tooltip=['Day', 'Type', alt.Tooltip('Cumulative Value:Q', format=',.2f')],
        xOffset='Type:N' # Group bars side-by-side
    ).properties(
        title=f'Cumulative Actual vs. Predicted Values Over {selected_days_cum} Days'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def sma_crossover_strategy_page():
    st.title("Simple Moving Average (SMA) Crossover Strategy")
    st.info("Fetches historical stock data from Yahoo Finance and calculates SMA crossover signals.")

    # --- User Inputs ---
    st.sidebar.header("SMA Strategy Inputs")
    ticker = st.sidebar.text_input("Stock Ticker Symbol:", "AAPL", key='sma_ticker')
    start_date_sma = st.sidebar.date_input("Select Start Date:", datetime(2021, 1, 1).date(), key='sma_start')
    end_date_sma = st.sidebar.date_input("Select End Date:", datetime(2025, 1, 1).date(), key='sma_end')
    short_window = st.sidebar.slider("Short Window (days):", 10, 100, 50, 5, key='sma_short')
    long_window = st.sidebar.slider("Long Window (days):", 50, 250, 200, 10, key='sma_long')

    if short_window >= long_window:
        st.sidebar.warning("Short window should be less than long window.")
        return

    if start_date_sma >= end_date_sma:
        st.sidebar.warning("Start date must be before end date.")
        return

    # --- Data Fetching ---
    st.subheader(f"Fetching Data for: {ticker}")
    st.write(f"Date Range: {start_date_sma} to {end_date_sma}")

    try:
        def get_stock_data(ticker, start, end):
            return yf.download(ticker, start=start, end=end)

        stock_data = get_stock_data(ticker, start_date_sma, end_date_sma)

        if stock_data.empty:
            st.error(f"No data returned for {ticker}. Check the ticker symbol and date range.")
            return

        st.subheader("Historical Stock Price Data")
        st.dataframe(stock_data.tail())

        # --- Strategy Calculation ---
        signals = pd.DataFrame(index=stock_data.index)
        signals['Close'] = stock_data['Close']
        signals['signal'] = 0.0

        signals['short_mavg'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_mavg'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()

        # Safe assignment using .loc to avoid SettingWithCopyWarning
        signals.loc[signals.index[short_window:], 'signal'] = np.where(
            signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0
        )

        signals['positions'] = signals['signal'].diff()

        st.subheader("Strategy Signals and Moving Averages")
        st.dataframe(signals.tail())

        # --- Visualization ---
        st.subheader("Stock Price with SMA and Trading Signals")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='lines', name='Close Price', line=dict(color='skyblue')))
        fig.add_trace(go.Scatter(x=signals.index, y=signals['short_mavg'], mode='lines', name=f'SMA {short_window}', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=signals.index, y=signals['long_mavg'], mode='lines', name=f'SMA {long_window}', line=dict(color='purple')))

        buy_signals = signals[signals['positions'] == 1.0]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['short_mavg'],
                                 mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
        sell_signals = signals[signals['positions'] == -1.0]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['short_mavg'],
                                 mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

        fig.update_layout(
            title=f'{ticker} SMA Crossover Strategy ({short_window}/{long_window})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("An unexpected error occurred while processing the data.")
        st.exception(e)



def campaign_analysis_pie_page():
    st.title("Campaign Contribution Analysis (Demo)")
    st.info("Uses sample data showing hypothetical percentage contributions of campaigns to a product's success.")

    # Sample data
    @st.cache_data
    def generate_campaign_contrib_data():
        data = {
            'Product': ['Product A', 'Product B', 'Product C'],
            'Campaign_X_Contrib (%)': [30, 40, 20],
            'Campaign_Y_Contrib (%)': [40, 30, 55],
            'Campaign_Z_Contrib (%)': [20, 25, 15],
            # Adding 'Other Factors' to make percentages sum closer to 100 (or represent base sales)
            'Other_Factors (%)': [10, 5, 10]
        }
        df = pd.DataFrame(data)
        df = df.set_index('Product')
        # Normalize rows to sum to 100 (optional, depends on interpretation)
        # df = df.apply(lambda row: row / row.sum() * 100, axis=1)
        return df

    df_cap = generate_campaign_contrib_data()

    st.write("### Sample Campaign Contribution Data (%)")
    st.dataframe(df_cap)

    st.header("Analyze Campaign Contributions")
    selected_product_cap = st.selectbox("Select Product:", df_cap.index.tolist(), key='cap_select_prod')

    # Get data for the selected product
    product_contributions = df_cap.loc[selected_product_cap]

    # Display contributions
    st.write(f"#### Contributions for {selected_product_cap}:")
    st.dataframe(product_contributions)

    # Create pie chart using Altair
    st.write("#### Contribution Distribution")
    pie_data = product_contributions.reset_index()
    pie_data.columns = ['Campaign/Factor', 'Contribution (%)']

    # Ensure contributions are numeric
    pie_data['Contribution (%)'] = pd.to_numeric(pie_data['Contribution (%)'], errors='coerce')
    pie_data = pie_data.dropna()

    if not pie_data.empty:
        chart = alt.Chart(pie_data).mark_arc(outerRadius=120).encode(
            theta=alt.Theta(field="Contribution (%)", type="quantitative", stack=True),
            color=alt.Color(field="Campaign/Factor", type="nominal"),
            tooltip=['Campaign/Factor', alt.Tooltip("Contribution (%)", format='.1f')]
        ).properties(
            title=f'Campaign Contribution Distribution for {selected_product_cap}'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No valid contribution data to plot.")

    # Example calculation based on contributions (interpret with caution)
    st.header("Hypothetical Impact Calculation")
    total_metric_value = st.number_input(f"Enter a total metric value for {selected_product_cap} (e.g., Total Sales, Total Leads):", min_value=0.0, value=10000.0, step=100.0, key='cap_total_metric')

    st.write("#### Estimated Impact per Campaign/Factor:")
    results = {}
    for factor, contribution in product_contributions.items():
         factor_impact = (contribution / 100.0) * total_metric_value
         results[factor] = factor_impact
         st.metric(factor, f"{factor_impact:,.2f}")


def sales_kpi_calculator_page():
    st.title("Sales KPI Calculator (Campaign Day Impact Demo)")
    st.info("Uses randomly generated synthetic sales data to show differences on campaign vs. non-campaign days.")

    # Function to generate synthetic sales data
    @st.cache_data
    def generate_sales_kpi_data(num_records=100):
        np.random.seed(0) # for reproducibility

        campaign_days = np.random.choice([0, 1], size=num_records, p=[0.7, 0.3]) # 30% campaign days
        base_units_sold = np.random.randint(50, 300, size=num_records)
        base_revenue = base_units_sold * np.random.uniform(15, 50, size=num_records)
        advertising_cost = np.random.randint(100, 1000, size=num_records)

        # Apply campaign day uplift factor
        campaign_uplift_factor = 1.5
        units_sold = np.where(campaign_days == 1, base_units_sold * campaign_uplift_factor, base_units_sold).round().astype(int)
        revenue = np.where(campaign_days == 1, base_revenue * campaign_uplift_factor, base_revenue).round(2)
        # Adjust ad cost slightly higher on campaign days?
        advertising_cost = np.where(campaign_days == 1, advertising_cost * 1.2, advertising_cost).round(2)


        data = {
            'Record_ID': np.arange(1, num_records + 1),
            'Units Sold': units_sold,
            'Revenue': revenue,
            'Advertising Cost': advertising_cost,
            'Campaign Day': campaign_days # 1 for campaign, 0 for non-campaign
        }
        df = pd.DataFrame(data)
        df['Campaign Day'] = df['Campaign Day'].map({1: 'Yes', 0: 'No'}) # Map to Yes/No for clarity
        return df

    num_records_kpi = st.sidebar.number_input("Number of Sales Records to Generate:", min_value=10, step=10, value=100, key='kpi_num_records')
    sales_data_kpi = generate_sales_kpi_data(num_records_kpi)

    if st.checkbox("Show Generated Synthetic Sales Data", key='kpi_show_data'):
        st.write("Generated Data Sample:")
        st.dataframe(sales_data_kpi.head())

    st.header("Analysis: Impact of Campaign Days")

    # Calculate average KPIs by Campaign Day status
    kpi_summary = sales_data_kpi.groupby('Campaign Day').agg(
        Avg_Units_Sold=('Units Sold', 'mean'),
        Avg_Revenue=('Revenue', 'mean'),
        Avg_Advertising_Cost=('Advertising Cost', 'mean'),
        Total_Records=('Record_ID', 'count')
    ).reset_index()

    # Calculate Avg Revenue Per Unit
    kpi_summary['Avg_Revenue_Per_Unit'] = kpi_summary['Avg_Revenue'] / kpi_summary['Avg_Units_Sold']
    # Calculate rough ROI (Avg Revenue / Avg Ad Cost) - simplified
    kpi_summary['Approx_ROI'] = kpi_summary['Avg_Revenue'] / kpi_summary['Avg_Advertising_Cost']

    st.subheader("Average KPIs: Campaign Day vs. Non-Campaign Day")
    st.dataframe(kpi_summary.round(2))

    # --- Visualizations ---
    st.subheader("Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart comparing Avg Units Sold
        chart_units = alt.Chart(kpi_summary).mark_bar().encode(
            x=alt.X('Campaign Day:N', title=None, axis=alt.Axis(labels=False)), # Hide x-axis labels
            y=alt.Y('Avg_Units_Sold:Q', title='Average Units Sold'),
            color='Campaign Day:N',
            tooltip=['Campaign Day', alt.Tooltip('Avg_Units_Sold', format='.1f')]
        ).properties(
            title='Avg. Units Sold'
        )
        text_units = chart_units.mark_text(
            align='center',
            baseline='bottom',
            dy=-5 # Nudge text up slightly
        ).encode(
            text=alt.Text('Avg_Units_Sold:Q', format='.1f')
        )
        st.altair_chart(chart_units + text_units, use_container_width=True)

    with col2:
        # Bar chart comparing Avg Revenue
        chart_rev = alt.Chart(kpi_summary).mark_bar().encode(
            x=alt.X('Campaign Day:N', title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('Avg_Revenue:Q', title='Average Revenue ($)'),
            color='Campaign Day:N',
            tooltip=['Campaign Day', alt.Tooltip('Avg_Revenue', format=',.2f')]
        ).properties(
            title='Avg. Revenue'
        )
        text_rev = chart_rev.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text=alt.Text('Avg_Revenue:Q', format=',.0f')
        )
        st.altair_chart(chart_rev + text_rev, use_container_width=True)

    # Pie chart for proportion of *total* units sold on campaign vs non-campaign days
    st.subheader("Proportion of Total Units Sold")
    total_units_summary = sales_data_kpi.groupby('Campaign Day')['Units Sold'].sum().reset_index()

    if not total_units_summary.empty and total_units_summary['Units Sold'].sum() > 0:
        pie_units = alt.Chart(total_units_summary).mark_arc(outerRadius=100).encode(
            theta=alt.Theta(field="Units Sold", type="quantitative", stack=True),
            color=alt.Color(field="Campaign Day", type="nominal"),
            tooltip=['Campaign Day', 'Units Sold']
        ).properties(
            title='Proportion of Total Units Sold: Campaign vs Non-Campaign Days'
        )
        st.altair_chart(pie_units, use_container_width=True)
    else:
        st.warning("Could not calculate total units sold proportions.")


def attribution_roi_page():
    st.title("Attribution Modeling and ROI Analysis (Placeholder Demo)")
    st.info("Uses sample data and *placeholder* logic for attribution and ROI calculation. Real attribution is complex (first-touch, last-touch, multi-touch models).")

    # Sample data
    @st.cache_data
    def generate_attribution_data(num_days=31):
        np.random.seed(42)
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        channels = ['Google Ads', 'Facebook Ads', 'Email', 'Organic Search', 'Direct']
        data = {
            'Date': np.random.choice(dates, num_days*3, replace=True), # More records
            'Channel': np.random.choice(channels, num_days*3, p=[0.3, 0.3, 0.15, 0.15, 0.1]),
            'Clicks': np.random.randint(50, 1000, num_days*3),
            'Cost': np.random.uniform(20, 500, num_days*3),
            'Conversions': np.random.randint(1, 50, num_days*3)
        }
        df = pd.DataFrame(data)
        # Simulate cost being related to clicks and conversions somewhat
        df['Cost'] = (df['Clicks'] * np.random.uniform(0.1, 1.5, num_days*3) + df['Conversions'] * np.random.uniform(5, 20, num_days*3)).round(2)
        df['Conversions'] = np.where(df['Clicks'] < 100, df['Conversions'] * 0.5, df['Conversions']).astype(int) # Fewer conversions for low clicks
        df = df.sort_values(by=['Date', 'Channel']).reset_index(drop=True)
        return df

    df_atr = generate_attribution_data()

    st.subheader("Sample Channel Performance Data")
    st.dataframe(df_atr.head())

    # --- Placeholder Attribution ---
    st.subheader("Attribution Modeling (Placeholder - Simple 'Last Touch' Implied)")
    # This example doesn't implement complex models. It assumes conversions are directly linked to the channel record.
    # A simple *placeholder* calculation: Assume each channel gets credit proportional to conversions recorded against it.
    attribution_weight = st.slider("Placeholder Attribution Weight (0 to 1):", 0.0, 1.0, 0.5, 0.1, key='atr_weight', help="Example: 0.5 means 50% of recorded conversions are attributed.")
    df_atr['Attributed_Conversions'] = (df_atr['Conversions'] * attribution_weight).round(2)

    # --- Placeholder ROI Analysis ---
    st.subheader("ROI Analysis (Based on Placeholder Attribution)")
    # ROI = (Revenue - Cost) / Cost
    # Need revenue. Let's estimate revenue per attributed conversion.
    avg_revenue_per_conversion = st.number_input("Estimated Average Revenue per Conversion ($):", min_value=0.0, value=50.0, step=5.0, key='atr_rev_per_conv')

    df_atr['Estimated_Revenue'] = df_atr['Attributed_Conversions'] * avg_revenue_per_conversion
    # Calculate ROI, handling division by zero for cost
    df_atr['ROI (%)'] = np.where(
        df_atr['Cost'] > 0,
        ((df_atr['Estimated_Revenue'] - df_atr['Cost']) / df_atr['Cost']) * 100,
        0 # Assign 0 ROI if cost is 0
    )


    st.write("#### Data with Attributed Conversions and ROI:")
    st.dataframe(df_atr.round(2))

    # --- Summary View ---
    st.subheader("Summary by Channel")
    channel_summary = df_atr.groupby('Channel').agg(
        Total_Clicks=('Clicks', 'sum'),
        Total_Cost=('Cost', 'sum'),
        Total_Conversions=('Conversions', 'sum'),
        Total_Attributed_Conversions=('Attributed_Conversions', 'sum'),
        Total_Estimated_Revenue=('Estimated_Revenue', 'sum')
    ).reset_index()

    # Recalculate overall ROI for the summary
    channel_summary['Overall ROI (%)'] = np.where(
        channel_summary['Total_Cost'] > 0,
        ((channel_summary['Total_Estimated_Revenue'] - channel_summary['Total_Cost']) / channel_summary['Total_Cost']) * 100,
        0
    )
    st.dataframe(channel_summary.round(2))

    # --- Visualization ---
    st.subheader("Channel ROI Comparison")
    chart_roi = alt.Chart(channel_summary).mark_bar().encode(
        x=alt.X('Channel:N', sort='-y'),
        y=alt.Y('Overall ROI (%):Q'),
        color='Channel:N',
        tooltip=['Channel', alt.Tooltip('Overall ROI (%)', format='.1f'), alt.Tooltip('Total_Estimated_Revenue', format=',.0f'), alt.Tooltip('Total_Cost', format=',.0f')]
    ).properties(
        title='Overall ROI by Channel'
    ).interactive()
    st.altair_chart(chart_roi, use_container_width=True)


def audience_segmentation_personalization_page():
    st.title("Audience Segmentation and Personalization (Conceptual Demo)")
    st.info("Uses sample audience data and *placeholder* logic for segmentation and ad personalization.")

    # Sample audience data
    @st.cache_data
    def generate_audience_data(num_users=100):
        np.random.seed(42)
        ages = np.random.randint(18, 65, num_users)
        genders = np.random.choice(['Male', 'Female', 'Other'], num_users, p=[0.48, 0.48, 0.04])
        locations = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Online'], num_users, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        interests_list = ['Technology', 'Fashion', 'Sports', 'Travel', 'Food', 'Books', 'Gaming']
        interests = [list(np.random.choice(interests_list, np.random.randint(1, 4), replace=False)) for _ in range(num_users)]
        behavior_list = ['Frequent online shopper', 'Active social media user', 'Reads news daily', 'Streams videos often', 'Infrequent user']
        behavior = [list(np.random.choice(behavior_list, np.random.randint(1, 3), replace=False)) for _ in range(num_users)]
        data = {
            'User_ID': range(1, num_users + 1),
            'Age': ages,
            'Gender': genders,
            'Location': locations,
            'Interests': interests,
            'Online_Behavior': behavior
        }
        return pd.DataFrame(data)

    df_aud = generate_audience_data()

    st.subheader("Sample Audience Data")
    st.dataframe(df_aud.head())

    # --- Placeholder Segmentation ---
    st.subheader("Audience Segmentation (Placeholder Logic)")
    # Simple rule-based segmentation for demonstration
    def segment_audience_simple(row):
        if 'Technology' in row['Interests'] and row['Age'] < 35:
            return 'Young Tech Enthusiasts'
        elif 'Fashion' in row['Interests'] and row['Location'] in ['New York', 'Los Angeles']:
            return 'Urban Fashion Shoppers'
        elif 'Travel' in row['Interests'] and row['Age'] >= 40:
            return 'Mature Travelers'
        elif 'Gaming' in row['Interests']:
            return 'Gamers'
        else:
            return 'General Audience' # Default segment

    df_aud['Segment'] = df_aud.apply(segment_audience_simple, axis=1)

    st.write("Segmented Audience Data (with placeholder segments):")
    st.dataframe(df_aud[['User_ID', 'Age', 'Gender', 'Location', 'Segment']].head())

    # --- Display Segment Counts ---
    st.subheader("Segment Distribution")
    segment_counts = df_aud['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    chart_seg = alt.Chart(segment_counts).mark_bar().encode(
        x=alt.X('Segment:N', sort='-y'),
        y='Count:Q',
        color='Segment:N',
        tooltip=['Segment', 'Count']
    ).properties(
        title='Audience Segment Sizes'
    )
    st.altair_chart(chart_seg, use_container_width=True)

    # --- Placeholder Personalization ---
    st.subheader("Personalized Ad Recommendations (Placeholder)")
    # Simple mapping from segment to ad content suggestions
    personalized_ads_map = {
        'Young Tech Enthusiasts': "Ad: Latest Gadgets Sale! Focus on features & specs. Platform: Tech Blogs, YouTube Tech Channels.",
        'Urban Fashion Shoppers': "Ad: New Season Arrivals! Highlight style & brands. Platform: Instagram, Fashion Magazines.",
        'Mature Travelers': "Ad: Luxury Cruise Deals! Emphasize comfort & experience. Platform: Travel Websites, Facebook Groups.",
        'Gamers': "Ad: Pre-order New Game! Offer exclusive content. Platform: Twitch, Gaming Forums.",
        'General Audience': "Ad: Brand Awareness Campaign. Broad appeal message. Platform: General News Sites, Facebook."
    }

    selected_segment_pers = st.selectbox("Select a segment to view ad recommendations:", df_aud['Segment'].unique(), key='aud_select_seg')

    if selected_segment_pers:
        recommendation = personalized_ads_map.get(selected_segment_pers, "No specific recommendation for this segment.")
        st.write(f"#### Recommendations for: {selected_segment_pers}")
        st.info(recommendation)

        # Show some sample users from the selected segment
        st.write("Sample Users in this Segment:")
        st.dataframe(df_aud[df_aud['Segment'] == selected_segment_pers].head())


def supply_chain_ad_dashboard_page():
    st.title("Supply Chain & Ad Campaign Dashboard (Demo)")
    st.info("Displays multiple sample dataframes related to supply chain ops and ad campaigns. Includes placeholder functions for advanced features.")

    # --- Sample Data Generation ---
    @st.cache_data
    def generate_supply_ad_data():
        np.random.seed(42)
        products = ['Widget A', 'Gadget B', 'Thingamajig C']
        inventory_data = {
            'Product': products,
            'Inventory_Level': np.random.randint(50, 500, 3),
            'Reorder_Point': np.random.randint(50, 150, 3)
        }
        production_data = {
            'Product': products,
            'Production_Status': np.random.choice(['On Track', 'Delayed', 'Completed', 'Planning'], 3, replace=False),
            'Est_Completion': [datetime.today().date() + timedelta(days=np.random.randint(5,30)) for _ in range(3)]
        }
        suppliers = ['Supplier X', 'Supplier Y', 'Supplier Z', 'Supplier W']
        supplier_performance_data = {
            'Supplier': suppliers,
            'OnTime_Delivery (%)': np.random.randint(80, 100, 4),
            'Quality_Rating (1-5)': np.random.uniform(3.5, 5.0, 4).round(1)
        }
        campaigns = ['Spring Sale', 'Summer Promo', 'Q4 Push']
        ad_campaign_data = {
            'Campaign': campaigns,
            'Status': np.random.choice(['Active', 'Planning', 'Completed'], 3, replace=False),
            'Spend ($)': np.random.randint(2000, 10000, 3),
            'Conversions': np.random.randint(50, 300, 3)
        }

        inventory_df = pd.DataFrame(inventory_data)
        production_df = pd.DataFrame(production_data)
        supplier_performance_df = pd.DataFrame(supplier_performance_data)
        ad_campaign_df = pd.DataFrame(ad_campaign_data)

        # Add simple calculated fields
        inventory_df['Status'] = np.where(inventory_df['Inventory_Level'] < inventory_df['Reorder_Point'], 'Below Reorder', 'OK')
        ad_campaign_df['Cost Per Conversion ($)'] = (ad_campaign_df['Spend ($)'] / ad_campaign_df['Conversions']).round(2)

        return inventory_df, production_df, supplier_performance_df, ad_campaign_df

    inventory_df, production_df, supplier_performance_df, ad_campaign_df = generate_supply_ad_data()

    # --- Dashboard Display ---
    st.sidebar.header("Dashboard Options")
    option = st.sidebar.radio("Select Dashboard Section",
                              ["Real-time Monitoring",
                               "Trigger Alerts (Placeholder)",
                               "Scenario Analysis (Placeholder)",
                               "Continuous Improvement (Placeholder)"], key='sc_option')

    if option == "Real-time Monitoring":
        st.header("Real-time Monitoring")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Inventory Levels")
            st.dataframe(inventory_df)
            low_inventory = inventory_df[inventory_df['Status'] == 'Below Reorder']
            if not low_inventory.empty:
                st.warning(f"Low Inventory Alert: {', '.join(low_inventory['Product'].tolist())}")

            st.subheader("Production Status")
            st.dataframe(production_df)
            delayed_production = production_df[production_df['Production_Status'] == 'Delayed']
            if not delayed_production.empty:
                st.error(f"Production Delayed: {', '.join(delayed_production['Product'].tolist())}")

        with col2:
            st.subheader("Supplier Performance")
            st.dataframe(supplier_performance_df)
            low_perf_suppliers = supplier_performance_df[
                (supplier_performance_df['OnTime_Delivery (%)'] < 85) |
                (supplier_performance_df['Quality_Rating (1-5)'] < 4.0)
            ]
            if not low_perf_suppliers.empty:
                st.warning(f"Check Supplier Performance: {', '.join(low_perf_suppliers['Supplier'].tolist())}")


            st.subheader("Ad Campaign Summary")
            st.dataframe(ad_campaign_df)
            active_campaigns = ad_campaign_df[ad_campaign_df['Status'] == 'Active']
            if not active_campaigns.empty:
                st.info(f"Active Campaigns: {', '.join(active_campaigns['Campaign'].tolist())}")


    elif option == "Trigger Alerts (Placeholder)":
        st.header("Trigger Alerts")
        st.info("This section would contain logic to automatically trigger alerts (e.g., email, notification) based on predefined conditions (like low inventory, delayed production, poor supplier performance, low campaign ROI). Implementation requires external services or more complex Streamlit features.")
        st.write("**Example Conditions:**")
        st.write("- Inventory Level < Reorder Point for > 2 days")
        st.write("- Production Status = 'Delayed'")
        st.write("- Supplier On-Time Delivery < 80%")
        st.write("- Active Ad Campaign ROI < 1.0")

    elif option == "Scenario Analysis (Placeholder)":
        st.header("Scenario Analysis ('What-If')")
        st.info("This section would allow users to simulate different scenarios. For example: 'What if demand increases by 20%?' or 'What if Supplier Y is unavailable?' Requires underlying simulation models.")
        st.write("**Example Scenarios:**")
        demand_increase = st.slider("Simulate Demand Increase (%):", 0, 100, 10, key='sc_demand_scen')
        st.write(f"-> Run simulation to see impact on inventory and production with {demand_increase}% higher demand.")
        supplier_outage = st.selectbox("Simulate Supplier Outage:", ['None'] + suppliers, key='sc_supplier_scen')
        if supplier_outage != 'None':
            st.write(f"-> Run simulation to see impact on production and costs if {supplier_outage} is unavailable.")

    elif option == "Continuous Improvement (Placeholder)":
        st.header("Continuous Improvement")
        st.info("This section would focus on analyzing historical data and simulation results to identify areas for improvement in the supply chain and ad strategies. It could involve comparing different strategies or tracking KPIs over time.")
        st.write("**Example Analyses:**")
        st.write("- Compare historical cost per conversion across different ad campaigns.")
        st.write("- Analyze lead times for different suppliers.")
        st.write("- Track inventory turnover ratios.")
        st.write("- Evaluate the effectiveness of past production schedule adjustments.")


def demand_forecasting_page():
    st.title("Advertisement Demand Forecasting Tool (Demo)")
    st.info("Uses sample data and Linear Regression to forecast 'Sales' based on various factors.")

    # Sample data
    @st.cache_data
    def generate_demand_data():
        np.random.seed(42)
        months = pd.date_range(start='2023-01-01', periods=24, freq='M').strftime('%Y-%m')
        clicks = np.random.randint(1000, 5000, 24)
        marketing_exp = np.random.randint(5000, 15000, 24)
        price = np.random.uniform(8.0, 12.0, 24).round(2)
        competitor_activity = np.random.uniform(1.0, 5.0, 24).round(1) # Scale 1-5
        # Simple seasonality (higher in Q4, lower in Q1)
        month_num = pd.to_datetime(months).month
        seasonality = np.select(
            [month_num.isin([1,2,3]), month_num.isin([10,11,12]), True],
            [0.85, 1.15, 1.0], # Factors
            default=1.0
        ) * np.random.uniform(0.95, 1.05, 24) # Add noise

        # Generate Sales based loosely on factors
        base_sales = 500 + clicks * 0.1 + marketing_exp * 0.05 - price * 20 - competitor_activity * 10
        sales = (base_sales * seasonality * np.random.uniform(0.9, 1.1, 24)).astype(int)
        sales = np.clip(sales, 200, None) # Ensure min sales


        data = {
            'Month': months,
            'Clicks': clicks,
            'Marketing_Expenditure': marketing_exp,
            'Price': price,
            'Competitor_Activity': competitor_activity,
            'Seasonality_Factor': seasonality.round(2),
            'Sales': sales
        }
        return pd.DataFrame(data)

    df_dem = generate_demand_data()

    st.subheader("Sample Advertisement & Sales Data")
    st.dataframe(df_dem)

    # --- Model Training ---
    st.subheader("Sales Forecasting Model (Linear Regression)")
    try:
        # Select features and target variable
        features = ['Clicks', 'Marketing_Expenditure', 'Price', 'Competitor_Activity', 'Seasonality_Factor']
        target = 'Sales'
        X_dem = df_dem[features]
        y_dem = df_dem[target]

        # Handle potential NaNs (though generated data shouldn't have them)
        X_dem = X_dem.fillna(X_dem.mean())
        y_dem = y_dem.fillna(y_dem.mean())

        # Split data (optional for small demo, but good practice)
        X_train_dem, X_test_dem, y_train_dem, y_test_dem = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)

        # Train the linear regression model
        model_dem = LinearRegression()
        model_dem.fit(X_train_dem, y_train_dem)

        # Evaluate model (optional display)
        y_pred_dem = model_dem.predict(X_test_dem)
        mse_dem = mean_squared_error(y_test_dem, y_pred_dem)
        r2_dem = r2_score(y_test_dem, y_pred_dem)

        with st.expander("Show Model Evaluation Metrics (Test Set)"):
            st.metric("Mean Squared Error (MSE)", f"{mse_dem:.2f}")
            st.metric("R-squared (R2) Score", f"{r2_dem:.3f}")
            # Display coefficients for interpretability
            coeffs = pd.DataFrame(model_dem.coef_, index=features, columns=['Coefficient'])
            st.write("Model Coefficients:")
            st.dataframe(coeffs)

    except Exception as e:
        st.error(f"Error during model training: {e}")
        return

    # --- Forecasting Form ---
    st.subheader("Forecast Sales for Next Period")
    st.write("Enter the expected values for the input factors:")
    col1, col2, col3 = st.columns(3)
    with col1:
        clicks_input = st.number_input("Expected Clicks:", min_value=0, value=int(df_dem['Clicks'].mean()), key='dem_f_clicks')
        price_input = st.number_input("Expected Price ($):", min_value=0.0, value=df_dem['Price'].mean(), format="%.2f", key='dem_f_price')
    with col2:
        marketing_exp_input = st.number_input("Expected Marketing Expenditure ($):", min_value=0, value=int(df_dem['Marketing_Expenditure'].mean()), key='dem_f_mktg')
        competitor_activity_input = st.number_input("Expected Competitor Activity (1-5):", min_value=1.0, max_value=5.0, value=df_dem['Competitor_Activity'].mean(), format="%.1f", key='dem_f_comp')
    with col3:
        seasonality_input = st.number_input("Expected Seasonality Factor:", min_value=0.0, value=df_dem['Seasonality_Factor'].mean(), format="%.2f", key='dem_f_season')


    # Make demand forecast for user input
    user_input_df = pd.DataFrame([[clicks_input, marketing_exp_input, price_input, competitor_activity_input, seasonality_input]], columns=features)
    demand_forecast = model_dem.predict(user_input_df)

    st.success(f"### Predicted Sales: {demand_forecast[0]:.0f} units")


def top_cities_page():
    st.title("Top Cities by Product Category Sales (Demo)")
    st.info("Uses sample data showing sales units by city and category.")

    # Sample data
    @st.cache_data
    def generate_city_sales_data():
        np.random.seed(42)
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        categories = ['Electronics', 'Clothing', 'Home Appliances', 'Sports Equipment', 'Furniture', 'Books']
        data = {'City': cities}
        for cat in categories:
            data[cat] = np.random.randint(50, 250, len(cities)) * np.random.choice([0.8, 1.0, 1.2], len(cities)) # Add city factor
        df = pd.DataFrame(data)
        df = df.set_index('City')
        return df.astype(int)

    df_topc = generate_city_sales_data()

    st.subheader("Sample Sales Data (Units Sold)")
    st.dataframe(df_topc)

    # --- Analysis ---
    st.subheader("Find Top Cities")
    category_select = st.selectbox("Select Product Category:", df_topc.columns, key='topc_cat')
    num_top_cities = st.slider("Number of top cities to display:", min_value=1, max_value=len(df_topc), value=5, key='topc_num')

    if category_select:
        # Get top N cities for the selected category
        top_cities = df_topc[category_select].nlargest(num_top_cities)

        st.write(f"#### Top {num_top_cities} Cities for '{category_select}':")
        st.dataframe(top_cities)

        # Visualization
        st.write("#### Visualization")
        top_cities_df = top_cities.reset_index() # Convert Series to DataFrame for Altair
        top_cities_df.columns = ['City', 'Sales']

        chart = alt.Chart(top_cities_df).mark_bar().encode(
            x=alt.X('Sales:Q', title=f'Sales Units ({category_select})'),
            y=alt.Y('City:N', title='City', sort='-x'), # Sort by sales descending
            tooltip=['City', 'Sales']
        ).properties(
            title=f'Top {num_top_cities} Cities by {category_select} Sales'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)


def targeted_sales_ads_page():
    st.title("Targeted Sales Products through Advertisements (Occasion Demo)")
    st.info("Uses sample ad data and filters based on occasion mentioned in the ad name.")

    # Sample ad data
    @st.cache_data
    def generate_occasion_ad_data():
        np.random.seed(42)
        ad_names = [
            'Back to School Sale - Laptops', 'Summer Tech Deals - Phones', 'Holiday Specials - Headphones',
            'New Year Discounts - Watches', 'Spring Clearance - Tablets', 'Back to School - Backpacks',
            'Summer Fun - Cameras', 'Holiday Gifts - Gaming', 'New Year Fitness - Trackers',
            'Spring Cleaning - Vacuums'
        ]
        product_names = [
            'Laptop Model X', 'Smartphone Z', 'Headphones Pro', 'Smartwatch V', 'Tablet S',
            'Backpack Deluxe', 'Camera 4K', 'Gaming Console Next', 'Fitness Tracker Fit', 'Vacuum Clean'
        ]
        data = {
            'Ad_ID': range(101, 101 + len(ad_names)),
            'Ad_Name': ad_names,
            'Product_Name': product_names,
            'Campaign_Budget': np.random.randint(1000, 6000, len(ad_names)),
            'Target_Audience': np.random.choice(['Students', 'General', 'Gamers', 'Fitness Enthusiasts', 'Homeowners'], len(ad_names))
        }
        return pd.DataFrame(data)

    df_tsa = generate_occasion_ad_data()

    st.subheader("Sample Advertisement Data")
    st.dataframe(df_tsa)

    # Occasion keyword selection
    st.subheader("Filter Ads by Occasion Keyword")
    # Extract potential keywords from ad names for the dropdown
    keywords = set()
    common_keywords = ['Sale', 'Deals', 'Specials', 'Discounts', 'Clearance', 'Fun', 'Gifts', 'Fitness', 'Cleaning', 'Back to School', 'Summer', 'Holiday', 'New Year', 'Spring']
    for name in df_tsa['Ad_Name']:
        for word in name.split():
             # Add common keywords or words starting with a capital letter (potential occasions)
             if word in common_keywords or word.istitle():
                  keywords.add(word.replace('-', '').replace(',', '')) # Clean up
    occasion_keyword = st.selectbox("Select Occasion Keyword to Filter Ads:", sorted(list(keywords)), key='tsa_keyword')

    # Filter data based on selected occasion keyword (case-insensitive)
    if occasion_keyword:
        filtered_ads = df_tsa[df_tsa['Ad_Name'].str.contains(occasion_keyword, case=False, na=False)]

        st.write(f"#### Advertisements containing '{occasion_keyword}':")
        if filtered_ads.empty:
            st.warning(f"No ads found containing the keyword '{occasion_keyword}'.")
        else:
            st.dataframe(filtered_ads)


def merged_product_ad_page():
    st.title("Merged Product and Advertisement Data (Demo)")
    st.info("Merges sample product and advertisement dataframes.")

    # Sample product data
    product_data = {
        'Product_ID': [1, 2, 3, 4, 5, 6],
        'Product_Name': ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch', 'Tablet', 'Camera'],
        'Category': ['Electronics', 'Electronics', 'Accessories', 'Wearable', 'Electronics', 'Electronics'],
        'Price': [1000, 800, 150, 300, 500, 600],
        'Stock': [50, 120, 200, 80, 100, 30]
    }
    product_df = pd.DataFrame(product_data)

    # Sample advertisement data
    advertisement_data = {
        'Ad_ID': [101, 102, 103, 104, 105, 106, 107],
        'Ad_Name': ['Back to School Sale', 'Summer Tech Deals', 'Holiday Specials', 'New Year Discounts', 'Spring Clearance', 'Flash Sale', 'Summer Travel'],
        'Product_ID': [1, 2, 3, 4, 5, 1, 6], # Note: Product 1 advertised twice, Product 6 advertised once
        'Campaign_Budget': [5000, 3000, 2000, 4000, 3500, 2500, 3000],
        'Start_Date': pd.to_datetime(['2024-08-01', '2024-06-01', '2024-11-15', '2024-01-01', '2024-04-01', '2024-09-01', '2024-07-01'])
    }
    advertisement_df = pd.DataFrame(advertisement_data)

    st.subheader("Sample Product Data")
    st.dataframe(product_df)

    st.subheader("Sample Advertisement Data")
    st.dataframe(advertisement_df)

    # --- Merge Data ---
    st.subheader("Merged Product and Advertisement Data")
    # Use a left merge to keep all ads and add product info where available
    merged_df = pd.merge(advertisement_df, product_df, on='Product_ID', how='left')

    # Handle cases where Product_ID in ads doesn't exist in products (would result in NaNs)
    # merged_df.fillna({'Product_Name': 'Unknown', 'Category': 'Unknown', 'Price': 0, 'Stock': 0}, inplace=True) # Option to fill NaNs

    st.dataframe(merged_df)

    st.write("**Merge Type:** Left Merge (keeping all advertisements, adding matching product info).")
    missing_products = merged_df['Product_Name'].isnull().sum()
    if missing_products > 0:
        st.warning(f"Note: {missing_products} ad(s) have Product IDs not found in the Product Data.")


def personalized_adv_segmentation_adv_page():
    st.title("Personalized Advertising and Customer Segmentation (Advanced Demo)")
    st.info("Uses sample customer data, allows filtering, performs grouping, generates recommendations, and displays various chart types including Geo.")

    # Create sample customer data
    @st.cache_data
    def generate_advanced_customer_data(num_customers=50):
        np.random.seed(42)
        ids = range(1, num_customers + 1)
        ages = np.random.randint(18, 70, num_customers)
        genders = np.random.choice(['Male', 'Female', 'Other'], num_customers, p=[0.48, 0.48, 0.04])
        # More diverse locations with coordinates
        locations_data = {
            'New York': (40.7128, -74.0060), 'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298), 'Seattle': (47.6062, -122.3321),
            'Houston': (29.7604, -95.3698), 'London': (51.5074, -0.1278),
            'Paris': (48.8566, 2.3522), 'Tokyo': (35.6895, 139.6917)
        }
        location_names = np.random.choice(list(locations_data.keys()), num_customers)
        latitudes = [locations_data[loc][0] + np.random.normal(0, 0.05) for loc in location_names] # Add noise
        longitudes = [locations_data[loc][1] + np.random.normal(0, 0.05) for loc in location_names]

        purchase_cats = ['Electronics', 'Clothing', 'Home Appliances', 'Sports Equipment', 'Books', 'Furniture', 'Travel', 'Groceries']
        purchase_history = [list(np.random.choice(purchase_cats, np.random.randint(1, 5), replace=False)) for _ in range(num_customers)]

        social_platforms = ['Facebook', 'Twitter', 'Instagram', 'LinkedIn', 'TikTok', 'Pinterest']
        social_media_activity = [list(np.random.choice(social_platforms, np.random.randint(0, 4), replace=False)) for _ in range(num_customers)] # Some users might not be active

        df = pd.DataFrame({
            'Customer ID': ids, 'Age': ages, 'Gender': genders, 'Location': location_names,
            'Latitude': latitudes, 'Longitude': longitudes,
            'Purchase History': purchase_history, 'Social Media Activity': social_media_activity
        })
        return df

    df_pers = generate_advanced_customer_data()

    st.write("### Sample Customer Data:")
    st.dataframe(df_pers.head())

    # --- Sidebar Filtering ---
    st.sidebar.header("Customer Filtering & Segmentation")
    age_filter = st.sidebar.slider("Age Range Filter:", 18, 70, (18, 70), 1, key='pers_age_filter')
    gender_filter = st.sidebar.multiselect("Gender Filter:", df_pers['Gender'].unique(), default=df_pers['Gender'].unique(), key='pers_gender_filter')
    location_filter = st.sidebar.multiselect("Location Filter:", df_pers['Location'].unique(), default=df_pers['Location'].unique(), key='pers_location_filter')

    # Apply filters
    filtered_data = df_pers[
        (df_pers['Age'].between(age_filter[0], age_filter[1])) &
        (df_pers['Gender'].isin(gender_filter)) &
        (df_pers['Location'].isin(location_filter))
    ].copy() # Use copy to avoid SettingWithCopyWarning

    st.write(f"### Filtered Customer Data ({len(filtered_data)} customers):")
    st.dataframe(filtered_data.head())

    # --- Analyze Segments (Grouped by Location for simplicity) ---
    st.header("Customer Segments (Grouped by Location)")
    # Simple segmentation based on location for this demo
    if not filtered_data.empty:
        segments = filtered_data.groupby('Location')
        for location, group in segments:
            with st.expander(f"Segment: {location} ({len(group)} customers)"):
                st.dataframe(group[['Customer ID', 'Age', 'Gender', 'Purchase History', 'Social Media Activity']])

                # --- Personalized Advertising Recommendations ---
                st.write(f"#### Personalized Recommendations for {location} Segment:")
                # Aggregate interests and platforms for the group
                all_purchases = set(item for sublist in group['Purchase History'] for item in sublist)
                all_social = set(item for sublist in group['Social Media Activity'] for item in sublist if sublist) # Handle empty lists

                st.write("**Targeted Advertising Channels:**")
                if all_social:
                    st.write(f"- {', '.join(list(all_social))}")
                else:
                    st.write("- (No dominant social media activity identified in this segment)")

                st.write("**Recommended Product Categories:**")
                if all_purchases:
                     st.write(f"- {', '.join(list(all_purchases))}")
                else:
                     st.write("- (No dominant purchase history identified)")

    else:
        st.warning("No customers match the current filter criteria.")


    # --- Chart Creation from Filtered Data ---
    st.header("Visualize Filtered Data")
    if not filtered_data.empty:
        chart_type = st.selectbox("Select Chart Type:", ["Bar Chart (Gender)", "Pie Chart (Location)", "Geo Chart (Location)", "Age Distribution"], key='pers_chart_type')
        # data_column = st.selectbox("Select Data to Visualize", ["Age", "Gender", "Location"]) # Simplified selection based on chart type

        if chart_type == "Bar Chart (Gender)":
            chart = alt.Chart(filtered_data).mark_bar().encode(
                x=alt.X('Gender:N', title='Gender'),
                y=alt.Y('count()', title='Number of Customers'),
                color='Gender:N',
                tooltip=['Gender', 'count()']
            ).properties(
                title='Customer Count by Gender'
            )
            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "Pie Chart (Location)":
            pie_data = filtered_data['Location'].value_counts().reset_index()
            pie_data.columns = ['Location', 'Count']
            chart = alt.Chart(pie_data).mark_arc(outerRadius=120).encode(
                theta=alt.Theta(field='Count', type='quantitative', stack=True),
                color=alt.Color(field='Location', type='nominal'),
                tooltip=['Location', 'Count']
            ).properties(
                title='Customer Distribution by Location'
            )
            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "Geo Chart (Location)":
            st.write("Customer Locations Map (Size represents number of customers)")
            # Count customers per location for sizing
            location_counts = filtered_data.groupby(['Location', 'Latitude', 'Longitude']).size().reset_index(name='Count')

            # Pydeck chart
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=location_counts,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]', # RGBA color
                get_radius='Count * 10000', # Adjust multiplier for appropriate radius
                pickable=True,
                auto_highlight=True
            )
            # Set initial view state
            view_state = pdk.ViewState(
                latitude=location_counts['Latitude'].mean(),
                longitude=location_counts['Longitude'].mean(),
                zoom=1, # Adjust zoom level
                pitch=30
            )
            # Render map
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9', # Or other map styles
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "{Location}\nCustomers: {Count}"}
            ))

        elif chart_type == "Age Distribution":
             chart = alt.Chart(filtered_data).mark_bar().encode(
                alt.X("Age:Q", bin=alt.Bin(maxbins=10), title="Age Group"), # Bin ages
                y=alt.Y('count()', title='Number of Customers'),
                tooltip=[alt.Tooltip("Age:Q", bin=alt.Bin(maxbins=10), title="Age Group"), 'count()']
            ).properties(
                title='Customer Age Distribution'
            ).interactive()
             st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("Cannot create charts with no filtered data.")


def revenue_ad_analysis_page():
    st.title('Revenue and Advertising Analysis (Dashboard Demo)')
    st.info("Uses sample data and provides various visualizations like Multi-Line, Waterfall, and Heatmap. Includes a basic revenue prediction model.")

    # --- Sample Data Generation ---
    @st.cache_data
    def generate_revenue_ad_data():
        np.random.seed(42)
        months = pd.date_range(start='2023-01-01', periods=24, freq='M')
        data = {'Month': months.strftime('%Y-%m')}
        products = ['A', 'B', 'C']
        prices = {'A': 10, 'B': 15, 'C': 20}
        for p in products:
            sales = np.random.randint(100, 500 + int(p=='C')*100, 24) # Product C slightly higher sales
            ads = np.random.randint(10, 50 + int(p=='B')*10, 24) # Product B slightly higher ads
            data[f'Product_{p}_Sales'] = sales
            data[f'Product_{p}_Ads'] = ads
            data[f'Product_{p}_Revenue'] = sales * prices[p] * np.random.uniform(0.95, 1.05, 24) # Revenue = Sales * Price + noise
        df = pd.DataFrame(data)
        df['Total_Revenue'] = df[[f'Product_{p}_Revenue' for p in products]].sum(axis=1)
        df['Total_Ads'] = df[[f'Product_{p}_Ads' for p in products]].sum(axis=1)
        df['Month_Date'] = pd.to_datetime(months) # Add date column for time series plots
        return df

    df_rev = generate_revenue_ad_data()

    st.write("""
    ### Visualization Dashboard
    Visualize revenue generated from sales products based on advertising expenditure.
    """)

    # --- Filtering ---
    st.sidebar.header("Filter Data")
    all_months = df_rev['Month'].tolist()
    selected_months = st.sidebar.multiselect("Select months to display:", all_months, default=all_months[-12:], key='rev_months') # Default last 12 months
    filtered_df = df_rev[df_rev['Month'].isin(selected_months)].copy()

    if filtered_df.empty:
        st.warning("No data selected. Please select months in the sidebar.")
        return

    # --- Chart Selection and Display ---
    st.header("Revenue & Ad Spend Visualization")
    chart_type = st.selectbox("Select chart type:", ["Multi-Line Chart", "Waterfall Chart (Total Revenue)", "Heatmap (Revenue by Product)"], key='rev_chart_type')

    if chart_type == "Multi-Line Chart":
        st.write("Revenue and Advertising Expenditure per Product Over Selected Months")
        # Melt the DataFrame
        plot_vars = [col for col in filtered_df.columns if 'Revenue' in col or 'Ads' in col]
        df_long = filtered_df.melt(id_vars=['Month_Date', 'Month'], value_vars=plot_vars,
                                   var_name='Metric', value_name='Value')

        # Define the chart
        chart = alt.Chart(df_long).mark_line(point=True).encode(
            x=alt.X('Month_Date:T', title='Month'),
            y=alt.Y('Value:Q', title='Amount ($ or Units)'),
            color='Metric:N',
            tooltip=['Month', 'Metric', alt.Tooltip('Value:Q', format=',.0f')]
        ).properties(
            title='Revenue & Ad Spend Over Time'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    elif chart_type == "Waterfall Chart (Total Revenue)":
        st.write("Cumulative Change in Total Revenue Across Selected Months")
        # Prepare data for waterfall chart
        waterfall_df = filtered_df[['Month', 'Total_Revenue']].copy()
        # Calculate change from previous month
        waterfall_df['Revenue Change'] = waterfall_df['Total_Revenue'].diff().fillna(waterfall_df['Total_Revenue'].iloc[0]) # First month change is the month's total

        # Create waterfall chart using Plotly
        fig = go.Figure(go.Waterfall(
            name="Revenue Flow", orientation="v",
            measure=["relative"] * len(waterfall_df), # All are relative changes month-to-month
            x=waterfall_df['Month'],
            textposition="outside",
            text=waterfall_df['Revenue Change'].apply(lambda x: f"{x:,.0f}"), # Display change on bar
            y=waterfall_df['Revenue Change'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        # Add a final 'Total' bar (optional)
        # fig.add_trace(go.Waterfall(measure=['total'], x=['Total'], y=[waterfall_df['Total_Revenue'].iloc[-1]]))

        fig.update_layout(
            title="Monthly Change in Total Revenue",
            showlegend=False,
            yaxis_title="Change in Revenue ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap (Revenue by Product)":
        st.write("Monthly Revenue Heatmap by Product")
        heatmap_data = filtered_df.melt(id_vars=['Month'], value_vars=[col for col in filtered_df if 'Revenue' in col and 'Product' in col],
                                        var_name='Product Revenue Metric', value_name='Revenue')
        # Extract product name
        heatmap_data['Product'] = heatmap_data['Product Revenue Metric'].str.extract(r'Product_([A-Z])_Revenue')[0]
        heatmap_data = heatmap_data.pivot(index='Product', columns='Month', values='Revenue')

        if not heatmap_data.empty:
            fig_hm, ax_hm = plt.subplots(figsize=(12, 4)) # Adjust size
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="viridis", linewidths=.5, ax=ax_hm)
            ax_hm.set_title("Monthly Revenue by Product")
            ax_hm.set_xlabel("Month")
            ax_hm.set_ylabel("Product")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig_hm)
            plt.close(fig_hm)
        else:
            st.warning("Could not generate heatmap data.")


    # --- Revenue Prediction Model ---
    with st.expander("Predict Future Revenue (Simple Model)"):
        st.info("Trains a basic Linear Regression model using past Total Ads spend to predict next month's Total Revenue.")
        try:
            # Prepare data: Use lagged Ad spend to predict Revenue
            df_pred = df_rev[['Month_Date', 'Total_Ads', 'Total_Revenue']].copy()
            df_pred['Lagged_Total_Ads'] = df_pred['Total_Ads'].shift(1)
            df_pred = df_pred.dropna()

            if len(df_pred) < 5:
                st.warning("Not enough historical data (need at least 5 months with Ads and Revenue) for prediction.")
            else:
                X_pred = df_pred[['Lagged_Total_Ads']]
                y_pred_target = df_pred['Total_Revenue']

                # Train model
                model_pred = LinearRegression()
                model_pred.fit(X_pred, y_pred_target)

                st.write(f"Model Fit: Revenue ≈ {model_pred.coef_[0]:.2f} * Previous_Month_Ad_Spend + {model_pred.intercept_:.2f}")

                # Predict for next month
                last_ad_spend = df_rev['Total_Ads'].iloc[-1]
                if pd.isna(last_ad_spend):
                    st.warning("Last month's ad spend is missing. Cannot predict next month.")
                else:
                    next_month_prediction = model_pred.predict([[last_ad_spend]])
                    next_month_date = df_rev['Month_Date'].iloc[-1] + pd.DateOffset(months=1)
                    st.metric(f"Predicted Total Revenue for {next_month_date.strftime('%Y-%m')}", f"${next_month_prediction[0]:,.0f}",
                              help=f"Based on previous month's Ad Spend of ${last_ad_spend:,.0f}")

        except Exception as e:
            st.error(f"Error during revenue prediction: {e}")


def sales_revenue_dashboard_page():
    st.title('Sales and Revenue Analysis Dashboard (Alt. Demo)')
    st.info("Uses sample data and provides visualizations including cumulative charts and prediction with RandomForest.")

    # Simulate a dataset
    @st.cache_data
    def generate_sales_rev_data(num_days=100):
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
        sales = np.random.randint(80, 300, size=num_days) + np.linspace(0, 50, num_days) # Trend
        revenue = sales * np.random.uniform(10, 30, size=num_days) + np.random.normal(0, 100, num_days)
        ad_spend = np.random.uniform(50, 200, size=num_days) + sales * 0.1 # Ad spend related to sales
        df = pd.DataFrame({
            'Date': dates,
            'Sales': sales.astype(int),
            'Revenue': revenue.round(2),
            'Advertisement_Spend': ad_spend.round(2)
        })
        df['Cumulative_Sales'] = df['Sales'].cumsum()
        df['Cumulative_Revenue'] = df['Revenue'].cumsum()
        return df

    df_srd = generate_sales_rev_data()

    # --- Sidebar Filtering ---
    st.sidebar.header('Filter Data by Date')
    min_date_srd = df_srd['Date'].min().date()
    max_date_srd = df_srd['Date'].max().date()
    start_date_srd = st.sidebar.date_input('Start Date', min_date_srd, min_value=min_date_srd, max_value=max_date_srd, key='srd_start')
    end_date_srd = st.sidebar.date_input('End Date', max_date_srd, min_value=min_date_srd, max_value=max_date_srd, key='srd_end')

    # Convert dates for filtering
    start_date_dt = pd.to_datetime(start_date_srd)
    end_date_dt = pd.to_datetime(end_date_srd)

    filtered_df_srd = df_srd[(df_srd['Date'] >= start_date_dt) & (df_srd['Date'] <= end_date_dt)].copy()

    if filtered_df_srd.empty:
        st.warning("No data available for the selected date range.")
        return

    # --- Display Data ---
    if st.checkbox('Show Filtered Data', key='srd_show_data'):
        st.subheader("Filtered Sales and Revenue Data")
        st.dataframe(filtered_df_srd)

    # --- Visualizations ---
    st.header("Visual Analysis")

    # Multi-line Chart (Daily Sales & Revenue)
    st.subheader("Daily Sales and Revenue")
    df_melted_daily = filtered_df_srd.melt(id_vars=['Date'], value_vars=['Sales', 'Revenue'], var_name='Metric', value_name='Value')
    chart_daily = alt.Chart(df_melted_daily).mark_line(point=False).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title='Value'),
        color='Metric:N',
        tooltip=['Date', 'Metric', alt.Tooltip('Value:Q', format=',.0f')]
    ).properties(
        title='Daily Sales and Revenue Over Time'
    ).interactive()
    st.altair_chart(chart_daily, use_container_width=True)

    # Waterfall Chart (Cumulative Revenue) - Using the diff approach
    st.subheader("Cumulative Revenue Waterfall")
    waterfall_data = filtered_df_srd[['Date', 'Revenue']].copy()
    waterfall_data['Revenue Change'] = waterfall_data['Revenue'].diff().fillna(waterfall_data['Revenue'].iloc[0])
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(waterfall_data),
        x=waterfall_data['Date'].dt.strftime('%Y-%m-%d'),
        y=waterfall_data['Revenue Change'],
        text=waterfall_data['Revenue Change'].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        name="Daily Change"
        # connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(title="Daily Change in Revenue")
    st.plotly_chart(fig_waterfall, use_container_width=True)


    # Heat Map (Correlations)
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered_df_srd.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
         corr_matrix = filtered_df_srd[numeric_cols].corr()
         fig_hm, ax_hm = plt.subplots()
         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_hm)
         ax_hm.set_title("Feature Correlation Heatmap")
         st.pyplot(fig_hm)
         plt.close(fig_hm)
    else:
         st.info("Not enough numeric columns for a heatmap.")


    # --- Prediction for next 30 days ---
    with st.expander('Show Revenue Prediction for Next 30 Days'):
        try:
            # Prepare the data - Use lagged features and other metrics
            df_pred_srd = df_srd.copy()
            df_pred_srd['DayOfYear'] = df_pred_srd['Date'].dt.dayofyear
            df_pred_srd['Month'] = df_pred_srd['Date'].dt.month
            df_pred_srd['Lagged_Revenue_1'] = df_pred_srd['Revenue'].shift(1)
            df_pred_srd['Lagged_Sales_1'] = df_pred_srd['Sales'].shift(1)
            df_pred_srd = df_pred_srd.dropna()

            features_srd = ['DayOfYear', 'Month', 'Lagged_Revenue_1', 'Lagged_Sales_1', 'Advertisement_Spend']
            target_srd = 'Revenue'

            if len(df_pred_srd) < 10:
                 st.warning("Not enough data for reliable prediction model.")
                 return

            X_srd = df_pred_srd[features_srd]
            y_srd = df_pred_srd[target_srd]

            # Use all available historical data for training the final model
            model_srd = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            model_srd.fit(X_srd, y_srd)

            # Generate feature values for the next 30 days
            last_date = df_srd['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
            future_df['Month'] = future_df['Date'].dt.month

            # Need lagged values and ad spend for future dates - requires assumptions/forecasts
            # Simple assumption: Use last known values or mean for lagged, assume avg ad spend
            last_revenue = df_srd['Revenue'].iloc[-1]
            last_sales = df_srd['Sales'].iloc[-1]
            avg_ad_spend = df_srd['Advertisement_Spend'].mean()

            future_df['Lagged_Revenue_1'] = last_revenue # Simplistic: assume yesterday was last known day
            future_df['Lagged_Sales_1'] = last_sales
            future_df['Advertisement_Spend'] = avg_ad_spend # Simplistic: assume avg spend

            # Predict
            future_predictions = model_srd.predict(future_df[features_srd])
            future_df['Revenue_Prediction'] = future_predictions

            st.write("#### Predicted Revenue (Next 30 Days)")
            st.dataframe(future_df[['Date', 'Revenue_Prediction']].round(2))

            # Plotting future predictions
            plot_hist_df = df_srd[['Date', 'Revenue']].rename(columns={'Revenue':'Actual Revenue'})
            plot_fut_df = future_df[['Date', 'Revenue_Prediction']].rename(columns={'Revenue_Prediction':'Predicted Revenue'})
            plot_combined = pd.concat([plot_hist_df.melt(id_vars=['Date']), plot_fut_df.melt(id_vars=['Date'])])


            chart_pred = alt.Chart(plot_combined).mark_line(point=False).encode(
                x=alt.X('Date:T'),
                y=alt.Y('value:Q', title='Revenue'),
                color='variable:N', # Differentiates Actual vs Predicted
                tooltip=['Date', 'variable', alt.Tooltip('value:Q', format=',.0f')]
            ).properties(
                title='Historical and Predicted Revenue'
            ).interactive()
            st.altair_chart(chart_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)


def dynamic_chart_visualization_page():
    st.title('Dynamic Chart Visualization (Demo)')
    st.info("Select different chart types to visualize sample Area Sales/Revenue/Ad data.")

    # Sample data
    @st.cache_data
    def generate_area_data():
        np.random.seed(42)
        areas = ['North', 'South', 'East', 'West', 'Central', 'Online']
        data = {
            'Area': areas,
            'Sales': np.random.randint(100, 500, len(areas)),
            'Revenue': np.random.randint(1000, 8000, len(areas)),
            'Advertisement': np.random.randint(200, 1500, len(areas))
        }
        df = pd.DataFrame(data)
        # Make data somewhat consistent (e.g., higher sales -> higher revenue)
        df['Revenue'] = (df['Sales'] * np.random.uniform(15, 30, len(areas))).astype(int)
        df['Advertisement'] = (df['Revenue'] * np.random.uniform(0.05, 0.2, len(areas))).astype(int)
        return df

    df_dyn = generate_area_data()

    st.subheader("Sample Data by Area")
    st.dataframe(df_dyn)

    st.header("Chart Visualization")
    chart_type = st.selectbox(
        "Select the type of chart to display:",
        ["Bar Chart (Sales)", "Line Chart (Revenue)", "Pie Chart (Revenue %)",
         "Radar Chart (All Metrics)", "Column Chart (Ads)", "Histogram (Sales Dist - If applicable)"], # Added titles for clarity
        key='dyn_chart_type'
    )

    # Generate and display the selected chart using Altair or Plotly
    if chart_type == "Bar Chart (Sales)":
        chart = alt.Chart(df_dyn).mark_bar().encode(
            x=alt.X('Area:N', sort='-y'),
            y=alt.Y('Sales:Q'),
            color='Area:N',
            tooltip=['Area', 'Sales']
        ).properties(title='Sales by Area')
        st.altair_chart(chart, use_container_width=True)

    elif chart_type == "Line Chart (Revenue)":
        # Line chart might not be ideal for categorical area data unless ordered meaningfully
        # Using bar chart instead for revenue comparison
        chart = alt.Chart(df_dyn).mark_bar().encode(
             x=alt.X('Area:N', sort='-y'),
             y=alt.Y('Revenue:Q'),
             color='Area:N',
             tooltip=['Area', alt.Tooltip('Revenue', format=',.0f')]
         ).properties(title='Revenue by Area')
        st.altair_chart(chart, use_container_width=True)


    elif chart_type == "Pie Chart (Revenue %)":
        # Ensure Revenue > 0 for percentage calculation
        df_pie = df_dyn[df_dyn['Revenue'] > 0].copy()
        if not df_pie.empty:
            chart = alt.Chart(df_pie).mark_arc(outerRadius=120).encode(
                theta=alt.Theta(field="Revenue", type="quantitative", stack=True, title="Revenue"),
                color=alt.Color(field="Area", type="nominal"),
                tooltip=['Area', alt.Tooltip('Revenue', format=',.0f')]
            ).properties(title='Revenue Distribution by Area')
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No positive revenue data to display in pie chart.")

    elif chart_type == "Radar Chart (All Metrics)":
        # Need numeric data only, melt for Plotly Express line_polar
        df_radar = df_dyn.melt(id_vars=['Area'], var_name='Metric', value_name='Value')
        fig_radar = px.line_polar(df_radar, r='Value', theta='Metric', color='Area', line_close=True,
                                  title='Performance Metrics by Area (Radar Chart)')
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar, use_container_width=True)

    elif chart_type == "Column Chart (Ads)": # Same as bar chart essentially
        chart = alt.Chart(df_dyn).mark_bar().encode(
            x=alt.X('Area:N', sort='-y'),
            y=alt.Y('Advertisement:Q', title='Advertising Spend ($)'),
            color='Area:N',
            tooltip=['Area', alt.Tooltip('Advertisement', format=',.0f')]
        ).properties(title='Advertising Spend by Area')
        st.altair_chart(chart, use_container_width=True)

    elif chart_type == "Histogram (Sales Dist - If applicable)":
        st.info("Histogram shows the distribution of a single metric. Using 'Sales' from the sample data.")
        chart = alt.Chart(df_dyn).mark_bar().encode(
            alt.X("Sales:Q", bin=alt.Bin(maxbins=5), title="Sales Bins"), # Bin the sales data
            y=alt.Y('count()', title='Number of Areas'),
            tooltip=[alt.Tooltip("Sales:Q", bin=alt.Bin(maxbins=5), title="Sales Bin"), 'count()']
        ).properties(title='Distribution of Sales Values Across Areas')
        st.altair_chart(chart, use_container_width=True)


def campaign_ticket_booking_page():
    st.title('Campaign Ticket Booking System (Demo)')
    st.info("Simple booking interface using sample campaign data.")

    # Sample campaign data
    @st.cache_data
    def generate_booking_campaign_data():
        data = {
            'Campaign_ID': [101, 102, 103, 104, 105],
            'Campaign_Name': ['Music Fest 2024', 'Food & Wine Expo', 'Tech Summit Global', 'Art Fair Weekend', 'Charity Gala Night'],
            'Date': pd.to_datetime(['2024-07-15', '2024-08-20', '2024-09-10', '2024-10-05', '2024-11-22']).strftime('%Y-%m-%d'),
            'Location': ['Central Park, NY', 'Convention Center, LA', 'Online', 'Art Gallery, Chicago', 'Grand Ballroom, Houston'],
            'Price': [75, 50, 299, 25, 500]
        }
        return pd.DataFrame(data)

    df_campaigns = generate_booking_campaign_data()

    st.header('Available Campaigns')
    st.dataframe(df_campaigns)

    # --- Booking Form ---
    st.header('Book Your Campaign Ticket')
    with st.form(key='booking_form'):
        selected_campaign_id = st.selectbox('Select Campaign:', df_campaigns['Campaign_ID'], format_func=lambda x: f"{x} - {df_campaigns.loc[df_campaigns['Campaign_ID']==x, 'Campaign_Name'].iloc[0]}")
        name = st.text_input('Your Name:')
        email = st.text_input('Your Email:')
        num_tickets = st.number_input('Number of Tickets:', min_value=1, max_value=10, value=1)

        submitted = st.form_submit_button('Book Ticket')

        if submitted:
            if not name or not email:
                st.warning("Please enter your Name and Email.")
            elif '@' not in email or '.' not in email: # Basic email validation
                 st.warning("Please enter a valid Email address.")
            else:
                try:
                    selected_campaign = df_campaigns[df_campaigns['Campaign_ID'] == selected_campaign_id].iloc[0]
                    total_price = selected_campaign['Price'] * num_tickets

                    st.success('### Booking Successful! Confirmation:')
                    st.write(f"- **Name:** {name}")
                    st.write(f"- **Email:** {email}")
                    st.write(f"- **Campaign:** {selected_campaign['Campaign_Name']} (ID: {selected_campaign['Campaign_ID']})")
                    st.write(f"- **Date:** {selected_campaign['Date']}")
                    st.write(f"- **Location:** {selected_campaign['Location']}")
                    st.write(f"- **Number of Tickets:** {num_tickets}")
                    st.write(f"- **Total Price:** ${total_price:,.2f}")
                    st.balloons()
                except IndexError:
                     st.error("Selected campaign details not found. Please try again.")
                except Exception as e:
                     st.error(f"An error occurred during booking: {e}")

    # --- Display details of the selected campaign dynamically ---
    st.header('Selected Campaign Details')
    details_campaign_id = st.selectbox('View details for Campaign ID:', df_campaigns['Campaign_ID'], key='details_select')
    if details_campaign_id:
        try:
            details_campaign = df_campaigns[df_campaigns['Campaign_ID'] == details_campaign_id].iloc[0]
            st.write(f"**Name:** {details_campaign['Campaign_Name']}")
            st.write(f"**Date:** {details_campaign['Date']}")
            st.write(f"**Location:** {details_campaign['Location']}")
            st.write(f"**Price per Ticket:** ${details_campaign['Price']:,.2f}")
        except IndexError:
            st.error("Details not found for the selected campaign ID.")


def billboard_content_management_page():
    st.title('Billboard Content Management (Visit-Based Demo)')
    st.info("Manages hypothetical billboard content based on location and duration, showing sample customer visit frequencies.")

    # --- Sample Data ---
    # Use st.session_state to make billboard content editable across reruns
    if 'billboard_data' not in st.session_state:
        st.session_state.billboard_data = pd.DataFrame({
            'Location': ['Times Square', 'Downtown LA', 'Millennium Park', 'Pike Place Market', 'Space Needle'],
            'Content_14_Days': ['Ad A: Fashion Sale', 'Ad B: New Movie Release', 'Ad C: Museum Exhibit', 'Ad D: Fresh Produce', 'Ad E: City Views'],
            'Content_30_Days': ['Ad F: Broadway Show', 'Ad G: Concert Tickets', 'Ad H: Park Events', 'Ad I: Seafood Special', 'Ad J: Tourist Info']
        })

    if 'visit_data' not in st.session_state:
         @st.cache_data # Cache the generated visit data
         def generate_visit_data():
            np.random.seed(42)
            num_visits = 200
            locations = st.session_state.billboard_data['Location'].tolist()
            customer_ids = np.random.randint(1, 51, num_visits)
            visit_locations = np.random.choice(locations, num_visits, p=[0.3, 0.2, 0.15, 0.2, 0.15]) # Uneven distribution
            # Generate dates within the last ~60 days
            today = datetime.today()
            visit_dates = [today - timedelta(days=np.random.randint(0, 60)) for _ in range(num_visits)]
            df = pd.DataFrame({
                'Customer_ID': customer_ids,
                'Location': visit_locations,
                'Visit_Date': visit_dates
            })
            df['Visit_Date'] = pd.to_datetime(df['Visit_Date'].dt.date) # Normalize to date only
            return df
         st.session_state.visit_data = generate_visit_data()

    df_visits = st.session_state.visit_data
    df_billboards = st.session_state.billboard_data


    st.sidebar.header("Update Billboard Content")
    location_filter = st.sidebar.selectbox("Select Location to Update:", df_billboards['Location'].unique(), key='bb_loc_filter')
    days_filter = st.sidebar.radio("Select Duration to Update:", [14, 30], key='bb_days_filter', format_func=lambda x: f"{x} Days")
    current_content_col = f'Content_{days_filter}_Days'
    current_content = df_billboards.loc[df_billboards['Location'] == location_filter, current_content_col].iloc[0]

    new_content = st.sidebar.text_area(f"New Content for {days_filter} Days at {location_filter}:", value=current_content, key='bb_new_content')

    if st.sidebar.button("Update Content", key='bb_update_btn'):
        # Update the DataFrame in session state
        st.session_state.billboard_data.loc[st.session_state.billboard_data['Location'] == location_filter, current_content_col] = new_content
        st.sidebar.success("Content Updated!")
        # No need to explicitly rerun, Streamlit handles it on widget interaction

    st.header("Current Billboard Content Schedule")
    st.dataframe(st.session_state.billboard_data) # Display the potentially updated data

    # --- Visit Frequency Analysis ---
    st.header("Customer Visit Frequency Analysis")
    try:
        today_date = pd.to_datetime(datetime.today().date())
        date_14_days_ago = today_date - timedelta(days=14)
        date_30_days_ago = today_date - timedelta(days=30)

        # Filter visits within the last 14 and 30 days
        visits_14_days = df_visits[df_visits['Visit_Date'] >= date_14_days_ago]
        visits_30_days = df_visits[df_visits['Visit_Date'] >= date_30_days_ago]

        # Calculate frequencies
        visit_freq_14_days = visits_14_days['Location'].value_counts().reset_index()
        visit_freq_14_days.columns = ['Location', 'Visits (Last 14 Days)']

        visit_freq_30_days = visits_30_days['Location'].value_counts().reset_index()
        visit_freq_30_days.columns = ['Location', 'Visits (Last 30 Days)']

        # Merge frequencies with billboard data
        df_billboard_visits = pd.merge(st.session_state.billboard_data, visit_freq_14_days, on='Location', how='left')
        df_billboard_visits = pd.merge(df_billboard_visits, visit_freq_30_days, on='Location', how='left')
        # Fill NaN visit counts with 0
        df_billboard_visits[['Visits (Last 14 Days)', 'Visits (Last 30 Days)']] = df_billboard_visits[['Visits (Last 14 Days)', 'Visits (Last 30 Days)']].fillna(0).astype(int)


        st.subheader("Billboard Content with Recent Visit Counts")
        st.dataframe(df_billboard_visits)

        # --- Insights ---
        st.subheader("Insights for Content Management")
        st.write("""
        - **High Traffic Locations:** Identify locations with high visit counts (e.g., > threshold in last 14/30 days). Consider prioritizing premium or timely content for these billboards.
        - **Content Relevance:** Does the current content align with the likely demographics/interests of visitors to high-traffic locations? (Requires more data than available here).
        - **Duration Strategy:** Locations with consistently high traffic might benefit from more frequent content updates (e.g., using the 14-day slot actively), while lower traffic areas might suffice with 30-day content.
        - **Analyze Trends:** Compare 14-day vs 30-day counts. A high 14-day count relative to 30-day might indicate a recent event or surge in interest.
        """)

        # Simple example insight:
        highest_traffic_14 = df_billboard_visits.loc[df_billboard_visits['Visits (Last 14 Days)'].idxmax()]
        st.info(f"**Insight Example:** '{highest_traffic_14['Location']}' had the highest visits ({highest_traffic_14['Visits (Last 14 Days)']}) in the last 14 days. Ensure the content '{highest_traffic_14['Content_14_Days']}' is compelling for current visitors.")

    except Exception as e:
        st.error(f"An error occurred during visit analysis: {e}")


def dynamic_billboard_content_page():
    st.title('Dynamic Billboard Content Simulation')
    st.info("Simulates a billboard changing content based on 'customer presence' and selection. Requires local video files - paths need adjustment.")
    st.warning(f"""
    This page attempts to load videos from:
    1. Ice Cream: `{ICE_CREAM_VIDEO_PATH}`
    2. Food: `{FOOD_VIDEO_PATH}`
    Please ensure these files exist at the specified paths or update the paths in the script.
    Alternatively, replace with web-accessible video URLs.
    """)

    st.write("""
    ### Billboard Display Area
    """)

    # Simulate customer looking
    customer_looking = st.checkbox("Simulate: Customer is looking at the billboard", key='dybb_looking')

    if customer_looking:
        st.success("Customer detected! Displaying dynamic content.")
        content_type = st.radio("Select content type to display:", ["Ice Cream Ad", "Food Ad"], key='dybb_content_type', horizontal=True)

        video_path = None
        if content_type == "Ice Cream Ad":
            video_path = ICE_CREAM_VIDEO_PATH
        elif content_type == "Food Ad":
            video_path = FOOD_VIDEO_PATH

        if video_path:
            try:
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()
            except FileNotFoundError:
                st.error(f"Video file not found at: {video_path}. Please check the path.")
            except Exception as e:
                st.error(f"Could not load video: {e}")
        else:
             st.warning("No video path defined for the selected content type.")

    else:
        st.info("No customer detected. Displaying default content (placeholder).")
        # Optionally display a default static image or message
        st.image("https://via.placeholder.com/800x450.png?text=Default+Billboard+Content", use_column_width=True)


def innovative_aspects_page():
    st.title("Innovative Aspects of Datacost Fusion")
    st.info("This page lists potential innovative aspects of a conceptual Datacost Fusion system.")

    st.write("""
    Datacost Fusion, as a concept integrating data analysis with cost management and potentially advertising or operational data,
    can be innovative in several ways by leveraging modern technologies and approaches:
    """)

    aspects = [
        "**Real-Time Data Fusion & Costing:** Ingesting data from diverse sources (IoT, sales, ads, supply chain) in real-time and immediately calculating associated costs or ROI.",
        "**AI-Powered Predictive Costing:** Using machine learning to predict future costs based on anticipated operational changes, market conditions, or planned ad campaigns.",
        "**Dynamic Resource & Budget Allocation:** Automatically adjusting advertising budgets, production schedules, or resource allocation based on fused data insights and predictive cost/ROI models to maximize efficiency.",
        "**Cross-Domain Anomaly Detection:** Identifying unusual patterns or cost overruns by correlating data across previously siloed domains (e.g., linking a spike in ad spend with an unexpected dip in supply chain efficiency).",
        "**Prescriptive Analytics for Cost Reduction:** Not just predicting costs, but recommending specific actions (e.g., change supplier, adjust ad targeting, modify product feature) to optimize cost-effectiveness based on fused data.",
        "**Privacy-Preserving Fusion for Collaboration:** Enabling analysis across datasets from different departments or even partner companies without exposing raw sensitive data, using techniques like federated learning or differential privacy.",
        "**Blockchain for Transparent Cost Tracking:** Using blockchain to create an immutable and transparent ledger for tracking costs associated with specific campaigns, products, or processes across multiple stakeholders.",
        "**Edge Computing Fusion for Immediacy:** Performing initial data fusion and cost analysis directly on edge devices (like sensors in a factory or PoS systems) for faster local decision-making before sending aggregated data to the cloud.",
        "**Human-in-the-Loop AI for Complex Decisions:** Combining AI-driven cost analysis with human expertise through interactive dashboards, allowing users to explore scenarios, override suggestions, and validate complex cost-saving strategies.",
        "**Ethical & Explainable AI in Costing:** Ensuring that AI models used for cost prediction and resource allocation are fair, transparent, and explainable, avoiding biases that could unfairly impact certain products, campaigns, or suppliers."
    ]

    st.write("### Potential Innovative Aspects:")
    for i, aspect in enumerate(aspects):
        st.markdown(f"{i+1}. {aspect}")


# --- Main Application Logic ---

# Dictionary mapping page names to their functions
PAGES = {
    "🏠 Home": home_page,
    " campaings 🧮 Retention Calculator": retention_calculator_page,
    " Fusion & Clustering": data_fusion_page,
    " Budget Pie Chart": budget_pie_chart_page,
    "⚙️ Advertising Optimization": advertising_optimization_page,
    "💲 Price Optimization (Demand)": price_optimization_page,
    " Budget Allocation Tool": budget_allocation_tool_page,
    " Customer Segmentation": customer_segmentation_page,
    "📊 Product Quality Analysis": product_quality_analysis_page,
    " Immersive Analytics": immersive_analytics_page,
    "📈 5-Year Prediction Chart": prediction_chart_page,
    " Campaign Creation Filter": campaign_creation_page,
    " Campaign Forecasting Budget": campaign_forecasting_budget_page,
    "🏭 Sales & Production Suggestion": sales_analysis_production_page,
    "🛍️ Product Differentiation": product_differentiation_page,
    " Budget Estimation (Occasion)": ad_budget_estimation_page,
    "💰 Cost Analysis (Types)": cost_analysis_page,
    " Cost Optimization Algos": cost_optimization_algorithms_page,
    " Nested Model (Clicks/Spend)": nested_model_page,
    " Proportion Actual/Predicted": proportion_actual_predicted_page,
    "📉 Actual vs Predicted Line": actual_predicted_line_page,
    "📊 Cumulative Actual/Predicted": cumulative_actual_predicted_page,
    "💹 SMA Crossover Strategy": sma_crossover_strategy_page,
    " Campaign Analysis Pie": campaign_analysis_pie_page,
    " Sales KPI Calculator": sales_kpi_calculator_page,
    " Attribution & ROI": attribution_roi_page,
    " Audience Segmentation": audience_segmentation_personalization_page,
    "🔗 Supply Chain & Ads Dashboard": supply_chain_ad_dashboard_page,
    " Demand Forecasting": demand_forecasting_page,
    "📍 Top Cities by Category": top_cities_page,
    " Targeted Sales Ads": targeted_sales_ads_page,
    " Merged Product/Ad Data": merged_product_ad_page,
    " Personalized Ads (Advanced)": personalized_adv_segmentation_adv_page,
    " Revenue/Ad Dashboard": revenue_ad_analysis_page,
    " Sales/Revenue Dashboard (Alt)": sales_revenue_dashboard_page,
    " Dynamic Chart Visualizer": dynamic_chart_visualization_page,
    "🎟️ Campaign Ticket Booking": campaign_ticket_booking_page,
    " Billboard Content Mgmt": billboard_content_management_page,
    " Dynamic Billboard Sim": dynamic_billboard_content_page,
    "💡 Innovative Aspects": innovative_aspects_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Call the function corresponding to the selection
page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("Datacost Fusion System - M.Tech Datascience Project|Developed Sangita Biswas")


