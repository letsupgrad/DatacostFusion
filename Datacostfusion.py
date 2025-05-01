import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io





def page1():
    st.title("Page 1")
    st.write("Content for Page 1 goes here.")



# Load company image

  

image = Image.open('C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Website-image-advertising-1024x640.jpg')


# Set page layout
st.set_page_config(layout="wide")

# Set title and background image
st.title("Welcome to Datacost Fusion System")
st.write("Welcome to the Datacost Fusion System! This platform enables seamless integration and analysis of diverse data sources to drive insights and decision-making.")
st.write("Please explore the various features and functionalities available.")
st.title('Campaign Customer Retention Calculator')
st.image(image, use_column_width=True)

# Sidebar
st.sidebar.header('Input Parameters')
campaign_cost = st.sidebar.number_input('Campaign Cost ($)', min_value=0.0, value=10000.0)
initial_customers = st.sidebar.number_input('Initial Number of Customers', min_value=0, value=1000)
new_customers = st.sidebar.number_input('New Customers Acquired', min_value=0, value=500)
retained_customers = st.sidebar.number_input('Retained Customers from Previous Campaign', min_value=0, value=800)

# Calculate retention rate and retained customers
total_customers = initial_customers + new_customers
retention_rate = retained_customers / total_customers

# Calculate revenue from retained customers
revenue_per_customer = 100  # Placeholder value
revenue_from_retained_customers = retained_customers * revenue_per_customer

# Calculate ROI
roi = (revenue_from_retained_customers - campaign_cost) / campaign_cost * 100

# Display results
st.header('Campaign Results')
col1, col2 = st.columns(2)
col1.metric("Retention Rate", f"{retention_rate:.2%}")
col2.metric("Revenue from Retained Customers", f"${revenue_from_retained_customers:,.2f}")
st.write(f"Return on Investment (ROI): {roi:.2f}%")



def page2():
    st.title("Page 2")
    st.write("Content for Page 2 goes here.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import numpy as np
import yfinance as yf
from PIL import Image





def cluster_and_fuse_data(df1, df2, df3, num_clusters):
    # Merge dataframes
    merged_df = pd.concat([df1, df2, df3], axis=1)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    merged_df_imputed = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df_imputed)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    merged_df_imputed['Cluster'] = kmeans.fit_predict(scaled_data)

    return merged_df_imputed

def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def main():
    st.title('Data Fusion')

    df1 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising Budget and Sales.csv")
    df2 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\advertising.csv")
    df3 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising_Data.csv")

    # Perform data fusion
    df_fused = pd.concat([df1, df2, df3])

    # Allow the user to select columns to display
    selected_columns = st.multiselect('Columns to display', df_fused.columns)

    # Display selected columns
    st.dataframe(df_fused[selected_columns])

    # Allow the user to download fused data
    st.download_button('Download fused data', df_fused.to_csv())

    # Get the number of clusters from the user
    num_clusters = st.slider("Choose the number of clusters", min_value=2, max_value=10, value=3)

    # Perform data fusion and clustering
    fused_data = cluster_and_fuse_data(df1, df2, df3, num_clusters)

    # Rename duplicate columns
    fused_data = rename_duplicates(fused_data)

    # Display the resulting DataFrame
    st.dataframe(fused_data)

    # Create a bar plot of cluster distribution
    cluster_counts = fused_data['Cluster'].value_counts()
    st.bar_chart(cluster_counts)



    # Create a heatmap
    st.subheader("Heatmap based on Fused Data")
    plt.figure(figsize=(10, 8))
    sns.heatmap(fused_data.corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title("Heatmap of Data")
    st.pyplot()

    # Check the available columns in the fused dataframe
    selected_column = st.selectbox("Select a column for the histogram", df_fused.columns)

    # Create a histogram using seaborn
    st.title("Histogram of {}".format(selected_column))
    plt.figure(figsize=(10, 6))
    sns.histplot(df_fused[selected_column], bins=30, kde=True)
    plt.title('Distribution of {}'.format(selected_column))
    plt.xlabel(selected_column)
    plt.ylabel('Frequency')
    st.pyplot()
    
    
        
if __name__ == "__main__":
    main()

# Load the dataset
dataset_path = "C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising Budget and Sales.csv"
df = pd.read_csv(dataset_path)

# Streamlit app
def main():
    st.title('Pie Chart for Advertising Budget and TV Budget Estimation')

    # Display the loaded dataset
    st.subheader('Loaded Dataset:')
    st.write(df)

    # Allow the user to select the budget type
    selected_budget = st.radio("Select the budget type:", ['Radio Ad Budget ($)', 'TV Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)'])

    # Filter the data based on the selected budget type
    selected_column = df[selected_budget]

    # Calculate the total budget for the selected budget type
    total_budget = selected_column.sum()

    # Calculate the percentage of each budget type
    pie_data = [total_budget, df['TV Ad Budget ($)'].sum() - total_budget]

    # Plotting the pie chart
    labels = ['Selected Budget Type', 'Other Budget Types']
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart in Streamlit
    st.subheader('Pie Chart for Budget Distribution')
    st.pyplot(fig1)

    # Show the sales on budget estimation
    if selected_budget == 'Sales ($)':
        st.subheader('Sales on Budget Estimation')
        st.write(f"Total Sales: {df['Sales ($)'].sum()}")
        st.write(f"Total Budget: {total_budget}")
        st.write(f"Sales on Budget Estimation: {total_budget / df['Sales ($)'].sum() * 100:.2f}%")

    # Estimate the budget for TV Ad Budget ($)
    if selected_budget == 'TV Ad Budget ($)':
        st.subheader('TV Ad Budget Estimation')

        # Prepare the data for linear regression
        X = df[['Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)']]
        y = df['TV Ad Budget ($)']

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the TV Ad Budget ($)
        tv_budget_prediction = model.predict([[100000, 50000, 3000000]])  # Example input values
        st.write(f"Estimated TV Ad Budget ($): {tv_budget_prediction[0]:.2f}")

       
if __name__ == "__main__":
    main()


def page3():
    st.title("Page 3")
    st.write("Content for Page 3 goes here.")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Function to predict advertisement cost
def predict_advertisement_cost(df):
    # Assuming 'Sales' is the independent variable and 'Google_Ads' is the dependent variable
    X = df[['Sales']]
    y = df['Google_Ads']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform linear regression (or any other regression model)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict advertisement cost for the test set
    ad_cost_predictions = model.predict(X_test)

    return ad_cost_predictions

# Function to optimize advertisement cost
def optimize_advertisement_cost(df):
    # Implement your cost optimization logic here
    # This could involve adjusting advertising budgets based on performance metrics, etc.
    # For simplicity, let's assume a fixed cost reduction of 10%
    df['Optimized_Ad_Cost'] = df['Google_Ads'] * 0.9
    return df



# Streamlit app
def main():
    st.title('Data Fusion and Advertising Optimization')

    df1 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising Budget and Sales.csv")
    df2 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\advertising.csv")
    df3 = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising_Data.csv")

    # Perform data fusion and clustering
    num_clusters = st.slider("Choose the number of clusters", key="clusters", min_value=2, max_value=10, value=3)
    fused_data = cluster_and_fuse_data(df1, df2, df3, num_clusters)

    # Rename duplicate columns
    fused_data = rename_duplicates(fused_data)

    # Display the resulting DataFrame
    st.dataframe(fused_data)

    
    # Predict advertisement cost
    st.subheader("Predicted Advertisement Cost")
    ad_cost_predictions = predict_advertisement_cost(fused_data)
    st.write("Sample Predictions:", ad_cost_predictions[:5])

    # Optimize advertisement cost
    st.subheader("Optimized Advertisement Cost")
    optimized_data = optimize_advertisement_cost(fused_data)
    st.dataframe(optimized_data)

   


# Function to optimize data
def optimize_data(df):
    # Fill missing values with mean
    df.fillna(df.mean(), inplace=True)
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

# Streamlit app
def main():
    st.title("Advertising Budget and Sales Data Optimization")

    # Load data
    file_path = "C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising Budget and Sales.csv"
    df = pd.read_csv(file_path)

    # Show original data
    st.subheader("Original Data")
    st.write(df)

    # Randomly select 20% of the data
    sampled_df = df.sample(frac=0.2, random_state=42)

    # Show sampled data
    st.subheader("Sampled Data (20%)")
    st.write(sampled_df)

    # Optimize sampled data
    optimized_df = optimize_data(sampled_df)

    # Show optimized data
    st.subheader("Optimized Sampled Data (20%)")
    st.write(optimized_df)

# Function to predict advertisement cost
def predict_advertisement_cost(df):
    # Assume 'Sales' is the target variable
    target_variable = 'Sales'

    # Split data into features and target
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    return predictions

# Function to optimize advertisement cost
def optimize_advertisement_cost(df):
    # Implement your cost optimization logic here
    # This could involve adjusting advertising budgets based on performance metrics, etc.
    # For simplicity, let's assume a fixed cost reduction of 10%
    df['Optimized_Ad_Cost'] = df['Google_Ads'] * 0.9
    return df


if __name__ == "__main__":
    main()
import streamlit as st
from scipy.optimize import linprog

def linear_programming(budget, prices, revenues):
    # Objective function coefficients (negative since linprog minimizes)
    c = [-revenue for revenue in revenues]

    # Constraint matrix (budget constraint)
    A = [[price for price in prices]]

    # Constraint bounds (budget)
    b = [budget]

    # Solve linear programming problem
    result = linprog(c, A_ub=A, b_ub=b)

    return result

def mixed_integer_linear_programming(budget, prices, revenues):
    # Placeholder for MILP implementation
    st.write("Mixed-Integer Linear Programming (MILP) implementation is not available.")

def simulated_annealing(budget, prices, revenues):
    # Placeholder for simulated annealing implementation
    st.write("Simulated Annealing implementation is not available.")

def main():
    st.title("Optimization Algorithms")

    algorithm = st.selectbox("Select Optimization Algorithm", 
                             ["Linear Programming (LP)", 
                              "Mixed-Integer Linear Programming (MILP)", 
                              "Simulated Annealing"])

    budget = st.number_input("Enter budget:", min_value=0.0, step=0.01, format="%.2f")

    num_products = st.number_input("Enter the number of products:", min_value=1, step=1, format="%d")

    st.write("Enter the prices and revenues for each product:")

    prices = []
    revenues = []

    for i in range(num_products):
        price = st.number_input(f"Price of Product {i+1}:", min_value=0.0, step=0.01, format="%.2f")
        prices.append(price)
        
        revenue = st.number_input(f"Revenue of Product {i+1}:", min_value=0.0, step=0.01, format="%.2f")
        revenues.append(revenue)

    if st.button("Optimize"):
        if algorithm == "Linear Programming (LP)":
            result = linear_programming(budget, prices, revenues)
            st.write("### Optimization Result")
            st.write(f"Total Revenue: ${-result.fun:.2f}")
            st.write("Allocation:")
            for i, x in enumerate(result.x):
                st.write(f"Product {i+1}: {x:.2f}")
        elif algorithm == "Mixed-Integer Linear Programming (MILP)":
            mixed_integer_linear_programming(budget, prices, revenues)
        elif algorithm == "Simulated Annealing":
            simulated_annealing(budget, prices, revenues)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    st.title("Price Optimization")

    # Get user input for product data
    num_products = st.number_input("Enter the number of products:", min_value=1, value=1, step=1)

    product_data = []
    for i in range(num_products):
        st.write(f"### Product {i + 1}")
        price = st.number_input(f"Enter the price for Product {i + 1}:", min_value=0.01, value=10.00, step=0.01)
        demand = st.number_input(f"Enter the demand level for Product {i + 1}:", min_value=0, value=100, step=1)
        product_data.append({'Price': price, 'Demand': demand})

    df = pd.DataFrame(product_data)

    # Train linear regression model
    X = df[['Price']]
    y = df['Demand']
    model = LinearRegression()
    model.fit(X, y)

    # Predict optimal price
    optimal_price = model.predict([[df['Price'].mean()]])[0]

    st.write(f"Optimal Price for maximizing demand across products: ${optimal_price:.2f}")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd

# Sample data for demonstration
customer_data = {
    'Customer': ['Customer A', 'Customer B', 'Customer C'],
    'Segment': ['Segment 1', 'Segment 2', 'Segment 1'],
    'Budget_Allocation': [5000, 7000, 6000]
}

product_data = {
    'Product': ['Pen', 'Pencil', 'Notebook'],
    'Unit_Cost': [1, 0.5, 2]
}

# Create DataFrames
customer_df = pd.DataFrame(customer_data)
product_df = pd.DataFrame(product_data)

def budget_allocation(customer_df, product_df):
    st.write("## Budget Allocation Tool")

    selected_customer = st.selectbox("Select Customer", customer_df['Customer'].unique())
    selected_segment = customer_df[customer_df['Customer'] == selected_customer]['Segment'].values[0]

    st.write(f"### Customer: {selected_customer}, Segment: {selected_segment}")

    # Display product-wise budget allocation for the selected customer
    st.write("### Product-wise Budget Allocation:")
    st.write(customer_df[customer_df['Customer'] == selected_customer])

    # Calculate total budget allocated for the selected customer
    total_budget = customer_df[customer_df['Customer'] == selected_customer]['Budget_Allocation'].values[0]
    st.write(f"### Total Budget Allocated: ${total_budget}")

    # Display unit cost and allow user to input quantity for each product
    st.write("### Product-wise Quantity:")
    quantities = {}
    for index, row in product_df.iterrows():
        quantity = st.number_input(f"Enter Quantity for {row['Product']}", min_value=0, step=1)
        quantities[row['Product']] = quantity

    # Calculate total cost based on the quantities entered
    total_cost = sum(quantities.get(product, 0) * product_df.loc[product_df['Product'] == product, 'Unit_Cost'].iloc[0] for product in quantities)
    st.write(f"### Total Cost: ${total_cost}")

    # Check if total cost exceeds allocated budget
    if total_cost > total_budget:
        st.error("Total cost exceeds allocated budget!")
    else:
        st.success("Total cost is within allocated budget.")

def main():
    budget_allocation(customer_df, product_df)

if __name__ == "__main__":
    main()




import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sample customer data for demonstration
customer_data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Age': [35, 45, 30, 50, 40],
    'Income': [50000, 60000, 40000, 70000, 55000],
    'Category': ['Electronics', 'Books', 'Electronics', 'Fashion', 'Advertisement'],
    'Purchase_Amount': [100, 120, 90, 130, 110]
}

df = pd.DataFrame(customer_data)

# Encoding categorical variable
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

def main():
    st.title("Customer Segmentation with Advertisement")

    st.write("### Customer Data")
    st.write(df)

    # Select features and target variable
    X = df[['Age', 'Income', 'Category']]
    y = df['Purchase_Amount']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    score = model.score(X_test, y_test)
    st.write(f"Model Score: {score}")

    # Display customer segmentation form
    st.write("### Customer Segmentation")
    age = st.number_input("Enter Age:")
    income = st.number_input("Enter Income:")
    category = st.selectbox("Select Category", ['Electronics', 'Books', 'Fashion', 'Advertisement'])

    # Convert category to encoded value
    category_encoded = label_encoder.transform([category])[0]

    # Make prediction for customer segment
    predicted_purchase_amount = model.predict([[age, income, category_encoded]])
    st.write(f"Predicted Purchase Amount: {predicted_purchase_amount[0]}")

if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Simulate a dataset
def create_dataset():
    data = {
        'ProductID': range(1, 101),
        'Rating': np.random.randint(1, 6, size=100),
        'Review': np.random.choice(['good', 'excellent', 'not bad', 'poor', 'terrible'], 100),
        'Sales': np.random.randint(100, 1000, size=100)
    }
    df = pd.DataFrame(data)
    df['Review'] = df['Review'].replace({'good': 2, 'excellent': 5, 'not bad': 3, 'poor': 1, 'terrible': 0})
    return df

df = create_dataset()

# Streamlit UI
st.title('Product Quality Analysis Dashboard')
st.sidebar.header('Dataset')
if st.sidebar.button('Generate New Data'):
    df = create_dataset()
    st.sidebar.success('Data Regenerated!')

# Display Data
if st.checkbox('Show Data'):
    st.write(df)

# Split data
X = df[['Rating', 'Review', 'Sales']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
def train_model(X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"### Model Evaluation")
st.write(f"**Root Mean Square Error (RMSE):** {rmse:.2f}")

# Feature Importance
if hasattr(model.named_steps['regressor'], 'feature_importances_'):
    st.write("### Feature Importances")
    importances = model.named_steps['regressor'].feature_importances_
    features = X.columns
    df_importances = pd.DataFrame({'Features': features, 'Importance': importances})
    st.bar_chart(df_importances.set_index('Features'))

# Instructions for Improvement
st.write("""
## Insights for Product Improvement
- Review the feature importances to determine what factors most influence product ratings.
- Use this information to focus on high-impact areas for product development.
- Regularly update the model with new data to keep the insights current.
""")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Sample data generation
def generate_ad_data():
    np.random.seed(42)
    ad_data = pd.DataFrame({
        'Ad_ID': range(1, 101),
        'Clicks': np.random.randint(100, 1000, 100),
        'Views': np.random.randint(1000, 10000, 100),
        'CTR': np.random.uniform(0.05, 0.15, 100),
        'Revenue': np.random.uniform(1000, 10000, 100)
    })
    return ad_data

def immersive_analytics():
    st.title("Immersive Analytics with Advertisement Data")

    # Generate sample ad data
    ad_data = generate_ad_data()

    # Display sample ad data
    st.write("### Sample Advertisement Data:")
    st.write(ad_data.head())

    # Plot interactive 3D visualization
    fig = px.scatter_3d(ad_data, x='Clicks', y='Views', z='Revenue', color='CTR', title='Advertisement Performance')
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)

def interactive_simulations():
    st.title("Interactive Simulations with Advertisement Data")

    # Generate sample ad data
    ad_data = generate_ad_data()

    # Interactive component: CTR threshold slider
    ctr_threshold = st.slider("Select CTR Threshold", min_value=0.0, max_value=0.2, value=0.1, step=0.01)

    # Filter advertisements based on CTR threshold
    filtered_ads = ad_data[ad_data['CTR'] > ctr_threshold]

    # Display filtered advertisements
    st.write("### Advertisements with CTR above Threshold:")
    st.write(filtered_ads)

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Select a page", ["Immersive Analytics", "Interactive Simulations"])

    if page == "Immersive Analytics":
        immersive_analytics()
    elif page == "Interactive Simulations":
        interactive_simulations()

if __name__ == "__main__":
    main()

def page4():
    st.title("Page 4")
    st.write("Content for Page 4 goes here.")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# Function to generate sample predictions for the next 5 years
def generate_predictions(historical_data, start_date, brand):
    # Generate sample predictions for the next 5 years
    future_dates = pd.date_range(start=start_date, periods=365*5, freq='D')

    # Ensure brand column is numeric
    historical_data[brand] = pd.to_numeric(historical_data[brand], errors='coerce')

    # Generate predictions
    random_values = np.random.randn(len(future_dates)).cumsum()
    predicted_values = historical_data[brand].iloc[-1] + random_values

    predictions = pd.DataFrame({
        'date': future_dates,
        'brand': brand,
        'value': predicted_values
    })

    return predictions

def main():
    st.title('5-Year Prediction Multiline Chart')

    # Load your dataset (replace with the actual path)
    dataset_path = "C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising.campaigns.result.csv"
    df = pd.read_csv(dataset_path)

    # Convert 'date' column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])

    # Display the dataset
    st.subheader('Your Dataset')
    st.dataframe(df)

    # Input start date for predictions
    start_date = st.date_input('Select the start date for predictions', df['date'].max().date())

    # Generate predictions for each brand
    brands = df.columns[1:]  # assuming first column is 'date' and the rest are brands
    predictions_list = []

    for brand in brands:
        predictions = generate_predictions(df, start_date, brand)
        predictions_list.append(predictions)

    # Concatenate all predictions into a single DataFrame
    predictions_df = pd.concat(predictions_list, axis=0)

    # Melt historical data for Altair
    historical_df = df.melt(id_vars=['date'], value_vars=brands, var_name='brand', value_name='value')

    # Combine historical and predicted data
    combined_df = pd.concat([historical_df, predictions_df])

    # Multiline chart using Altair
    line_chart = alt.Chart(combined_df).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='brand:N',
        tooltip=['date:T', 'brand:N', 'value:Q']
    ).properties(
        width=800,
        height=400,
        title='Historical and Predicted Data for Multiple Brands'
    )

    st.altair_chart(line_chart, use_container_width=True)

if __name__ == "__main__":
    main()





import streamlit as st
import pandas as pd

# Load the dataset
dataset_path = "C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising.campaigns.result.csv"
df = pd.read_csv(dataset_path)

# Function to create a campaign
def create_campaign(df, new_users_threshold):  # Corrected the variable name
    # Assuming 'new_users' is the column representing the new users
    successful_campaign = df[df['new_users'] > new_users_threshold]

    return successful_campaign

# Streamlit app
def main():
    st.title('Campaign Creation Based on new_user')

    # Display the loaded dataset
    st.subheader('Loaded Dataset:')
    st.write(df)

    # Set a threshold for successful new users
    new_users_threshold = st.number_input("Enter the new_users threshold for a successful campaign:", min_value=0, step=100, value=500)

    # Create a campaign based on the new users threshold
    successful_campaign = create_campaign(df, new_users_threshold)  # Corrected the variable name

    # Display the successful campaign
    st.subheader('Successful Campaign:')
    st.write(successful_campaign)

if __name__ == "__main__":
    main()


import streamlit as st
import datetime
import numpy as np

# Function to calculate the forecasted budget for each day
def forecast_budget(start_date, num_days=14, initial_users=100, new_user_per_ad=456):
    campaign_dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    total_budget = [initial_users * new_user_per_ad + np.random.normal(scale=1000) for _ in range(num_days)]
    return campaign_dates, total_budget

# Streamlit app
def main():
    st.title('Campaign Forecasting Budget')

    # Input: Start date of the campaign
    start_date = st.date_input('Select the start date of the campaign', datetime.date(2024, 3, 5))

    # Calculate end date
    end_date = start_date + datetime.timedelta(days=14)

    # Forecast budget for each day
    campaign_dates, total_budget = forecast_budget(start_date)

    # Display campaign details
    st.subheader('Campaign Dates:')
    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {end_date}")

    # Display forecasted budget for each day
    st.subheader('Forecasted Budget for Each Day:')
    for i in range(len(campaign_dates)):
        st.write(f"Date: {campaign_dates[i]}, Forecasted Budget: ${total_budget[i]:.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

# Load the dataset
dataset_path = "C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising.campaigns.result.csv"
df = pd.read_csv(dataset_path)

# Function to suggest increasing production
def suggest_increase_production(df, low_sales_threshold, high_sales_threshold):
    # Filter the dataset based on Sales threshold
    low_sales_data = df[df['Sales'] < low_sales_threshold]
    high_sales_data = df[df['Sales'] > high_sales_threshold]

    # Calculate the percentage of low and high sales compared to total sales
    low_sales_percentage = (len(low_sales_data) / len(df)) * 100
    high_sales_percentage = (len(high_sales_data) / len(df)) * 100

    # Suggest to increase production based on sales thresholds
    if high_sales_percentage > 20:
        return "The project sales are excellent ({:.2f}%), consider increasing production to satisfy customer demand and ensure the successful running of the campaign.".format(high_sales_percentage)
    elif low_sales_percentage > 20:
        return "The project sales are low ({:.2f}%). Consider strategies to increase sales.".format(low_sales_percentage)
    else:
        return "The project sales are within expected levels."

# Streamlit app
def main():
    st.title('Sales Analysis and Production Suggestion')

    # Set thresholds for low and high sales
    low_sales_threshold = st.number_input("Enter the threshold for low sales:", min_value=0, step=100, value=10000000)
    high_sales_threshold = st.number_input("Enter the threshold for high sales:", min_value=0, step=100, value=20007909)

    # Suggest increasing production based on sales thresholds
    suggestion = suggest_increase_production(df, low_sales_threshold, high_sales_threshold)

    # Display the suggestion
    st.subheader('Production Suggestion:')
    st.write(suggestion)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
products = ['Product_A', 'Product_B', 'Product_C']
data = {
    'date': np.random.choice(dates, 1000),
    'product': np.random.choice(products, 1000),
    'customer_id': np.random.randint(1, 300, 1000)
}
df = pd.DataFrame(data)

# Title of the app
st.title('Product Differentiation Based on Customer Purchases')

# Display the dataset
st.subheader('Sample Dataset')
st.write(df)

# Select a date to filter the dataset
selected_date = st.date_input('Select a date to view purchases', value=pd.to_datetime('2023-01-01'))

# Filter the dataset based on the selected date
filtered_df = df[df['date'] == pd.to_datetime(selected_date)]

if not filtered_df.empty:
    # Display the filtered dataset
    st.subheader(f'Purchases on {selected_date}')
    st.write(filtered_df)

    # Count the number of purchases for each product
    product_counts = filtered_df['product'].value_counts().reset_index()
    product_counts.columns = ['product', 'count']

    # Display the product counts
    st.subheader('Number of Purchases per Product')
    st.write(product_counts)

    # Create a bar chart using Altair
    bar_chart = alt.Chart(product_counts).mark_bar().encode(
        x='product',
        y='count',
        color='product'
    ).properties(
        title=f'Number of Purchases per Product on {selected_date}',
        width=800,
        height=400
    )

    # Display the bar chart
    st.altair_chart(bar_chart, use_container_width=True)

    # Create a pie chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(product_counts['count'], labels=product_counts['product'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Proportion of Purchases per Product on {selected_date}')

    # Display the pie chart
    st.pyplot(fig)

else:
    st.write(f'No purchases found for {selected_date}')
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Sample sales and advertising data
np.random.seed(42)
data = {
    'occasion': ['New Year', 'Valentine\'s Day', 'Easter', 'Independence Day', 'Halloween', 'Thanksgiving', 'Christmas'],
    'sales': np.random.randint(5000, 20000, 7),
    'ad_budget': np.random.randint(1000, 5000, 7)
}

df = pd.DataFrame(data)

# Calculate the highest sales and corresponding advertising budget
max_sales_occasion = df.loc[df['sales'].idxmax()]
highest_sales = max_sales_occasion['sales']
highest_sales_occasion = max_sales_occasion['occasion']
estimated_budget = max_sales_occasion['ad_budget']

# Streamlit UI
st.title('Advertising Budget Estimation Based on Highest Sales Occasion')

# Display the dataset
st.subheader('Sales and Advertising Data')
st.write(df)

# Display the highest sales occasion and corresponding advertising budget
st.subheader('Occasion with Highest Sales')
st.write(f"**Occasion:** {highest_sales_occasion}")
st.write(f"**Sales:** {highest_sales}")
st.write(f"**Estimated Advertising Budget:** {estimated_budget}")

# Visualization
st.subheader('Sales and Advertising Budget by Occasion')

# Create bar charts using Altair
base = alt.Chart(df).encode(x='occasion')

sales_bar = base.mark_bar(color='blue').encode(y='sales').properties(title='Sales by Occasion')
budget_bar = base.mark_bar(color='orange').encode(y='ad_budget').properties(title='Advertising Budget by Occasion')

# Display the bar charts side by side
st.altair_chart(alt.hconcat(sales_bar, budget_bar), use_container_width=True)

# Additional input to adjust budget based on a multiplier
st.subheader('Adjust Advertising Budget')
multiplier = st.slider('Select a multiplier to adjust the advertising budget', min_value=0.5, max_value=2.0, value=1.0, step=0.1)
adjusted_budget = estimated_budget * multiplier
st.write(f"**Adjusted Advertising Budget for {highest_sales_occasion}:** {adjusted_budget}")

# Allow users to select an occasion and input a custom sales value to estimate the budget
st.subheader('Estimate Budget for a Custom Sales Value')
selected_occasion = st.selectbox('Select an occasion', df['occasion'])
custom_sales = st.number_input('Enter custom sales value', min_value=0, value=int(highest_sales))

# Find the corresponding ad budget and calculate the ratio
selected_occasion_data = df[df['occasion'] == selected_occasion].iloc[0]
ad_budget_ratio = selected_occasion_data['ad_budget'] / selected_occasion_data['sales']
estimated_custom_budget = custom_sales * ad_budget_ratio

st.write(f"**Estimated Advertising Budget for {custom_sales} sales on {selected_occasion}:** {estimated_custom_budget}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data for different products
data = {
    'product': ['Product A', 'Product B', 'Product C', 'Product D'],
    'current_cost': np.random.randint(100, 500, 4),
    'incremental_cost': np.random.randint(20, 50, 4),
    'opportunity_cost': np.random.randint(10, 30, 4),
    'sunk_cost': np.random.randint(50, 100, 4)
}

df = pd.DataFrame(data)
df['differential_cost'] = df['current_cost'] + df['incremental_cost'] - df['opportunity_cost']
df['relevant_cost'] = df['incremental_cost'] + df['opportunity_cost']  # Example relevant cost calculation

# Streamlit UI
st.title('Cost Analysis for Decision-Making')

# Display the dataset
st.subheader('Product Cost Data')
st.write(df)

# Allow users to select a product
st.subheader('Select a Product')
selected_product = st.selectbox('Select a product to analyze', df['product'])

# Retrieve the cost data for the selected product
product_data = df[df['product'] == selected_product].iloc[0]
current_cost = product_data['current_cost']
incremental_cost = product_data['incremental_cost']
opportunity_cost = product_data['opportunity_cost']
sunk_cost = product_data['sunk_cost']
differential_cost = product_data['differential_cost']
relevant_cost = product_data['relevant_cost']

# Display the cost data
st.subheader(f'Cost Data for {selected_product}')
st.write(f"**Current Cost:** {current_cost}")
st.write(f"**Incremental Cost:** {incremental_cost}")
st.write(f"**Opportunity Cost:** {opportunity_cost}")
st.write(f"**Sunk Cost:** {sunk_cost}")
st.write(f"**Relevant Cost:** {relevant_cost}")
st.write(f"**Differential Cost:** {differential_cost}")

# Visualization
st.subheader('Cost Breakdown Visualization')

# Create a DataFrame for the selected product's cost breakdown
cost_breakdown = pd.DataFrame({
    'Cost Type': ['Current Cost', 'Incremental Cost', 'Opportunity Cost', 'Sunk Cost', 'Relevant Cost', 'Differential Cost'],
    'Amount': [current_cost, incremental_cost, opportunity_cost, sunk_cost, relevant_cost, differential_cost]
})

# Plot a pie chart for cost breakdown
fig, ax = plt.subplots()
ax.pie(cost_breakdown['Amount'], labels=cost_breakdown['Cost Type'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
st.pyplot(fig)

# Additional inputs for custom cost analysis
st.subheader('Custom Cost Analysis')

# User inputs for custom cost values
custom_current_cost = st.number_input('Enter custom current cost', min_value=0, value=int(current_cost))
custom_incremental_cost = st.number_input('Enter custom incremental cost', min_value=0, value=int(incremental_cost))
custom_opportunity_cost = st.number_input('Enter custom opportunity cost', min_value=0, value=int(opportunity_cost))
custom_sunk_cost = st.number_input('Enter custom sunk cost', min_value=0, value=int(sunk_cost))

# Calculate differential and relevant cost based on custom inputs
custom_differential_cost = custom_current_cost + custom_incremental_cost - custom_opportunity_cost
custom_relevant_cost = custom_incremental_cost + custom_opportunity_cost  # Example relevant cost calculation

st.write(f"**Custom Differential Cost:** {custom_differential_cost}")
st.write(f"**Custom Relevant Cost:** {custom_relevant_cost}")
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Example historical performance data
data = {
    'Channel': ['SEO', 'PPC', 'Email', 'Social Media', 'Affiliate'],
    'Expected Return': [10000, 15000, 8000, 12000, 7000],
    'Cost': [2000, 3000, 1500, 2500, 1000]
}

df = pd.DataFrame(data)

# Streamlit app
st.title('Cost Optimization Algorithms')

# Budget Allocation Model
st.header('Budget Allocation Model')

# Objective function coefficients (negative because linprog performs minimization)
c = -np.array(df['Expected Return'])

# Inequality constraints matrix (A_ub * x <= b_ub)
total_budget = st.number_input('Enter total budget', min_value=0, value=10000)
A_ub = np.array([df['Cost'], -df['Expected Return']])
b_ub = [total_budget, -50000]  # Example: Total budget <= user-defined budget and Expected Return >= $50,000

# Bounds for each variable (0 <= x_i <= 1, i.e., percentage of total budget allocated to each channel)
bounds = [(0, 1) for _ in range(len(df))]

# Linear programming to maximize expected return within the budget
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Allocated budget percentages
allocated_percentages = result.x
df['Allocated Budget'] = allocated_percentages * total_budget  # Scale to user-defined total budget

st.subheader('Allocated Budget')
st.write(df)

# ROI Maximization
st.header('ROI Maximization')

# Example data for marketing activities
activities = {
    'Activity': ['Ad Campaign 1', 'Ad Campaign 2', 'Ad Campaign 3', 'Ad Campaign 4'],
    'Expected Return': [10000, 15000, 8000, 12000],
    'Cost': [2000, 3000, 1500, 2500]
}

activities_df = pd.DataFrame(activities)
activities_df['ROI'] = activities_df['Expected Return'] / activities_df['Cost']

# Sort activities by ROI in descending order
activities_df = activities_df.sort_values(by='ROI', ascending=False)

st.subheader('Prioritized Activities by ROI')
st.write(activities_df)

# Visualization
st.header('Visualization')

# Pie chart for budget allocation
st.subheader('Budget Allocation Pie Chart')
fig, ax = plt.subplots()
ax.pie(df['Allocated Budget'], labels=df['Channel'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)




import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
# Make sure your dataset has a column named 'pv_cum'
df = pd.read_csv("C:\\Users\\DELL\\M.TECH PROJECT WITH FILE\\Advertising.campaigns.result.csv")

# Streamlit app
st.title('Nested Model for Predicting Ad Clicks')

# Display the loaded dataset
st.subheader('Loaded Dataset:')
st.write(df)

# Preprocess the data
df['spend'] = pd.to_numeric(df['spend'], errors='coerce')
X = df['spend'][:-1].values.reshape(-1, 1)  # Previous day's pv_cum values
y = df['clicks'][1:].values  # Next day's pv_cum values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

# Predict the next day's pv_cum value based on the last available value in the dataset
last_day_value = df['clicks'].iloc[-1]
next_day_prediction = model.predict([[last_day_value]])
st.write(f'Predicted Next spend for clicks: {next_day_prediction[0]}')

# Visualize the predictions
st.subheader('Nested Model Predictions vs. Actual Values')
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='black', label='Actual values')
ax.plot(X_test, predictions, color='blue', linewidth=3, label='Predicted values')
ax.set_title('Nested Model')
ax.set_xlabel('Previous spend for clicks')
ax.set_ylabel('Next Day spend for clicks')
ax.legend()

# Display the plot in the Streamlit app
st.pyplot(fig)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace this with your actual test data and predictions
np.random.seed(42)
y_test = np.random.randint(50, 200, 14)  # Actual values for 14 days
predictions = np.random.randint(50, 200, 14)  # Predicted values for 14 days

# Streamlit app
st.title('Proportion of Actual vs. Predicted new_users for Each Day Over the Campaign')

# Slider to select number of days
num_days = st.slider('Select number of days for the campaign', min_value=1, max_value=14, value=7)

# Calculate the total sum of actual and predicted values for the entire campaign
total_actual = np.sum(y_test[:num_days])
total_predicted = np.sum(predictions[:num_days])

# Create data for the pie chart for each day
labels = ['Actual Values', 'Predicted Values']
colors = ['black', 'blue']

# Streamlit app
st.write(f'Displaying pie charts for {num_days} days')

# Plot the pie chart for each day
fig, axs = plt.subplots(1, num_days, figsize=(num_days * 4, 4))
fig.suptitle(f'Proportion of Actual vs. Predicted new_users for Each Day Over the {num_days}-Day Campaign')

for i in range(num_days):
    sizes_actual = y_test[i] / total_actual if total_actual != 0 else 0
    sizes_predicted = predictions[i] / total_predicted if total_predicted != 0 else 0

    ax = axs[i] if num_days > 1 else axs
    ax.pie([sizes_actual, sizes_predicted], labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Day {i + 1}')

# Display the plot in the Streamlit app
st.pyplot(fig)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming X_test and y_test are your original test data
# Assuming predictions is the array of predicted values for the 14-day campaign

# Generate some sample data for demonstration
np.random.seed(42)
X_test = np.arange(1, 15).reshape(-1, 1)  # 14 days of campaign
y_test = np.random.randint(100, 200, 14).reshape(-1, 1)
predictions = np.random.randint(100, 200, 14).reshape(-1, 1)

# Streamlit app
st.title('Actual vs. Predicted Page Views for New Users Over a Campaign')

# Allow the user to enter the number of days for the campaign duration
max_days = len(y_test)
reduced_days = st.number_input(f'Enter the number of days (max {max_days})', min_value=1, max_value=max_days, value=max_days)

# Reduce the data to the selected number of days
X_test_reduced = X_test[:reduced_days]
y_test_reduced = y_test[:reduced_days]
predictions_reduced = predictions[:reduced_days]

# Generate an array of indices for the line plot
days_reduced = np.arange(1, reduced_days + 1)

# Plot the actual and predicted values for the reduced campaign duration
fig, ax = plt.subplots()
ax.plot(days_reduced, y_test_reduced.flatten(), color='black', marker='o', label='Actual Values')
ax.plot(days_reduced, predictions_reduced.flatten(), color='blue', marker='o', label='Predicted Values')

ax.set_xlabel('Day of Campaign')
ax.set_ylabel('PV for new_user')
ax.set_title(f'Actual vs. Predicted Page Views for New Users Over a {reduced_days}-Day Campaign')
ax.legend()
ax.grid(True)

# Display the plot in the Streamlit app
st.pyplot(fig)


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_test and y_test are your test data
# Assuming predictions is the array of predicted values for the 14-day campaign

# Define the maximum number of days for the slider
max_days = 31  # You can adjust this value as needed

# Slider widget for selecting the number of days
selected_days = st.slider("Select the number of days to visualize", min_value=1, max_value=max_days, value=14)

# Calculate the cumulative sum of actual and predicted values for the selected number of days
cumulative_actual = np.cumsum(y_test[:selected_days].flatten())
cumulative_predicted = np.cumsum(predictions[:selected_days].flatten())

# Generate an array of indices for the bar positions
days = np.arange(1, selected_days + 1)  # Ensure that days matches the length of cumulative_actual and cumulative_predicted

# Bar width for better visualization
bar_width = 0.35

# Streamlit app
st.title(f'Cumulative Actual vs. Predicted Page Views for New Users Over the {selected_days}-Day Campaign')

# Visualize the cumulative actual and predicted values using a bar chart
fig, ax = plt.subplots()
ax.bar(days, cumulative_actual, bar_width, color='black', label='Cumulative Actual Values')
ax.bar(days + bar_width, cumulative_predicted, bar_width, color='blue', label='Cumulative Predicted Values')

ax.set_xlabel('Day of Campaign')
ax.set_ylabel('Cumulative PV for New User')
ax.set_title(f'Cumulative Actual vs. Predicted Page Views for New Users Over the {selected_days}-Day Campaign')
ax.legend()

# Display the plot in the Streamlit app
st.pyplot(fig)



# Function to fetch historical stock prices using Yahoo Finance API
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to implement a simple moving average (SMA) crossover strategy
def sma_crossover_strategy(data, short_window=50, long_window=200):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Streamlit app
def main():
    st.title("Simple Moving Average (SMA) Crossover Strategy")

    # User inputs
    ticker = st.text_input("Enter stock ticker symbol:", "AAPL")
    start_date = st.date_input("Select start date:", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("Select end date:", pd.to_datetime('2025-01-01'))

    # Fetch historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    st.subheader("Stock Price Data")
    st.write(stock_data)

    # Apply SMA crossover strategy
    strategy_signals = sma_crossover_strategy(stock_data)
    
    st.subheader("Strategy Signals")
    st.write(strategy_signals)
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data for demonstration
product_campaign_data = {
    'Product': ['Product A', 'Product B', 'Product C'],
    'Campaign_A': [30, 40, 20],
    'Campaign_B': [40, 30, 25],
    'Campaign_C': [20, 25, 30]
}

df = pd.DataFrame(product_campaign_data)
df.set_index('Product', inplace=True)

def main():
    st.title("Campaign Analysis")

    st.write("### Campaign Data")
    st.write(df)

    selected_product = st.selectbox("Select Product", df.index.tolist())
    selected_campaign = st.selectbox("Select Campaign", df.columns.tolist())
    num_days = st.number_input("Enter Number of Days", min_value=1, max_value=365, value=30)

    campaign_percentage = df.loc[selected_product, selected_campaign]

    st.write(f"Percentage of Campaign {selected_campaign} for {selected_product}: {campaign_percentage}%")

    # Calculate campaign amount for the given number of days
    campaign_amount = (campaign_percentage / 100) * num_days

    st.write(f"Campaign {selected_campaign} amount for {num_days} days: {campaign_amount}")

    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(df.loc[selected_product], labels=df.columns.tolist(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("### Campaign Distribution")
    st.pyplot(fig)

if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic sales data
def generate_sales_data(num_records):
    np.random.seed(0)  # for reproducibility

    # Generate synthetic data for sales records, units sold, revenue, advertising cost, and campaign days
    sales_records = np.arange(1, num_records + 1)
    campaign_days = np.random.choice([0, 1], size=num_records, p=[0.7, 0.3])  # 30% chance of being a campaign day
    units_sold = np.random.randint(50, 500, size=num_records)
    revenue = np.random.randint(1000, 10000, size=num_records)
    advertising_cost = np.random.randint(100, 1000, size=num_records)

    # Adjust sales values for campaign days
    units_sold = np.where(campaign_days == 1, units_sold * 1.5, units_sold)  # increase sales by 50% on campaign days
    revenue = np.where(campaign_days == 1, revenue * 1.5, revenue)           # increase revenue by 50% on campaign days

    # Create a DataFrame
    data = {
        'Sales Record': sales_records,
        'Units Sold': units_sold,
        'Revenue': revenue,
        'Advertising Cost': advertising_cost,
        'Campaign Day': campaign_days
    }

    return pd.DataFrame(data)

# Streamlit app
def main():
    st.title("Sales KPI Calculator")

    num_records = st.sidebar.number_input("Number of Sales Records", min_value=1, step=1, value=100)

    # Generate synthetic sales data
    sales_data = generate_sales_data(num_records)

    st.write("Generated Synthetic Sales Data:")
    st.write(sales_data)

    # Show areas of more sales on campaign days
    campaign_days_sales = sales_data.groupby('Campaign Day')['Units Sold'].mean()
    st.write("Average Units Sold on Campaign Days vs Non-Campaign Days:")
    st.write(campaign_days_sales)

    # Plot a pie chart to visualize the proportion of sales on campaign days vs non-campaign days
    fig, ax = plt.subplots()
    ax.pie(campaign_days_sales, labels=campaign_days_sales.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("Pie Chart: Proportion of Sales on Campaign Days vs Non-Campaign Days")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np

def attribution_modeling(df):
    # Placeholder function to calculate the attributed conversions
    df['Attributed_Conversions'] = df['Conversions'] * 0.5  # Placeholder attribution weight
    return df

def roi_analysis(df):
    # Placeholder function to calculate the ROI
    df['ROI'] = (df['Attributed_Conversions'] / df['Clicks']) * 100
    return df

def main():
    st.title("Attribution Modeling and ROI Analysis")

    # Sample data for demonstration
    data = {
        'Date': pd.date_range(start='2022-01-01', end='2022-01-31'),
        'Channel': np.random.choice(['Google Ads', 'Facebook Ads', 'Display Ads'], 31),
        'Clicks': np.random.randint(100, 1000, 31),
        'Conversions': np.random.randint(10, 100, 31)
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Perform attribution modeling
    df = attribution_modeling(df)

    # Perform ROI analysis
    df = roi_analysis(df)

    # Display the DataFrame with attributed conversions and ROI
    st.write("Attribution Modeling and ROI Analysis Results:")
    st.write(df)

if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd

# Sample audience data for demonstration
data = {
    'User_ID': [1, 2, 3, 4, 5],
    'Age': [25, 35, 45, 30, 40],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Interests': ['Technology', 'Fashion', 'Sports', 'Technology', 'Food'],
    'Online_Behavior': ['Frequent online shopper', 'Active social media user', 'Sports enthusiast', 'Tech enthusiast', 'Foodie']
}

# Create a DataFrame
df = pd.DataFrame(data)

def segment_audience(df):
    # Perform audience segmentation based on demographics, interests, and online behavior
    # Placeholder function for demonstration purposes
    segmented_df = df.copy()
    segmented_df['Segment'] = 'Segment A'  # Placeholder segment name
    
    return segmented_df

def personalize_ads(segmented_df):
    # Personalize ad content and targeting strategies for different audience segments
    # Placeholder function for demonstration purposes
    personalized_ads = {
        'Segment A': 'Personalized ad content for Segment A',
        'Segment B': 'Personalized ad content for Segment B'
    }
    
    return personalized_ads

def main():
    st.title("Audience Segmentation and Personalization")

    # Perform audience segmentation
    segmented_df = segment_audience(df)

    # Display segmented audience data
    st.write("Segmented Audience Data:")
    st.write(segmented_df)

    # Personalize ads for different audience segments
    personalized_ads = personalize_ads(segmented_df)

    # Display personalized ads for each segment
    st.write("Personalized Ads:")
    for segment, ad_content in personalized_ads.items():
        st.write(f"- {segment}: {ad_content}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np

# Sample data for demonstration
inventory_data = {
    'Product': ['Product A', 'Product B', 'Product C'],
    'Inventory_Level': np.random.randint(10, 100, 3)
}

production_data = {
    'Product': ['Product A', 'Product B', 'Product C'],
    'Production_Status': np.random.choice(['On Track', 'Delayed', 'Completed'], 3)
}

supplier_performance_data = {
    'Supplier': ['Supplier X', 'Supplier Y', 'Supplier Z'],
    'Performance_Rating': np.random.randint(1, 5, 3)
}

ad_campaign_data = {
    'Campaign': ['Campaign 1', 'Campaign 2', 'Campaign 3'],
    'Clicks': np.random.randint(1000, 5000, 3),
    'Conversions': np.random.randint(50, 200, 3),
    'ROI': np.random.uniform(0.5, 5.0, 3)
}

# Create DataFrames
inventory_df = pd.DataFrame(inventory_data)
production_df = pd.DataFrame(production_data)
supplier_performance_df = pd.DataFrame(supplier_performance_data)
ad_campaign_df = pd.DataFrame(ad_campaign_data)

def real_time_monitoring():
    st.write("### Real-time Monitoring of Supply Chain Operations")
    st.write("#### Inventory Levels")
    st.write(inventory_df)

    st.write("#### Production Status")
    st.write(production_df)

    st.write("#### Supplier Performance")
    st.write(supplier_performance_df)

    st.write("### Ad Campaign Information")
    st.write(ad_campaign_df)

def trigger_alerts():
    st.write("### Trigger Alerts")
    # Placeholder for triggering alerts based on specific conditions
    pass

def scenario_analysis():
    st.write("### Scenario Analysis and What-If Scenarios")
    # Placeholder for scenario analysis and what-if simulations
    pass

def continuous_improvement():
    st.write("### Continuous Improvement")
    # Placeholder for evaluating supply chain decisions and strategies
    pass

def main():
    st.title("Supply Chain Management and Ad Campaign Dashboard")

    option = st.sidebar.selectbox("Select Option", 
                                  ["Real-time Monitoring", 
                                   "Trigger Alerts", 
                                   "Scenario Analysis", 
                                   "Continuous Improvement"])

    if option == "Real-time Monitoring":
        real_time_monitoring()
    elif option == "Trigger Alerts":
        trigger_alerts()
    elif option == "Scenario Analysis":
        scenario_analysis()
    elif option == "Continuous Improvement":
        continuous_improvement()

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data for demonstration
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Clicks': [1000, 1200, 1300, 1400, 1250],
    'Sales': [800, 850, 900, 950, 1000],
    'Marketing_Expenditure': [5000, 6000, 5500, 5800, 5700],
    'Price': [10, 9.5, 9, 9.2, 9.3],
    'Competitor_Activity': [3, 2, 4, 3.5, 3],
    'Seasonality': [0.8, 0.9, 1.2, 1.1, 1.0]
}

df = pd.DataFrame(data)

def main():
    st.title("Advertisement Demand Forecasting Tool")

    st.write("### Advertisement Data")
    st.write(df)

    # Select features and target variable
    X = df[['Clicks', 'Marketing_Expenditure', 'Price', 'Competitor_Activity', 'Seasonality']]
    y = df['Sales']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared Score: {r2}")

    # Display prediction form
    st.write("### Demand Forecasting")
    clicks = st.number_input("Clicks:")
    marketing_expenditure = st.number_input("Marketing Expenditure:")
    price = st.number_input("Price:")
    competitor_activity = st.number_input("Competitor Activity:")
    seasonality = st.number_input("Seasonality:")

    # Make demand forecast for user input
    user_input = [[clicks, marketing_expenditure, price, competitor_activity, seasonality]]
    demand_forecast = model.predict(user_input)

    st.write(f"Predicted Demand: {demand_forecast[0]}")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd

# Sample data
data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
    'Electronics': [100, 120, 90, 80, 110, 95, 85, 75, 105, 115],
    'Clothing': [90, 110, 100, 95, 105, 80, 85, 70, 95, 100],
    'Home Appliances': [80, 100, 85, 90, 95, 75, 80, 65, 90, 95],
    'Sports Equipment': [110, 105, 95, 100, 115, 90, 100, 80, 110, 105],
    'Furniture': [95, 85, 80, 90, 100, 70, 75, 60, 85, 90]
}
df = pd.DataFrame(data)
df = df.set_index('City')

# Main function
def main():
    # Title
    st.title("Top 5 Cities by Product Category")

    # Product category selection
    category = st.selectbox("Select Product Category:", df.columns)

    # Get top 5 cities for selected category
    top_cities = df[category].nlargest(5)

    # Display top 5 cities
    st.write(f"Top 5 Cities for {category}:")
    st.write(top_cities)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd

# Sample data
data = {
    'Ad_ID': [101, 102, 103, 104, 105],
    'Ad_Name': ['Back to School Sale', 'Summer Tech Deals', 'Holiday Specials', 'New Year Discounts', 'Spring Clearance'],
    'Product_Name': ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch', 'Tablet'],
    'Campaign_Budget': [5000, 3000, 2000, 4000, 3500]
}
df = pd.DataFrame(data)

# Function to filter data based on selected occasion
def filter_ads_by_occasion(occasion):
    return df[df['Ad_Name'].str.contains(occasion, case=False)]

# Main function
def main():
    # Title
    st.title("Targeted Sales Products through Advertisements")

    # Occasion selection
    occasion = st.selectbox("Select Occasion:", df['Ad_Name'])

    # Filter data based on selected occasion
    filtered_ads = filter_ads_by_occasion(occasion)

    # Display filtered advertisements
    st.write(f"Advertisements for {occasion}:")
    st.write(filtered_ads)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd

# Sample product data
product_data = {
    'Product_ID': [1, 2, 3, 4, 5],
    'Product_Name': ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch', 'Tablet'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Wearable', 'Electronics'],
    'Price': [1000, 800, 200, 300, 500]
}

# Sample advertisement data
advertisement_data = {
    'Ad_ID': [101, 102, 103, 104, 105],
    'Ad_Name': ['Back to School Sale', 'Summer Tech Deals', 'Holiday Specials', 'New Year Discounts', 'Spring Clearance'],
    'Product_ID': [1, 2, 3, 4, 5],
    'Campaign_Budget': [5000, 3000, 2000, 4000, 3500]
}

# Convert dictionaries to DataFrames
product_df = pd.DataFrame(product_data)
advertisement_df = pd.DataFrame(advertisement_data)

# Merge product and advertisement data on 'Product_ID'
merged_df = pd.merge(product_df, advertisement_df, on='Product_ID')

# Streamlit code to display the merged data
def main():
    # Title of the app
    st.title("Merged Product and Advertisement Data")

    # Display the product data
    st.subheader("Product Data")
    st.write(product_df)

    # Display the advertisement data
    st.subheader("Advertisement Data")
    st.write(advertisement_df)

    # Display the merged data
    st.subheader("Merged Data")
    st.write(merged_df)

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

# Add a header
st.title("Personalized Advertising and Customer Segmentation")

# Create a sample customer data
customer_data = {
    'Customer ID': [1, 2, 3, 4, 5],
    'Age': [25, 35, 42, 28, 31],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Seattle', 'Houston'],
    'Purchase History': [['Electronics', 'Clothing'], ['Home Appliances'], ['Sports Equipment', 'Clothing'], ['Electronics', 'Books'], ['Furniture']],
    'Social Media Activity': [['Facebook', 'Twitter'], ['Instagram'], ['Facebook', 'LinkedIn'], ['Twitter', 'Instagram'], ['Facebook']]
}
df = pd.DataFrame(customer_data)

# Display the customer data
st.write("Customer Data:")
st.write(df)

# Create a sidebar for filtering and segmentation
st.sidebar.header("Customer Segmentation")
age_filter = st.sidebar.slider("Age Range", 18, 65, (20, 50), 1)
gender_filter = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
location_filter = st.sidebar.multiselect("Location", df['Location'].unique(), default=df['Location'].unique())

# Filter the data based on the selected criteria
filtered_data = df[
    (df['Age'].between(age_filter[0], age_filter[1])) &
    (df['Gender'].isin(gender_filter)) &
    (df['Location'].isin(location_filter))
]

# Display the filtered data
st.write("Filtered Customer Data:")
st.write(filtered_data)

# Analyze customer segments
segments = filtered_data.groupby(['Age', 'Gender', 'Location'])

# Display customer segments
st.write("Customer Segments:")
for segment, group in segments:
    st.write(f"Segment: {segment}")
    st.write(group)

# Personalized advertising recommendations
st.header("Personalized Advertising Recommendations")
for segment, group in segments:
    purchase_history = group['Purchase History'].tolist()
    social_media_activity = group['Social Media Activity'].tolist()
    st.write(f"Segment: {segment}")
    st.write("Targeted Advertising Channels:")
    for channel in set([item for sublist in social_media_activity for item in sublist]):
        st.write(f"- {channel}")
    st.write("Recommended Products/Services:")
    for product in set([item for sublist in purchase_history for item in sublist]):
        st.write(f"- {product}")
    st.write("---")

# Add a section for chart creation
st.header("Create Charts from Filtered Data")
chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Geo Chart", "Combo Chart"])
data_column = st.selectbox("Select Data to Visualize", ["Age", "Gender", "Location"])

if chart_type == "Bar Chart":
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=data_column,
        y='count()'
    ).properties(
        title=f'{chart_type} of {data_column}'
    )
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Pie Chart":
    pie_data = filtered_data[data_column].value_counts().reset_index()
    pie_data.columns = [data_column, 'count']
    chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field=data_column, type='nominal'),
        tooltip=[data_column, 'count']
    ).properties(
        title=f'{chart_type} of {data_column}'
    )
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Geo Chart":
    # Sample latitude and longitude data for locations
    location_data = {
        'Location': ['New York', 'Los Angeles', 'Chicago', 'Seattle', 'Houston'],
        'Latitude': [40.7128, 34.0522, 41.8781, 47.6062, 29.7604],
        'Longitude': [-74.0060, -118.2437, -87.6298, -122.3321, -95.3698]
    }
    location_df = pd.DataFrame(location_data)
    
    # Count the number of customers in each location
    location_counts = filtered_data['Location'].value_counts().reset_index()
    location_counts.columns = ['Location', 'Count']
    
    # Merge with the location data
    location_df = pd.merge(location_df, location_counts, on='Location', how='left').fillna(0)

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=37.7749,
            longitude=-122.4194,
            zoom=3,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=location_df,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius='Count * 1000',
            ),
            pdk.Layer(
                'CircleLayer',
                data=location_df,
                get_position='[Longitude, Latitude]',
                get_radius='Count * 1000',
                get_fill_color='[180, 0, 200, 140]',
                pickable=True
            ),
        ],
    ))

elif chart_type == "Combo Chart":
    combo_data = filtered_data.groupby(['Location']).size().reset_index(name='Count')
    bar_chart = alt.Chart(combo_data).mark_bar().encode(
        x='Location',
        y='Count'
    )
    line_chart = alt.Chart(combo_data).mark_line(color='red').encode(
        x='Location',
        y='Count'
    )
    st.altair_chart(bar_chart + line_chart, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
np.random.seed(42)
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Product_A_Sales': np.random.randint(100, 500, 12),
    'Product_B_Sales': np.random.randint(200, 600, 12),
    'Product_C_Sales': np.random.randint(300, 700, 12),
    'Product_A_Ads': np.random.randint(10, 50, 12),
    'Product_B_Ads': np.random.randint(20, 60, 12),
    'Product_C_Ads': np.random.randint(30, 70, 12)
}

df = pd.DataFrame(data)

# Calculate revenue based on sales (assuming revenue is sales multiplied by some constant factor, e.g., price)
price_a = 10
price_b = 15
price_c = 20

df['Product_A_Revenue'] = df['Product_A_Sales'] * price_a
df['Product_B_Revenue'] = df['Product_B_Sales'] * price_b
df['Product_C_Revenue'] = df['Product_C_Sales'] * price_c

# Streamlit UI
st.title('Revenue and Advertising Analysis')

st.write("""
## Visualization Dashboard
This dashboard visualizes the revenue generated from sales products based on advertising expenditure.
""")

# Select chart type
chart_type = st.selectbox("Select the type of chart to display", ["Multi-Line Chart", "Waterfall Chart", "Heatmap"])

# Filter options
selected_months = st.multiselect("Select months to display", df['Month'].unique(), default=df['Month'].unique())
filtered_df = df[df['Month'].isin(selected_months)]

if chart_type == "Multi-Line Chart":
    # Melt the DataFrame to long format for Altair
    df_long = filtered_df.melt(id_vars=['Month'], value_vars=['Product_A_Revenue', 'Product_B_Revenue', 'Product_C_Revenue', 
                                                              'Product_A_Ads', 'Product_B_Ads', 'Product_C_Ads'],
                               var_name='Category', value_name='Value')

    # Define the chart
    chart = alt.Chart(df_long).mark_line().encode(
        x='Month',
        y='Value',
        color='Category'
    ).properties(
        title='Revenue and Advertising Expenditure Over Months',
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Waterfall Chart":
    # Prepare data for waterfall chart
    waterfall_data = {
        'Type': ['Initial'] + list(filtered_df['Month']),
        'Product_A_Revenue': [0] + list(filtered_df['Product_A_Revenue']),
        'Product_B_Revenue': [0] + list(filtered_df['Product_B_Revenue']),
        'Product_C_Revenue': [0] + list(filtered_df['Product_C_Revenue'])
    }

    df_waterfall = pd.DataFrame(waterfall_data)
    df_waterfall['Total_Revenue'] = df_waterfall['Product_A_Revenue'] + df_waterfall['Product_B_Revenue'] + df_waterfall['Product_C_Revenue']

    # Calculate the cumulative sum for the waterfall chart
    df_waterfall['Cumulative_Revenue'] = df_waterfall['Total_Revenue'].cumsum()

    # Create waterfall chart using Plotly
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative"] * (len(df_waterfall) - 1) + ["total"],
        x=df_waterfall['Type'],
        textposition="outside",
        y=df_waterfall['Cumulative_Revenue'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="Cumulative Revenue Waterfall Chart",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Heatmap":
    heatmap_data = filtered_df.melt(id_vars=['Month'], value_vars=['Product_A_Sales', 'Product_B_Sales', 'Product_C_Sales'],
                                    var_name='Product', value_name='Sales')
    heatmap = px.density_heatmap(heatmap_data, x='Month', y='Product', z='Sales', color_continuous_scale='Viridis')
    heatmap.update_layout(title='Sales Heatmap', xaxis_title='Month', yaxis_title='Product')

    st.plotly_chart(heatmap, use_container_width=True)

st.write("### Selected Chart")
st.write(chart_type)

# Display Data
if st.checkbox('Show Data', key='show_data_checkbox'):
    st.write(filtered_df)

# Generate sample data for prediction
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31")
revenue = np.random.randint(1000, 5000, size=len(dates))

data = {
    'Date': dates,
    'Revenue': revenue
}

df_prediction = pd.DataFrame(data)

# Prepare the data for modeling
df_prediction['Date'] = pd.to_datetime(df_prediction['Date'])
df_prediction['Day'] = df_prediction['Date'].dt.dayofyear
X = df_prediction[['Day']]
y = df_prediction['Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"### Model Evaluation")
st.write(f"**Root Mean Square Error (RMSE):** {rmse:.2f}")

# Predict revenue for the next 30 days
future_dates = pd.date_range(start=df_prediction['Date'].max() + timedelta(days=1), periods=30)
future_days = pd.DataFrame({'Day': future_dates.dayofyear})
future_revenue_pred = model.predict(future_days)

# Create a DataFrame with the future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Revenue': future_revenue_pred})

# Display the future predictions
st.write("### Revenue Predictions for the Next 30 Days")
st.write(future_df)

# Plot the historical and predicted revenue
historical_chart = alt.Chart(df_prediction).mark_line().encode(
    x='Date',
    y='Revenue',
    tooltip=['Date', 'Revenue']
).properties(
    title='Historical Revenue'
)

future_chart = alt.Chart(future_df).mark_line(color='red').encode(
    x='Date',
    y='Predicted_Revenue',
    tooltip=['Date', 'Predicted_Revenue']
).properties(
    title='Predicted Revenue for the Next 30 Days'
)

combined_chart = alt.layer(historical_chart, future_chart).resolve_scale(
    y='independent'
).properties(
    width=800,
    height=400
)

st.altair_chart(combined_chart, use_container_width=True)

# Additional Visualization Options
st.write("## Additional Visualizations")

# Multiline Chart
if st.checkbox('Show Multiline Chart', key='multiline_chart_checkbox'):
    multiline_data = pd.melt(df_prediction, id_vars=['Date'], value_vars=['Revenue'])
    multiline_chart = alt.Chart(multiline_data).mark_line().encode(
        x='Date:T',
        y='value:Q',
        color='variable:N'
    ).properties(
        title='Multiline Chart'
    )
    st.altair_chart(multiline_chart, use_container_width=True)

# Waterfall Chart
if st.checkbox('Show Waterfall Chart', key='waterfall_chart_checkbox'):
    waterfall_data = future_df.copy()
    waterfall_data['Revenue Change'] = waterfall_data['Predicted_Revenue'].diff().fillna(waterfall_data['Predicted_Revenue'])
    waterfall_chart = go.Figure(go.Waterfall(
        x=waterfall_data['Date'].astype(str),
        y=waterfall_data['Revenue Change'],
        text=waterfall_data['Predicted_Revenue'],
        textposition='outside'
    ))
    waterfall_chart.update_layout(title='Waterfall Chart of Predicted Revenue Changes')
    st.plotly_chart(waterfall_chart, use_container_width=True)

# Heatmap
if st.checkbox('Show Heatmap', key='heatmap_chart_checkbox'):
    heatmap_data = df_prediction.pivot("Date", "Day", "Revenue")
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu")
    plt.title('Heatmap of Revenue')
    st.pyplot(plt)

# Slicer
if st.checkbox('Show Slicer', key='slicer_chart_checkbox'):
    date_slider = st.slider('Select date range:', min_value=min(df_prediction['Date']), max_value=max(df_prediction['Date']), value=(min(df_prediction['Date']), max(df_prediction['Date'])))
    sliced_df = df_prediction[(df_prediction['Date'] >= date_slider[0]) & (df_prediction['Date'] <= date_slider[1])]
    st.write(sliced_df)

# Predict future revenue for the next 30 days
st.write("## Predict Future Revenue for Next 30 Days")
future_days_30 = pd.date_range(start=pd.to_datetime('today'), periods=30)
future_days_df = pd.DataFrame({'Day': future_days_30.dayofyear})
future_revenue_30 = model.predict(future_days_df)
future_revenue_30_df = pd.DataFrame({'Date': future_days_30, 'Predicted_Revenue': future_revenue_30})
st.write(future_revenue_30_df)

# Plot the future predictions
future_prediction_chart = alt.Chart(future_revenue_30_df).mark_line(color='blue').encode(
    x='Date',
    y='Predicted_Revenue',
    tooltip=['Date', 'Predicted_Revenue']
).properties(
    title='Predicted Revenue for the Next 30 Days from Today'
)

st.altair_chart(future_prediction_chart, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# Simulate a dataset
np.random.seed(42)
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Sales': np.random.randint(100, 500, size=100),
    'Revenue': np.random.randint(1000, 5000, size=100),
    'Advertisement_Spend': np.random.randint(50, 200, size=100)
}
df = pd.DataFrame(data)
df['Cumulative_Sales'] = df['Sales'].cumsum()
df['Cumulative_Revenue'] = df['Revenue'].cumsum()

# Streamlit UI
st.title('Sales and Revenue Analysis Dashboard')

# Display Data
if st.checkbox('Show Data', key='show_data'):
    st.write(df)

# Multi-line Chart
if st.checkbox('Show Multi-Line Chart', key='multi_line_chart'):
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Sales'], label='Sales')
    ax.plot(df['Date'], df['Revenue'], label='Revenue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.legend()
    st.pyplot(fig)

# Waterfall Chart
if st.checkbox('Show Waterfall Chart', key='waterfall_chart'):
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative"] * len(df),
        x=df['Date'].dt.strftime('%Y-%m-%d'),
        y=df['Revenue'],
        textposition="outside",
        text=df['Revenue'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Revenue Waterfall Chart")
    st.plotly_chart(fig)

# Heat Map
if st.checkbox('Show Heat Map', key='heat_map'):
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Slicer for selecting date range
st.sidebar.header('Filter Data')
start_date = st.sidebar.date_input('Start Date', df['Date'].min().date())
end_date = st.sidebar.date_input('End Date', df['Date'].max().date())

# Ensure that start_date and end_date are converted to datetime64
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Display filtered data
if st.checkbox('Show Filtered Data', key='filtered_data'):
    st.write(filtered_df)

# Prediction for next 30 days
if st.checkbox('Show Revenue Prediction', key='revenue_prediction'):
    # Prepare the data
    df['Day'] = np.arange(len(df))
    X = df[['Day', 'Sales', 'Advertisement_Spend']]
    y = df['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)

    # Prediction
    future_days = np.arange(len(df), len(df) + 30)
    future_data = pd.DataFrame({
        'Day': future_days,
        'Sales': np.random.randint(100, 500, size=30),  # Simulate future sales
        'Advertisement_Spend': np.random.randint(50, 200, size=30)  # Simulate future ad spend
    })
    future_data['Revenue_Prediction'] = model.predict(future_data)

    st.write("### Revenue Predictions for the Next 30 Days")
    st.write(future_data[['Day', 'Revenue_Prediction']])

    # Plotting future predictions
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Revenue'], label='Actual Revenue')
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
    ax.plot(future_dates, future_data['Revenue_Prediction'], label='Predicted Revenue', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.legend()
    st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# Sample data
np.random.seed(42)
data = {
    'Area': ['North', 'South', 'East', 'West', 'Central'],
    'Sales': [150, 200, 300, 100, 250],
    'Revenue': [1000, 1500, 2500, 800, 1800],
    'Advertisement': [200, 250, 300, 150, 220]
}

df = pd.DataFrame(data)

# Streamlit UI
st.title('Dynamic Chart Visualization')

st.write("""
## Chart Visualization
Choose the type of chart you want to display.
""")

# Select chart type
chart_type = st.selectbox("Select the type of chart to display", ["Bar Chart", "Line Chart", "Pie Chart", "Radar Chart", "Column Chart", "Histogram"])

# Generate and display the selected chart
if chart_type == "Bar Chart":
    chart = alt.Chart(df).mark_bar().encode(
        x='Area',
        y='Sales'
    )
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Line Chart":
    chart = alt.Chart(df).mark_line().encode(
        x='Area',
        y='Revenue'
    )
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Pie Chart":
    df['Revenue_Percentage'] = df['Revenue'] / df['Revenue'].sum()
    chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field='Revenue_Percentage', type='quantitative'),
        color=alt.Color(field='Area', type='nominal')
    )
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Radar Chart":
    radar_df = pd.melt(df, id_vars=['Area'], value_vars=['Sales', 'Revenue', 'Advertisement'],
                       var_name='Metric', value_name='Value')
    fig = px.line_polar(radar_df, r='Value', theta='Metric', color='Area', line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Column Chart":
    chart = alt.Chart(df).mark_bar().encode(
        x='Area',
        y='Advertisement'
    ).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Histogram":
    sales_data = {'Sales': np.random.randint(100, 300, 100)}  # For histogram chart
    df_sales = pd.DataFrame(sales_data)
    chart = alt.Chart(df_sales).mark_bar().encode(
        alt.X("Sales:Q", bin=True),
        y='count()'
    ).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)

st.write("### Selected Chart")
st.write(chart_type)

# Instructions
st.write("""
### Instructions
- Use the dropdown menu to select the type of chart you want to visualize.
- The chart will update automatically based on your selection.
""")


import streamlit as st
import pandas as pd

# Sample campaign data
campaign_data = {
    'Campaign_ID': [1, 2, 3, 4, 5],
    'Campaign_Name': ['New Year Bash', 'Valentine\'s Day Special', 'Summer Festival', 'Halloween Night', 'Christmas Extravaganza'],
    'Date': ['2024-01-01', '2024-02-14', '2024-06-21', '2024-10-31', '2024-12-25'],
    'Location': ['New York', 'Paris', 'London', 'Los Angeles', 'Sydney'],
    'Price': [100, 150, 80, 120, 200]
}
df_campaigns = pd.DataFrame(campaign_data)

# Streamlit UI
st.title('Campaign Ticket Booking System')

# Display available campaigns
st.header('Available Campaigns')
st.write(df_campaigns)

# Booking form
st.header('Book Your Campaign Ticket')
selected_campaign_id = st.selectbox('Select Campaign', df_campaigns['Campaign_ID'])
name = st.text_input('Name')
email = st.text_input('Email')
num_tickets = st.number_input('Number of Tickets', min_value=1, max_value=10, value=1)

# Handle booking
if st.button('Book Ticket'):
    selected_campaign = df_campaigns[df_campaigns['Campaign_ID'] == selected_campaign_id].iloc[0]
    total_price = selected_campaign['Price'] * num_tickets

    st.write('### Booking Confirmation')
    st.write(f'**Name:** {name}')
    st.write(f'**Email:** {email}')
    st.write(f'**Campaign:** {selected_campaign["Campaign_Name"]}')
    st.write(f'**Date:** {selected_campaign["Date"]}')
    st.write(f'**Location:** {selected_campaign["Location"]}')
    st.write(f'**Number of Tickets:** {num_tickets}')
    st.write(f'**Total Price:** ${total_price}')
    st.success('Booking Successful!')

# Display details of the selected campaign
st.header('Campaign Details')
if selected_campaign_id:
    selected_campaign = df_campaigns[df_campaigns['Campaign_ID'] == selected_campaign_id].iloc[0]
    st.write(f'**Campaign Name:** {selected_campaign["Campaign_Name"]}')
    st.write(f'**Date:** {selected_campaign["Date"]}')
    st.write(f'**Location:** {selected_campaign["Location"]}')
    st.write(f'**Price per Ticket:** ${selected_campaign["Price"]}')


import streamlit as st
import pandas as pd
import datetime

# Sample customer visit data
visit_data = {
    'Customer_ID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'Location': ['Times Square', 'Downtown LA', 'Millennium Park', 'Pike Place Market', 'Space Needle', 'Times Square', 'Downtown LA', 'Millennium Park', 'Pike Place Market', 'Space Needle'],
    'Visit_Date': [
        '2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04', '2023-04-05',
        '2023-04-15', '2023-04-16', '2023-04-17', '2023-04-18', '2023-04-19'
    ]
}
df_visits = pd.DataFrame(visit_data)
df_visits['Visit_Date'] = pd.to_datetime(df_visits['Visit_Date'])

# Billboard content management
billboard_data = {
    'Location': ['Times Square', 'Downtown LA', 'Millennium Park', 'Pike Place Market', 'Space Needle'],
    'Content_14_Days': ['Ad A', 'Ad B', 'Ad C', 'Ad D', 'Ad E'],
    'Content_30_Days': ['Ad F', 'Ad G', 'Ad H', 'Ad I', 'Ad J']
}
df_billboards = pd.DataFrame(billboard_data)

# Streamlit UI
st.title('Billboard Content Management Based on Customer Visits')

# Display visit data
st.write("Customer Visit Data:")
st.write(df_visits)

# Display billboard content
st.write("Billboard Content:")
st.write(df_billboards)

# Select location to update
st.sidebar.header("Update Billboard Content")
location_filter = st.sidebar.selectbox("Select Location", df_billboards['Location'].unique())
days_filter = st.sidebar.radio("Select Duration", [14, 30])

# Update content
new_content = st.sidebar.text_input(f"New Content for {days_filter} Days")
if st.sidebar.button("Update Content"):
    if days_filter == 14:
        df_billboards.loc[df_billboards['Location'] == location_filter, 'Content_14_Days'] = new_content
    else:
        df_billboards.loc[df_billboards['Location'] == location_filter, 'Content_30_Days'] = new_content
    st.sidebar.success("Content Updated Successfully!")

# Display updated billboard content
st.write("Updated Billboard Content:")
st.write(df_billboards)

# Calculate visit frequencies
today = pd.to_datetime(datetime.date.today())
date_14_days_ago = today - pd.Timedelta(days=14)
date_30_days_ago = today - pd.Timedelta(days=30)

visit_freq_14_days = df_visits[df_visits['Visit_Date'] >= date_14_days_ago]['Location'].value_counts().reset_index()
visit_freq_14_days.columns = ['Location', 'Visit_Count_14_Days']

visit_freq_30_days = df_visits[df_visits['Visit_Date'] >= date_30_days_ago]['Location'].value_counts().reset_index()
visit_freq_30_days.columns = ['Location', 'Visit_Count_30_Days']

# Display visit frequencies
st.write("Visit Frequencies (Last 14 Days):")
st.write(visit_freq_14_days)

st.write("Visit Frequencies (Last 30 Days):")
st.write(visit_freq_30_days)

# Merge visit frequencies with billboard data
df_billboard_visits = pd.merge(df_billboards, visit_freq_14_days, on='Location', how='left').fillna(0)
df_billboard_visits = pd.merge(df_billboard_visits, visit_freq_30_days, on='Location', how='left').fillna(0)

# Display merged data
st.write("Billboard Content with Visit Frequencies:")
st.write(df_billboard_visits)

# Insights
st.write("""
## Insights for Billboard Content Management
- Update the billboard content based on the visit frequencies to maximize the impact of advertisements.
- Use the visit data to tailor the content specifically for the audience that frequently visits the locations.
- Regularly review and update the content to keep it relevant and engaging for the audience.
""")


import streamlit as st

# Streamlit UI
st.title('Dynamic Billboard Content')

st.write("""
## Billboard Animation Display
This billboard will display different animations based on customer interaction.
""")

# Simulate customer looking at the billboard
customer_looking = st.checkbox("Customer is looking at the billboard")

if customer_looking:
    st.write("### Customer detected!")
    content_type = st.selectbox("Select the type of animation to display", ["Ice Cream", "Food"])

    if content_type == "Ice Cream":
        st.video("C:/Users/SANGITA BISWAS/Downloads/855128-hd_1280_720_24fps.mp4")
    elif content_type == "Food":
        st.video("C:/Users/SANGITA BISWAS/Downloads/10200314-hd_2160_3840_25fps.mp4")
else:
    st.write("### No customer detected.")



def main():
    st.title("Innovative Aspects of Datacost Fusion")

    st.write("Datacost Fusion, as a concept, can be innovative in various ways. Here are some potential innovative aspects:")

    aspects = [
        "Real-Time Data Fusion",
        "AI-Powered Decision Making",
        "Predictive Analytics",
        "Dynamic Resource Allocation",
        "Cross-Domain Fusion",
        "Privacy-Preserving Fusion",
        "Blockchain Integration",
        "Edge Computing Fusion",
        "Human-Machine Collaboration",
        "Ethical Data Fusion Practices"
    ]

    st.write("### Innovative Aspects:")
    for aspect in aspects:
        st.write(f"- {aspect}")

if __name__ == "__main__":
    main()




def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Page 1", "Page 2", "Page 3", "Page 4"])

    if selection == "Page 1":
        page1()
    elif selection == "Page 2":
        pass
    elif selection == "Page 3":
        pass
    elif selection == "Page 4":
        pass
    
if __name__ == "__main__":
    main()



