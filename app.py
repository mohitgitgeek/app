import streamlit as st
import pandas as pd
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Amazon Sales Data Analysis')

# Download the dataset
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
    file_path = os.path.join(path, 'amazon.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        return None

df = load_data()

if df is not None:
    st.write("### Raw Data")
    st.dataframe(df.head())

    # Data Cleaning and Preparation
    df['actual_price_numeric'] = df['actual_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['actual_price_numeric'] = pd.to_numeric(df['actual_price_numeric'], errors='coerce')

    df['discounted_price_numeric'] = df['discounted_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['discounted_price_numeric'] = pd.to_numeric(df['discounted_price_numeric'], errors='coerce')

    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')

    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)


    st.write("### Data Info")
    buffer = pd.io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("### Data Description")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())


    # Objective 1: Revenue Generation Analysis
    st.write("### Revenue Generation Analysis")

    total_revenue = df['actual_price_numeric'].sum()
    st.write(f"Total Revenue (based on actual price): ₹{total_revenue:,.2f}")

    st.write("Top Products (based on average actual price):")
    top_products_price = df.groupby('product_name')['actual_price_numeric'].mean().sort_values(ascending=False).head(10)
    st.dataframe(top_products_price)

    if 'Location' in df.columns:
        st.write("Key Sales Locations (based on order count):")
        key_locations = df['Location'].value_counts().head(10)
        st.dataframe(key_locations)

    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        daily_activity = df.groupby(df['review_date'].dt.date).size()
        st.write("Daily Activity Patterns (based on review date):")
        fig, ax = plt.subplots(figsize=(12, 6))
        daily_activity.plot(ax=ax)
        ax.set_title('Daily Activity Patterns (based on review date)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Reviews')
        st.pyplot(fig)
        plt.close(fig)


    # Objective 2: Shipping and Performance Evaluation
    st.write("### Shipping and Performance Evaluation")

    if 'Fulfillment' in df.columns:
        st.write("Fulfillment Method Analysis (based on order count):")
        fulfillment_analysis = df.groupby('Fulfillment').size()
        st.dataframe(fulfillment_analysis)

    if 'Region' in df.columns:
        st.write("Regional Activity Contribution (based on order count):")
        regional_activity = df.groupby('Region').size()
        st.dataframe(regional_activity)

    # Sales Quantity by Category
    if 'Category' in df.columns and 'Quantity' in df.columns:
        df['Quantity_numeric'] = pd.to_numeric(df['Quantity'], errors='coerce')
        st.write("Sales Quantities by Category:")
        sales_by_category = df.groupby('Category')['Quantity_numeric'].sum().sort_values(ascending=False).head(10)
        st.dataframe(sales_by_category)
    else:
        st.write("Error: 'Category' or 'Quantity' column not found for sales by category analysis.")

    # Visualizations
    st.write("### Visualizations")

    # 1. Distribution of Actual Prices
    st.write("Distribution of Actual Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['actual_price_numeric'].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Actual Prices')
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.close(fig)

    # 2. Top 10 Product Categories by Count
    st.write("Top 10 Product Categories by Count")
    plt.figure(figsize=(12, 7))
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(y='main_category', data=df, order=df['main_category'].value_counts().index[:10], ax=ax)
    ax.set_title('Top 10 Product Categories by Count')
    ax.set_xlabel('Count')
    ax.set_ylabel('Main Category')
    st.pyplot(fig)
    plt.close(fig)


    # 3. Distribution of Ratings
    if 'rating_numeric' in df.columns:
        st.write("Distribution of Ratings")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['rating_numeric'].dropna(), bins=10, kde=True, ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("'rating_numeric' column not found for plotting rating distribution.")

    # Correlation Heatmap
    st.write("Correlation Heatmap of Numeric Columns")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Columns')
    st.pyplot(fig)
    plt.close(fig)


else:
    st.error("Error loading the dataset.")
