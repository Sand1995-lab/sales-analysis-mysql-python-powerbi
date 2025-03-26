#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas mysql-connector-python


# In[2]:


import pandas as pd
import mysql.connector


# # Connect to MySQL database

# In[3]:


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="sales_db"
)


# # Load data into pandas DataFrames

# In[4]:


pip install sqlalchemy


# In[5]:


import pandas as pd
from sqlalchemy import create_engine


# # # Create a SQLAlchemy engine
# # Replace 'mysql+mysqlconnector' with the appropriate dialect if you're using a different database

# In[6]:


engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')


# # Fetch data from the 'products', 'customers', and 'sales' tables

# In[7]:


products_df = pd.read_sql("SELECT * FROM products", engine)
customers_df = pd.read_sql("SELECT * FROM customers", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)


# # Display the first few rows of each DataFrame (optional)

# In[8]:


print("Products Table:")


# In[9]:


print(products_df.head())


# In[10]:


print("\nCustomers Table:")
print(customers_df.head())


# In[11]:


print("\nSales Table:")
print(sales_df.head())


# In[12]:


# Check for missing values
print(products_df.isnull().sum())
print(customers_df.isnull().sum())
print(sales_df.isnull().sum())

# Remove duplicates (if any)
products_df.drop_duplicates(inplace=True)
customers_df.drop_duplicates(inplace=True)
sales_df.drop_duplicates(inplace=True)

# Create a new column in sales_df for total_sale_amount
sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']


# In[9]:


# Calculate total revenue
total_revenue = sales_df['total_sale_amount'].sum()
print(f"Total Revenue: ${total_revenue:.2f}")

# Average sale amount
average_sale = sales_df['total_sale_amount'].mean()
print(f"Average Sale Amount: ${average_sale:.2f}")

# Number of unique customers
unique_customers = customers_df['customer_id'].nunique()
print(f"Number of Unique Customers: {unique_customers}")


# In[13]:


# Save to MySQL
sales_df.to_sql('cleaned_sales', con=engine, if_exists='replace', index=False)

# Save to Excel
with pd.ExcelWriter('sales_data.xlsx') as writer:
    products_df.to_excel(writer, sheet_name='Products', index=False)
    customers_df.to_excel(writer, sheet_name='Customers', index=False)
    sales_df.to_excel(writer, sheet_name='Sales', index=False)


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Revenue by product category
revenue_by_category = sales_df.groupby('product_id')['total_sale_amount'].sum().reset_index()
revenue_by_category = pd.merge(revenue_by_category, products_df[['product_id', 'category']], on='product_id', how='left')

plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='total_sale_amount', data=revenue_by_category)
plt.title('Revenue by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Revenue')
plt.show()


# In[12]:


# Save the entire process in a script (e.g., data_pipeline.py)
def data_pipeline():
    # Step 1: Fetch data
    engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')
    products_df = pd.read_sql("SELECT * FROM products", engine)
    customers_df = pd.read_sql("SELECT * FROM customers", engine)
    sales_df = pd.read_sql("SELECT * FROM sales", engine)

    # Step 2: Clean and transform data
    sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
    sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

    # Step 3: Save cleaned data
    sales_df.to_sql('cleaned_sales', con=engine, if_exists='replace', index=False)

    print("Data pipeline executed successfully!")

# Run the pipeline
if __name__ == "__main__":
    data_pipeline()


# In[15]:


import pandas as pd
from sqlalchemy import create_engine

# Create a SQLAlchemy engine with your database credentials
engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')

# Fetch data from the 'products', 'customers', and 'sales' tables
products_df = pd.read_sql("SELECT * FROM products", engine)
customers_df = pd.read_sql("SELECT * FROM customers", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)

# Clean and transform the data (if needed)
# Example: Calculate total_sale_amount in sales_df
sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

# Save the cleaned data to an Excel file
with pd.ExcelWriter('cleaned_sales_data.xlsx') as writer:
    products_df.to_excel(writer, sheet_name='Products', index=False)
    customers_df.to_excel(writer, sheet_name='Customers', index=False)
    sales_df.to_excel(writer, sheet_name='Sales', index=False)

print("Cleaned data saved to 'cleaned_sales_data.xlsx'")


# In[16]:


import pandas as pd
from sqlalchemy import create_engine

# Create a SQLAlchemy engine with your database credentials
engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')

# Fetch data from the 'products' and 'sales' tables
products_df = pd.read_sql("SELECT * FROM products", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)

# Clean and transform the data (if needed)
# Example: Calculate total_sale_amount in sales_df
sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

# Group by product_id and calculate total revenue for each product
product_revenue = sales_df.groupby('product_id')['total_sale_amount'].sum().reset_index()

# Merge with products_df to get product names and categories
product_revenue = pd.merge(product_revenue, products_df[['product_id', 'product_name', 'category']], on='product_id', how='left')

# Sort by total revenue in descending order
product_revenue_sorted = product_revenue.sort_values(by='total_sale_amount', ascending=False)

# Select the top 5 products by revenue
top_5_products = product_revenue_sorted.head(5)

# Display the top 5 products
print("Top 5 Products by Revenue:")
print(top_5_products)

# Save the top 5 products to an Excel file (optional)
top_5_products.to_excel('top_5_products_by_revenue.xlsx', index=False)
print("Top 5 products saved to 'top_5_products_by_revenue.xlsx'")


# In[17]:


import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Create a SQLAlchemy engine with your database credentials
engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')

# Fetch data from the 'products' and 'sales' tables
products_df = pd.read_sql("SELECT * FROM products", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)

# Clean and transform the data (if needed)
# Example: Calculate total_sale_amount in sales_df
sales_df = pd.merge(sales_df, products_df[['product_id', 'price', 'category']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

# Group by category and calculate total revenue for each category
revenue_by_category = sales_df.groupby('category')['total_sale_amount'].sum().reset_index()

# Sort by total revenue in descending order (optional)
revenue_by_category = revenue_by_category.sort_values(by='total_sale_amount', ascending=False)

# Display the revenue by category
print("Revenue by Product Category:")
print(revenue_by_category)

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='total_sale_amount', data=revenue_by_category, palette='viridis')
plt.title('Revenue by Product Category', fontsize=16)
plt.xlabel('Product Category', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# In[18]:


sns.barplot(x='category', y='total_sale_amount', data=revenue_by_category, palette='coolwarm')


# In[19]:


ax = sns.barplot(x='category', y='total_sale_amount', data=revenue_by_category, palette='viridis')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')


# In[20]:


import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Create a SQLAlchemy engine with your database credentials
engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')

# Fetch data from the 'products' and 'sales' tables
products_df = pd.read_sql("SELECT * FROM products", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)

# Clean and transform the data
# Merge sales with products to get price
sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

# Convert sale_date to datetime format
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])

# Filter data for 2023
sales_2023 = sales_df[sales_df['sale_date'].dt.year == 2023]

# Group by month and calculate total revenue
monthly_revenue = sales_2023.groupby(sales_2023['sale_date'].dt.to_period('M'))['total_sale_amount'].sum().reset_index()
monthly_revenue['sale_date'] = monthly_revenue['sale_date'].dt.to_timestamp()  # Convert period to timestamp for plotting

# Display the monthly revenue
print("Monthly Revenue Trends for 2023:")
print(monthly_revenue)

# Create a line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='sale_date', y='total_sale_amount', data=monthly_revenue, marker='o', color='blue')
plt.title('Monthly Revenue Trends for 2023', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for clarity
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Optional: Save the chart as an image
plt.savefig('monthly_revenue_2023.png', dpi=300, bbox_inches='tight')


# In[21]:


import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# Create a SQLAlchemy engine with your database credentials
engine = create_engine('mysql+mysqlconnector://root:admin@localhost/sales_db')

# Fetch data from the 'products' and 'sales' tables
products_df = pd.read_sql("SELECT * FROM products", engine)
sales_df = pd.read_sql("SELECT * FROM sales", engine)

# Clean and transform the data
# Merge sales with products to get price
sales_df = pd.merge(sales_df, products_df[['product_id', 'price']], on='product_id', how='left')
sales_df['total_sale_amount'] = sales_df['quantity_sold'] * sales_df['price']

# Group by region and calculate total revenue
revenue_by_region = sales_df.groupby('region')['total_sale_amount'].sum().reset_index()

# Display the revenue by region
print("Revenue by Region:")
print(revenue_by_region)

# For mapping, we need a way to associate regions with geographical data
# Since the dataset only has 'North', 'South', 'East', 'West', we'll simulate U.S. regions with state codes
# This is a simplification; adjust as needed for your context
region_mapping = {
    'North': ['ND', 'SD', 'NE', 'MN', 'IA', 'WI', 'MI'],  # Example northern U.S. states
    'South': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA'],  # Example southern U.S. states
    'East': ['NY', 'PA', 'NJ', 'MD', 'VA', 'NC', 'SC'],   # Example eastern U.S. states
    'West': ['CA', 'NV', 'OR', 'WA', 'ID', 'MT', 'WY']    # Example western U.S. states
}

# Expand the regions into a DataFrame with state codes
region_expanded = []
for region, states in region_mapping.items():
    region_data = revenue_by_region[revenue_by_region['region'] == region]
    if not region_data.empty:
        revenue = region_data['total_sale_amount'].values[0]
        for state in states:
            region_expanded.append({'region': region, 'state': state, 'total_sale_amount': revenue})

region_expanded_df = pd.DataFrame(region_expanded)

# Create a choropleth map using Plotly
fig = px.choropleth(
    region_expanded_df,
    locations='state',
    locationmode='USA-states',
    color='total_sale_amount',
    scope='usa',
    color_continuous_scale='Viridis',
    title='Revenue by Region',
    labels={'total_sale_amount': 'Total Revenue'},
    hover_data=['region', 'total_sale_amount']
)

# Update layout for better visualization
fig.update_layout(
    title_text='Revenue by Region',
    geo=dict(
        lakecolor='rgb(255, 255, 255)',
    )
)

# Show the map
fig.show()

# Optional: Save the map as an HTML file
fig.write_html('revenue_by_region.html')
print("Map saved as 'revenue_by_region.html'")


# In[22]:


# Calculate total spend and purchase frequency per customer
customer_summary = sales_df.groupby('customer_id').agg(
    total_spend=('total_sale_amount', 'sum'),
    purchase_count=('sale_id', 'count')
).reset_index()

# Merge with customers_df for demographic data
customer_summary = pd.merge(customer_summary, customers_df, on='customer_id', how='left')

# Segment customers (e.g., High, Medium, Low spenders)
customer_summary['spend_segment'] = pd.qcut(customer_summary['total_spend'], q=3, labels=['Low', 'Medium', 'High'])

# Display the result
print("Customer Segmentation:")
print(customer_summary.head())


# In[25]:


# Convert sale_date to datetime
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])

# Group by month
monthly_sales = sales_df.groupby(sales_df['sale_date'].dt.to_period('M'))['total_sale_amount'].sum().reset_index()
monthly_sales['sale_date'] = monthly_sales['sale_date'].dt.to_timestamp()

# Calculate month-over-month growth
monthly_sales['previous_month_revenue'] = monthly_sales['total_sale_amount'].shift(1)
monthly_sales['growth_rate'] = ((monthly_sales['total_sale_amount'] - monthly_sales['previous_month_revenue']) / monthly_sales['previous_month_revenue']) * 100

# Display the result
print("Monthly Sales Growth:")
print(monthly_sales.head())


# In[26]:


# Calculate total revenue per customer
top_customers = sales_df.groupby('customer_id')['total_sale_amount'].sum().reset_index()
top_customers = pd.merge(top_customers, customers_df[['customer_id', 'customer_name']], on='customer_id', how='left')
top_customers = top_customers.sort_values(by='total_sale_amount', ascending=False).head(5)

# Display the result
print("Top 5 Customers by Revenue:")
print(top_customers)


# In[ ]:




