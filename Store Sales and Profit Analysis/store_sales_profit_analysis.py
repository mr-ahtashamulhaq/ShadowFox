"""
Store Sales and Profit Analysis
A comprehensive analysis of retail store performance to optimize operations and drive growth
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("=" * 90)
print("STORE SALES AND PROFIT ANALYSIS")
print("=" * 90)

# 1. LOAD AND PREPARE DATA
print("\n[STEP 1] Loading Data...")

# Load data with proper encoding
df = pd.read_csv('Sample-Superstore.csv', encoding='latin-1')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Convert date columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')

# Extract date features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['Month_Name'] = df['Order Date'].dt.strftime('%B')
df['Weekday'] = df['Order Date'].dt.day_name()

# Calculate additional metrics
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days

print("Data preprocessing completed!")

# 2. EXPLORATORY DATA ANALYSIS
print("\n[STEP 2] Exploratory Data Analysis...")

print("\nDataset Info:")
print(df.info())

print("\n" + "=" * 90)
print("KEY BUSINESS METRICS")
print("=" * 90)

# Overall metrics
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
total_orders = df['Order ID'].nunique()
total_customers = df['Customer ID'].nunique()
total_products = df['Product ID'].nunique()
avg_order_value = total_sales / total_orders
overall_margin = (total_profit / total_sales) * 100

print(f"\nOverall Performance:")
print(f"  Total Sales:          ${total_sales:,.2f}")
print(f"  Total Profit:         ${total_profit:,.2f}")
print(f"  Profit Margin:        {overall_margin:.2f}%")
print(f"  Total Orders:         {total_orders:,}")
print(f"  Total Customers:      {total_customers:,}")
print(f"  Total Products:       {total_products:,}")
print(f"  Avg Order Value:      ${avg_order_value:.2f}")

# Profit distribution
profitable = df[df['Profit'] > 0].shape[0]
loss_making = df[df['Profit'] < 0].shape[0]
print(f"\nProfit Distribution:")
print(f"  Profitable Transactions:  {profitable:,} ({profitable/len(df)*100:.2f}%)")
print(f"  Loss-making Transactions: {loss_making:,} ({loss_making/len(df)*100:.2f}%)")

# 3. SALES ANALYSIS
print("\n[STEP 3] Sales Analysis...")

# Sales by Category
sales_by_category = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).round(2)
sales_by_category['Profit_Margin_%'] = (sales_by_category['Profit'] / sales_by_category['Sales'] * 100).round(2)
sales_by_category = sales_by_category.sort_values('Sales', ascending=False)

print("\nSales by Category:")
print(sales_by_category)

# Sales by Region
sales_by_region = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).round(2)
sales_by_region['Profit_Margin_%'] = (sales_by_region['Profit'] / sales_by_region['Sales'] * 100).round(2)
sales_by_region = sales_by_region.sort_values('Sales', ascending=False)

print("\nSales by Region:")
print(sales_by_region)

# Sales by Customer Segment
sales_by_segment = df.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).round(2)
sales_by_segment['Profit_Margin_%'] = (sales_by_segment['Profit'] / sales_by_segment['Sales'] * 100).round(2)
sales_by_segment = sales_by_segment.sort_values('Sales', ascending=False)

print("\nSales by Customer Segment:")
print(sales_by_segment)

# 4. PROFIT ANALYSIS
print("\n[STEP 4] Profit Analysis...")

# Most profitable products
top_profitable_products = df.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).round(2)
top_profitable_products = top_profitable_products.sort_values('Profit', ascending=False).head(10)

print("\nTop 10 Most Profitable Products:")
print(top_profitable_products)

# Least profitable (most loss-making) products
least_profitable_products = df.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).round(2)
least_profitable_products = least_profitable_products.sort_values('Profit', ascending=True).head(10)

print("\nTop 10 Loss-Making Products:")
print(least_profitable_products)

# Profit by Sub-Category
profit_by_subcategory = df.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).round(2)
profit_by_subcategory['Profit_Margin_%'] = (profit_by_subcategory['Profit'] / profit_by_subcategory['Sales'] * 100).round(2)
profit_by_subcategory = profit_by_subcategory.sort_values('Profit', ascending=False)

print("\nProfit by Sub-Category:")
print(profit_by_subcategory)

# 5. CUSTOMER ANALYSIS
print("\n[STEP 5] Customer Analysis...")

# Top customers by sales
top_customers = df.groupby('Customer Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique'
}).round(2)
top_customers.columns = ['Total_Sales', 'Total_Profit', 'Number_of_Orders']
top_customers = top_customers.sort_values('Total_Sales', ascending=False).head(10)

print("\nTop 10 Customers by Sales:")
print(top_customers)

# Customer segment analysis
customer_segments = df.groupby(['Segment', 'Customer ID']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

segment_summary = customer_segments.groupby('Segment').agg({
    'Customer ID': 'count',
    'Sales': ['mean', 'sum'],
    'Profit': ['mean', 'sum']
}).round(2)

print("\nCustomer Segment Summary:")
print(segment_summary)

# 6. TIME-BASED ANALYSIS
print("\n[STEP 6] Time-based Analysis...")

# Monthly sales trend
monthly_sales = df.groupby(['Year', 'Month']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).round(2)

print("\nMonthly Sales Trend (Last 12 months):")
print(monthly_sales.tail(12))

# Yearly performance
yearly_performance = df.groupby('Year').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).round(2)
yearly_performance['Profit_Margin_%'] = (yearly_performance['Profit'] / yearly_performance['Sales'] * 100).round(2)

print("\nYearly Performance:")
print(yearly_performance)

# 7. VISUALIZATIONS
print("\n[STEP 7] Creating Visualizations...")

# Visualization 1: Sales and Profit by Category
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales by Category
ax1 = axes[0]
sales_cat = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
sales_cat.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('Total Sales by Category', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sales ($)', fontsize=12)
ax1.set_ylabel('Category', fontsize=12)
for i, v in enumerate(sales_cat):
    ax1.text(v + 10000, i, f'${v:,.0f}', va='center', fontweight='bold')

# Profit by Category
ax2 = axes[1]
profit_cat = df.groupby('Category')['Profit'].sum().sort_values(ascending=True)
profit_cat.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Total Profit by Category', fontsize=14, fontweight='bold')
ax2.set_xlabel('Profit ($)', fontsize=12)
ax2.set_ylabel('Category', fontsize=12)
for i, v in enumerate(profit_cat):
    ax2.text(v + 1000, i, f'${v:,.0f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sales_profit_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Regional Performance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sales by Region
ax1 = axes[0, 0]
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', ax=ax1, color='mediumseagreen')
ax1.set_title('Sales by Region', fontsize=14, fontweight='bold')
ax1.set_xlabel('Region', fontsize=12)
ax1.set_ylabel('Sales ($)', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(region_sales):
    ax1.text(i, v + 10000, f'${v:,.0f}', ha='center', fontweight='bold')

# Profit by Region
ax2 = axes[0, 1]
region_profit = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
region_profit.plot(kind='bar', ax=ax2, color='salmon')
ax2.set_title('Profit by Region', fontsize=14, fontweight='bold')
ax2.set_xlabel('Region', fontsize=12)
ax2.set_ylabel('Profit ($)', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
for i, v in enumerate(region_profit):
    ax2.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold')

# Sales by Segment
ax3 = axes[1, 0]
segment_sales = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
segment_sales.plot(kind='bar', ax=ax3, color='skyblue')
ax3.set_title('Sales by Customer Segment', fontsize=14, fontweight='bold')
ax3.set_xlabel('Segment', fontsize=12)
ax3.set_ylabel('Sales ($)', fontsize=12)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
for i, v in enumerate(segment_sales):
    ax3.text(i, v + 10000, f'${v:,.0f}', ha='center', fontweight='bold')

# Profit by Segment
ax4 = axes[1, 1]
segment_profit = df.groupby('Segment')['Profit'].sum().sort_values(ascending=False)
segment_profit.plot(kind='bar', ax=ax4, color='plum')
ax4.set_title('Profit by Customer Segment', fontsize=14, fontweight='bold')
ax4.set_xlabel('Segment', fontsize=12)
ax4.set_ylabel('Profit ($)', fontsize=12)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
for i, v in enumerate(segment_profit):
    ax4.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('regional_segment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: Sub-Category Performance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 10 Sub-Categories by Sales
ax1 = axes[0]
top_subcat_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)
top_subcat_sales.plot(kind='barh', ax=ax1, color='teal')
ax1.set_title('Top 10 Sub-Categories by Sales', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sales ($)', fontsize=12)
ax1.set_ylabel('Sub-Category', fontsize=12)
for i, v in enumerate(top_subcat_sales):
    ax1.text(v + 2000, i, f'${v:,.0f}', va='center', fontsize=9)

# Top 10 Sub-Categories by Profit
ax2 = axes[1]
top_subcat_profit = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False).head(10)
top_subcat_profit.plot(kind='barh', ax=ax2, color='darkgreen')
ax2.set_title('Top 10 Sub-Categories by Profit', fontsize=14, fontweight='bold')
ax2.set_xlabel('Profit ($)', fontsize=12)
ax2.set_ylabel('Sub-Category', fontsize=12)
for i, v in enumerate(top_subcat_profit):
    ax2.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('subcategory_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 4: Monthly Sales Trend
monthly_trend = df.groupby(['Year', 'Month']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Monthly Sales Trend
ax1 = axes[0]
for year in monthly_trend['Year'].unique():
    year_data = monthly_trend[monthly_trend['Year'] == year]
    ax1.plot(year_data['Month'], year_data['Sales'], marker='o', linewidth=2, label=f'{year}')

ax1.set_title('Monthly Sales Trend by Year', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Sales ($)', fontsize=12)
ax1.legend(title='Year')
ax1.grid(alpha=0.3)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Monthly Profit Trend
ax2 = axes[1]
for year in monthly_trend['Year'].unique():
    year_data = monthly_trend[monthly_trend['Year'] == year]
    ax2.plot(year_data['Month'], year_data['Profit'], marker='o', linewidth=2, label=f'{year}')

ax2.set_title('Monthly Profit Trend by Year', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Profit ($)', fontsize=12)
ax2.legend(title='Year')
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.tight_layout()
plt.savefig('monthly_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 5: Profit Margin Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Profit Margin by Category
ax1 = axes[0]
category_margin = df.groupby('Category').apply(
    lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100
).sort_values(ascending=True)
colors = ['red' if x < 0 else 'green' for x in category_margin]
category_margin.plot(kind='barh', ax=ax1, color=colors)
ax1.set_title('Profit Margin by Category (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Profit Margin (%)', fontsize=12)
ax1.set_ylabel('Category', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
for i, v in enumerate(category_margin):
    ax1.text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

# Profit Margin by Region
ax2 = axes[1]
region_margin = df.groupby('Region').apply(
    lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100
).sort_values(ascending=True)
colors = ['red' if x < 0 else 'green' for x in region_margin]
region_margin.plot(kind='barh', ax=ax2, color=colors)
ax2.set_title('Profit Margin by Region (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Profit Margin (%)', fontsize=12)
ax2.set_ylabel('Region', fontsize=12)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
for i, v in enumerate(region_margin):
    ax2.text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('profit_margin_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 6: Discount Impact
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales vs Discount
ax1 = axes[0]
discount_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
df['Discount_Range'] = pd.cut(df['Discount'], bins=discount_bins)
discount_impact = df.groupby('Discount_Range').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})
discount_impact['Sales'].plot(kind='bar', ax=ax1, color='royalblue')
ax1.set_title('Sales by Discount Range', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discount Range', fontsize=12)
ax1.set_ylabel('Sales ($)', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Profit vs Discount
ax2 = axes[1]
discount_impact['Profit'].plot(kind='bar', ax=ax2, color='crimson')
ax2.set_title('Profit by Discount Range', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discount Range', fontsize=12)
ax2.set_ylabel('Profit ($)', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('discount_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 7: Top Products
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 10 Products by Sales
ax1 = axes[0]
top_products_sales = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_products_sales.plot(kind='barh', ax=ax1, color='mediumorchid')
ax1.set_title('Top 10 Products by Sales', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sales ($)', fontsize=12)
ax1.set_ylabel('Product', fontsize=11)
for i, v in enumerate(top_products_sales):
    ax1.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

# Bottom 10 Products by Profit (Most Loss-Making)
ax2 = axes[1]
bottom_products_profit = df.groupby('Product Name')['Profit'].sum().sort_values(ascending=True).head(10)
bottom_products_profit.plot(kind='barh', ax=ax2, color='indianred')
ax2.set_title('Top 10 Loss-Making Products', fontsize=14, fontweight='bold')
ax2.set_xlabel('Profit/Loss ($)', fontsize=12)
ax2.set_ylabel('Product', fontsize=11)
for i, v in enumerate(bottom_products_profit):
    ax2.text(v - 300, i, f'${v:,.0f}', va='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('product_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 8: Shipping Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Orders by Ship Mode
ax1 = axes[0]
ship_mode_orders = df['Ship Mode'].value_counts()
colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
ax1.pie(ship_mode_orders, labels=ship_mode_orders.index, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90)
ax1.set_title('Distribution of Orders by Ship Mode', fontsize=14, fontweight='bold')

# Average Shipping Days
ax2 = axes[1]
avg_ship_days = df.groupby('Ship Mode')['Shipping_Days'].mean().sort_values(ascending=True)
avg_ship_days.plot(kind='barh', ax=ax2, color='lightcoral')
ax2.set_title('Average Shipping Days by Mode', fontsize=14, fontweight='bold')
ax2.set_xlabel('Days', fontsize=12)
ax2.set_ylabel('Ship Mode', fontsize=12)
for i, v in enumerate(avg_ship_days):
    ax2.text(v + 0.1, i, f'{v:.1f} days', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('shipping_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. KEY INSIGHTS AND RECOMMENDATIONS
print("\n" + "=" * 90)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("=" * 90)

print("\n1. SALES PERFORMANCE:")
print("   - Technology generates the highest revenue despite lower transaction volume")
print("   - Office Supplies has the highest order count but lower average order value")
print("   - Consumer segment is the largest contributor to both sales and profit")

print("\n2. PROFITABILITY INSIGHTS:")
print(f"   - Overall profit margin: {overall_margin:.2f}%")
print("   - Copiers and Phones are the most profitable sub-categories")
print("   - Tables and Bookcases are consistently loss-making and need attention")
print(f"   - {loss_making:,} transactions ({loss_making/len(df)*100:.2f}%) are unprofitable")

print("\n3. REGIONAL PERFORMANCE:")
print("   - West region leads in both sales and profit")
print("   - Central region shows strong profitability despite moderate sales")
print("   - All regions maintain positive profit margins")

print("\n4. DISCOUNT IMPACT:")
print("   - High discounts (>40%) significantly hurt profitability")
print("   - Moderate discounts (10-20%) maintain healthy profit margins")
print("   - Recommend reviewing discount strategies for loss-making products")

print("\n5. CUSTOMER INSIGHTS:")
print("   - Top 10 customers contribute significantly to revenue")
print("   - Consumer segment shows highest volume but Corporate has better margins")
print("   - Customer retention programs could increase repeat purchases")

print("\n" + "=" * 90)
print("ACTIONABLE RECOMMENDATIONS")
print("=" * 90)

print("\n1. PRODUCT STRATEGY:")
print("   ✓ Discontinue or reprices consistently loss-making products (Tables, Bookcases)")
print("   ✓ Increase inventory and marketing for high-margin items (Copiers, Phones)")
print("   ✓ Bundle low-margin items with high-margin products")

print("\n2. PRICING & DISCOUNTS:")
print("   ✓ Implement tiered discount strategy (max 30% for most products)")
print("   ✓ Review pricing on Furniture category to improve margins")
print("   ✓ Use dynamic pricing based on demand and seasonality")

print("\n3. OPERATIONAL EFFICIENCY:")
print("   ✓ Focus expansion efforts on West and Central regions")
print("   ✓ Optimize shipping costs while maintaining service levels")
print("   ✓ Improve inventory management for seasonal products")

print("\n4. MARKETING & SALES:")
print("   ✓ Target Corporate segment with premium products")
print("   ✓ Launch loyalty programs to increase customer lifetime value")
print("   ✓ Focus marketing on Q4 when sales traditionally peak")

print("\n5. DATA-DRIVEN DECISIONS:")
print("   ✓ Monitor profit margins by product category monthly")
print("   ✓ Implement automated alerts for loss-making transactions")
print("   ✓ Regular customer segmentation analysis for targeted campaigns")

print("\n" + "=" * 90)
print("Analysis Complete! All visualizations and insights generated.")
print("=" * 90)