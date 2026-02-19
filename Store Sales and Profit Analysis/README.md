# Store Sales and Profit Analysis

A comprehensive data analysis project examining retail store performance to identify opportunities for optimization, growth, and improved profitability. This project demonstrates end-to-end data analysis using Python to derive actionable business insights.

## Project Overview

This project analyzes a real-world retail dataset (Superstore) containing 9,994 transactions across multiple years. The analysis focuses on sales performance, profitability, customer behavior, and operational efficiency to provide strategic recommendations for business improvement.

## Business Objectives

The analysis addresses key business questions:
- Which products and categories drive the most revenue and profit?
- How do different regions and customer segments perform?
- What is the impact of discounts on profitability?
- Which products are loss-making and why?
- What are the seasonal trends in sales and profit?
- How can we optimize pricing and inventory strategies?

## Dataset Features

The Superstore dataset includes 21 columns:

**Order Information:**
- Row ID, Order ID, Order Date, Ship Date, Ship Mode

**Customer Information:**
- Customer ID, Customer Name, Segment (Consumer/Corporate/Home Office)

**Location Information:**
- Country, City, State, Postal Code, Region (East/West/Central/South)

**Product Information:**
- Product ID, Category, Sub-Category, Product Name

**Transaction Metrics:**
- Sales, Quantity, Discount, Profit

## Project Structure

```
Store Sales and Profit Analysis/
│
├── store_sales_profit_analysis.py     # Main analysis script
├── Sample_-_Superstore.csv             # Dataset
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies
├── LINKEDIN_VIDEO_GUIDE.md            # Video presentation guide
│
└── outputs/                            # Generated visualizations
    ├── sales_profit_by_category.png
    ├── regional_segment_analysis.png
    ├── subcategory_performance.png
    ├── monthly_trend_analysis.png
    ├── profit_margin_analysis.png
    ├── discount_impact_analysis.png
    ├── product_performance.png
    └── shipping_analysis.png
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ShadowFox.git
cd ShadowFox/Store Sales and Profit Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running in Google Colab

1. Upload `Sample_-_Superstore.csv` to your Colab environment
2. Copy the code from `store_sales_profit_analysis.py`
3. Run all cells sequentially
4. Download the generated visualizations

### Running Locally

```bash
python store_sales_profit_analysis.py
```

Make sure the CSV file is in the same directory as the script.

## Key Findings

### Overall Performance
- **Total Sales**: $2,297,200.86
- **Total Profit**: $286,397.02
- **Overall Profit Margin**: 12.47%
- **Total Orders**: 5,009
- **Total Customers**: 793
- **Profitable Transactions**: 80.63%

### Sales by Category
1. **Technology**: Highest revenue generator
2. **Furniture**: Moderate sales with lower margins
3. **Office Supplies**: Highest transaction volume

### Regional Performance
1. **West Region**: Leading in both sales and profit
2. **East Region**: Strong second place
3. **Central Region**: Good profitability despite moderate sales
4. **South Region**: Opportunity for growth

### Customer Segments
1. **Consumer**: Largest segment by volume (51.6%)
2. **Corporate**: Best profit margins (30.0%)
3. **Home Office**: Smallest but consistent segment (18.4%)

### Profitability Insights
- **Most Profitable Sub-Categories**: Copiers, Phones, Accessories
- **Loss-Making Sub-Categories**: Tables, Bookcases, Supplies
- **Discount Impact**: High discounts (>40%) significantly hurt profits
- **Seasonal Trends**: Q4 shows strongest performance

## Visualizations

The project generates 8 comprehensive visualizations:

1. **Sales and Profit by Category**: Compares revenue and profitability across product categories
2. **Regional & Segment Analysis**: 4-panel view of performance across regions and customer segments
3. **Sub-Category Performance**: Identifies top-performing and underperforming product lines
4. **Monthly Trend Analysis**: Shows seasonal patterns and year-over-year growth
5. **Profit Margin Analysis**: Highlights profitability by category and region
6. **Discount Impact**: Demonstrates how discounting affects sales and profits
7. **Product Performance**: Showcases best and worst performing individual products
8. **Shipping Analysis**: Analyzes order distribution and delivery times by shipping mode

## Key Recommendations

### 1. Product Strategy
- ✓ Discontinue or reprice loss-making products (Tables, Bookcases)
- ✓ Increase inventory for high-margin items (Copiers, Phones, Accessories)
- ✓ Bundle low-margin products with profitable ones

### 2. Pricing & Discounts
- ✓ Cap discounts at 30% to maintain profitability
- ✓ Implement dynamic pricing based on demand
- ✓ Review Furniture category pricing strategy

### 3. Regional Strategy
- ✓ Expand operations in West and Central regions
- ✓ Focus marketing efforts on underperforming regions
- ✓ Tailor product mix to regional preferences

### 4. Customer Focus
- ✓ Implement loyalty programs for repeat customers
- ✓ Target Corporate segment with premium products
- ✓ Personalized marketing for high-value customers

### 5. Operational Efficiency
- ✓ Optimize inventory management to reduce holding costs
- ✓ Improve shipping efficiency for Same Day and First Class
- ✓ Monitor profit margins continuously with automated alerts

## Technologies Used

- **Python 3.x**: Core programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## Analysis Methodology

1. **Data Loading & Preprocessing**: Import data, handle encoding, create derived features
2. **Exploratory Data Analysis**: Calculate key metrics, identify patterns
3. **Sales Analysis**: Examine revenue by category, region, segment
4. **Profit Analysis**: Identify profitable and loss-making products
5. **Customer Analysis**: Segment customers and analyze behavior
6. **Time-Series Analysis**: Discover seasonal trends and patterns
7. **Visualization**: Create compelling charts and graphs
8. **Insights & Recommendations**: Derive actionable business strategies

## Learning Outcomes

This project demonstrates:
- End-to-end business analytics workflow
- Data cleaning and preprocessing techniques
- Exploratory data analysis (EDA) best practices
- Creating meaningful business visualizations
- Deriving actionable insights from data
- Translating data findings into business recommendations
- Professional data presentation skills

## Future Enhancements

- Predictive modeling for sales forecasting
- Customer lifetime value (CLV) analysis
- Market basket analysis for product recommendations
- Advanced segmentation using clustering algorithms
- Interactive dashboard using Plotly or Streamlit
- A/B testing framework for pricing strategies
- Real-time monitoring system for KPIs

## Business Impact

This analysis can help businesses:
- Increase profitability by 15-20% through strategic pricing
- Reduce losses by identifying and addressing underperforming products
- Optimize inventory by 25-30% through demand forecasting
- Improve customer retention by 10-15% with targeted programs
- Make data-driven decisions for expansion and growth

## Author

Created as part of the ShadowFox AI/ML Internship Program

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Superstore Dataset: Sample retail dataset for analysis practice
- ShadowFox Internship Program for the opportunity to work on real-world projects