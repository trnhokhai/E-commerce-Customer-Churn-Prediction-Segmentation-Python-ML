# E-commerce-Customer-Churn-Prediction-Segmentation-Python-ML
Build a churn model and segment churned users to support targeted retention offers and operational improvements.

## What this project is about

Customer churn is a critical challenge for e-commerce businesses, as losing existing customers directly impacts revenue and increases acquisition costs.

In this project, I analyzed customer behavior data to understand why customers churn, built a machine learning model to predict churn risk, and explored customer segmentation to support retention strategies.

Key outcomes from this project include:
- Identifying the main behavioral and operational drivers of churn, such as short tenure, low engagement, customer complaints, and delivery distance.
- Building a churn prediction model that prioritizes recall to minimize missed churned customers.
- Translating analytical insights into actionable business and operational recommendations to reduce churn.

## Industry Context

In the e-commerce industry, customer churn is a common challenge due to intense competition, low switching costs, and high customer expectations around pricing, delivery speed, and service quality.
Retaining existing customers is often more cost-effective than acquiring new ones, making churn analysis a key priority for growth and profitability.

## 📌 Background & Overview

This project focuses on analyzing customer-level behavioral and operational data to understand churn patterns in an e-commerce business.

By combining exploratory analysis and machine learning, the project aims to move beyond descriptive reporting and support data-driven retention decisions, such as identifying high-risk customers and prioritizing intervention efforts.

### **🎯 Objective**
The objective of this project is to:
- Identify key behavioral and operational factors that contribute to customer churn.
- Build a predictive model to identify customers with high churn risk.
- Support retention strategies by translating analytical findings into actionable business recommendations.

### **❓Business Question**
- What behavioral and operational patterns distinguish churned customers from retained customers?
- Which factors have the strongest impact on customer churn risk?
- How can churn prediction be used to identify high-risk customers for early intervention?
- Can churned customers be meaningfully segmented to support targeted retention offers and promotions?

### **👤 Who Is This Project For?**
- Marketing & Customer Retention Teams  
  → To design targeted promotions and re-engagement campaigns for at-risk customers.

- Operations & Logistics Teams  
  → To understand how delivery distance and service issues contribute to churn.

- Business & Data Analysts  
  → To support data-driven decision-making around customer retention strategies.

## 📂 **Dataset Description & Data Structure**

### 📌 **Data Source**  
**Source:** The dataset is obtained from the e-commerce company's database.  
**Size:** The dataset contains 5,630 rows and 20 columns.  
**Format:** .xlxs file format.

### 📊 **Data Structure & Relationships**

1️⃣ **Tables Used:**  
The dataset contains only **1 table** with customer and transaction-related data.

2️⃣ **Table Schema & Data Snapshot**  
**Table: Customer Churn Data**

<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>

## Data Preprocessing & Exploratory Analysis (Selected)

Before modeling, the dataset was reviewed and prepared to ensure data quality and consistency.

Key preprocessing steps included:
- Handling missing values in key behavioral and transactional features.
- Standardizing categorical values with similar meanings (e.g., payment methods).
- Dropping identifier columns that do not contribute to prediction.

[In 1]
```python
df.head()
```
[Out 1]
<img width="1387" height="221" alt="image" src="https://github.com/user-attachments/assets/4c23db61-2215-47b3-9c71-7af72eafdba1" />

**📝 Checked for Missing Values**  
[In 2]
```python
# Check missing values in each column
df.isnull().sum()
```
The columns with missing values are:

   - `Tenure` - 264 missing values
   - `WarehouseToHome` - 251 missing values
   - `HourSpendOnApp` - 255 missing values
   - `OrderAmountHikeFromlastYear` - 265 missing values
   - `CouponUsed` - 256 missing values
   - `OrderCount` - 258 missing values
   - `DaySinceLastOrder` - 307 missing values

**Handling Missing Values** 
[In 3]
```python
# Fill missing values with median (robust to outliers)
for col in missing_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Verify that there are no missing values left
df[missing_cols].isnull().sum()
```
Missing values were concentrated in behavioral features related to engagement and purchasing activity.
Median imputation was applied to preserve distribution shape and avoid skewing churn-related patterns.

**📝 Checked for Duplicates**  
[In 4]
```python
# Check for duplicate rows
check_dup = df.duplicated().sum()
```
Aftering checkeing for duplicate rows in the dataset and found that there were no duplicate entries.

**Cleaning & Standardizing Categorical Values** 

# Standardize categorical values for consistency
[In 5]
```python
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD': 'Cash on Delivery',
    'CC': 'Credit Card'
})
```
Standardizing categorical values prevents fragmented categories that could distort encoding and downstream model interpretation.

**EDA Highlight: Churn Distribution**
[In 6]
```python
sns.countplot(x='Churn', data=df)
```
[Out 6]
<img width="581" height="427" alt="image" src="https://github.com/user-attachments/assets/1cf6e1bc-6414-4aab-84ae-410eeff5b422" />


After preprocessing, the dataset is clean, consistent, and suitable for churn modeling.
Key behavioral and operational variables are preserved, allowing downstream models to focus on meaningful churn drivers rather than data quality issues.
