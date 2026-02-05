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

## 1. Data Preprocessing 

Before modeling, the dataset was reviewed and prepared to ensure data quality and consistency.

Key preprocessing steps included:
- Handling missing values in key behavioral and transactional features.
- Standardizing categorical values with similar meanings (e.g., payment methods).


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

- Missing values are primarily concentrated in **behavioral and transactional features**, including:
  - Customer tenure
  - Engagement time
  - Order activity
  - Purchase recency

- These features are **directly related to the customer lifecycle and purchasing behavior**, making proper handling of missing values **critical for accurate churn modeling**.

📌 Columns with Missing Values
- **Tenure**
- **WarehouseToHome**
- **HourSpendOnApp**
- **OrderAmountHikeFromLastYear**
- **CouponUsed**
- **OrderCount**
- **DaySinceLastOrder**

**Handling Missing Values** 

[In 3]
```python
# Fill missing values with median (robust to outliers)
for col in missing_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Verify that there are no missing values left
df[missing_cols].isnull().sum()
```
### 🧠 Rationale for Missing Value Treatment

- **Median imputation** was applied to preserve the original data distribution and minimize sensitivity to outliers.
- This approach helps maintain **churn-related behavioral patterns** while ensuring **model compatibility and stability**.

**📝 Checked for Duplicates**  
[In 4]
```python
# Check for duplicate rows
check_dup = df.duplicated().sum()
```
No duplicate customer records were found, confirming that each row represents a unique customer.

**Cleaning & Standardizing Categorical Values** 

**Standardize categorical values for consistency**

[In 5]

```python
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD': 'Cash on Delivery',
    'CC': 'Credit Card'
})
```
Standardizing categorical values prevents fragmented categories that could distort encoding and downstream model interpretation.

## 2. EDA

**Churn Distribution**
[In 6]
```python
sns.countplot(x='Churn', data=df)
```

[Out 6]

<img width="581" height="427" alt="image" src="https://github.com/user-attachments/assets/1cf6e1bc-6414-4aab-84ae-410eeff5b422" />

**Customer Lifecycle: Tenure vs Churn**

<img width="480" height="460" alt="image" src="https://github.com/user-attachments/assets/e18620b6-12e5-4456-9ff8-5476376af213" />

**Engagement Recency: Days Since Last Order**

<img width="557" height="432" alt="image" src="https://github.com/user-attachments/assets/83ef973d-0e1c-465f-ba58-5f500baeec83" />

**Service Experience: Complaints vs Churn**

```python
# Visualize churn rate by complaint status

complain_churn_table.plot(
    kind='bar',
    figsize=(8, 5),
    color=['steelblue', 'salmon']
)

plt.title('Churn Rate by Complaint Status')
plt.xlabel('Complain (0 = No, 1 = Yes)')
plt.ylabel('Percentage (%)')
plt.legend(['Non-Churn', 'Churn'])
plt.xticks(rotation=0)
plt.show()
```

<img width="686" height="471" alt="image" src="https://github.com/user-attachments/assets/afb131ed-cd53-463a-84b6-e9d9dc85dc8b" />

## 📊 Key Insights & Business Implications

| Analysis Area | Insight | Business Implication |
|--------------|--------|----------------------|
| **Churn Distribution** | The dataset exhibits class imbalance, with churned customers representing a smaller proportion of the overall customer base. | Although churn is a minority event, it carries high business impact. Models should prioritize **recall** to reduce the risk of missing at-risk customers and enable proactive retention actions. |
| **Customer Lifecycle: Tenure vs Churn** | Churned customers tend to have significantly shorter tenure, indicating early disengagement. | Improving **early-stage onboarding and engagement** is critical to reducing churn risk. |
| **Engagement Recency: Days Since Last Order** | Churned customers typically have longer gaps since their last order, signaling declining engagement before churn occurs. | **Recency metrics** can be used as early warning signals to enable proactive customer retention actions. |
| **Service Experience: Complaints vs Churn** | Customers who submitted complaints show substantially higher churn rates compared to those without complaints. | **Service recovery and effective complaint resolution** play a critical role in customer retention. |

These insights highlight that churn is most likely to occur early in the customer lifecycle and is strongly influenced by service experience, making early engagement and complaint resolution the most actionable retention levers.

## 🔮 Churn Prediction – Identifying At-Risk Customers

The objective of this step is to determine whether **customer churn can be predicted in advance** using **behavioral, service, and logistics-related features**, enabling **early intervention and retention planning**.

Rather than optimizing for overall accuracy, the modeling approach **prioritizes recall**, as failing to identify **churned customers** carries a **higher business cost** than false positives.

**Feature Selection & Target Definition**
```python
# Define features and target

y = df['Churn']
X = df.drop(columns=['CustomerID', 'Churn'])

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
```
### 🧠 Rationale

- Input features capture signals related to **customer lifecycle**, **engagement**, **service experience**, and **logistics**.
- The target variable (**Churn**) represents whether a customer has **left the platform**.

**Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```
### 🧠 Rationale

- **Stratified splitting** preserves the original **churn ratio** in both the training and testing sets.
- This ensures **reliable model evaluation** under **class imbalance**.

**Baseline Model: Logistic Regression**
```python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print(classification_report(y_test, y_pred_lr))
```

<img width="457" height="162" alt="image" src="https://github.com/user-attachments/assets/d138f2c1-97ae-4cba-8915-08a993af014f" />

### 💡 Insight

- The baseline model confirms that **customer churn can be predicted** using the available customer data.
- However, **recall for churned customers is limited**, motivating the use of a **more flexible model**.

## 🌲 Final Model: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(classification_report(y_test, y_pred_rf))
```

<img width="430" height="172" alt="image" src="https://github.com/user-attachments/assets/8171721b-f236-43b5-be08-82c85cc9ba6d" />


### 💡 Insight

- The **Random Forest model** improves **recall for churned customers** compared to the baseline model.
- This makes it more suitable for **identifying at-risk customers** in a **business setting**.

# 📊 Model Evaluation Summary

### 🔑 Key Takeaways

- Customer churn can be predicted with **meaningful reliability** using **behavioral, service, and logistics data**.
- **Prioritizing recall** helps minimize missed churned customers, supporting **proactive retention efforts**.
- **Tree-based models** better capture **non-linear customer behavior patterns** compared to linear baselines.

## 🏢 Business Interpretation

From a **business and supply chain perspective**, churn prediction enables:

- **Early identification** of potential demand loss
- **Targeted retention actions** before customers disengage completely
- **Improved planning** for customer support and operational resources

The model serves as a **decision-support tool**, not an **automated decision-maker**.

## 🔑 Key Churn Drivers & Model Interpretation

This section focuses on explaining **why customers churn**, using **model interpretation** to translate predictive results into **actionable business and operational insights**.

Understanding churn drivers is critical to ensure that predictive models support **decision-making**, not just prediction.

```python
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf_model.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
```
<img width="882" height="457" alt="image" src="https://github.com/user-attachments/assets/555ded13-f5b8-46a3-ae8c-f755a979584e" />

### 🔑 Key Drivers Identified

The most influential churn drivers include:

- **Tenure**
- **DaySinceLastOrder**
- **CashbackAmount**
- **Complain**
- **WarehouseToHome**

These features capture a combination of **customer lifecycle**, **engagement behavior**, **service quality**, and **logistics experience**.

### 💡 Insights

**Customer lifecycle matters:**  
Customers with **shorter tenure** and **longer gaps since their last order** are significantly more likely to churn, indicating **early disengagement patterns**.

**Service experience is critical:**  
Customers who submitted **complaints** show a **higher churn risk**, emphasizing the importance of effective **service recovery**.

**Logistics plays a supporting role:**  
Greater **warehouse-to-home distance** correlates with higher churn risk, suggesting that **delivery experience** influences customer retention.

**Incentives alone are insufficient:**  
While **cashback** impacts retention, it cannot fully compensate for **poor engagement** or **service issues**.

## 🏭 Business & Supply Chain Implications

From an **operational and supply chain perspective**:

- Churn is not driven by a single factor, but by the **interaction of engagement, service, and logistics performance**.
- Monitoring **lifecycle and recency metrics** enables **early detection of potential demand loss**.
- **Service quality** and **last-mile delivery experience** should be integrated into **retention planning**, not treated separately from marketing initiatives.

These insights directly inform **targeted retention strategies** and **operational improvement initiatives**.


## 7. 👥 Customer Segmentation – Understanding Different Types of Churned Customers

While **churn prediction** identifies **who is at risk**, **segmentation** helps explain **how churned customers differ from one another**.

This step focuses on **segmenting churned customers only** to support **targeted retention strategies** and **operational actions**.
```python
cluster_features = [
    'Tenure',
    'HourSpendOnApp',
    'CashbackAmount',
    'WarehouseToHome',
    'DaySinceLastOrder'
]

X_cluster = df_churned[cluster_features]

X_cluster.head()
```
### 🧠 Rationale

- Segmentation focuses on **customers who have already churned**.
- Selected features represent **engagement**, **lifecycle**, **incentives**, and **logistics experience**.

**Determining the Number of Clusters (Elbow Method)**

```python
inertia = []

K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for KMeans')
plt.show()
```
<img width="547" height="393" alt="image" src="https://github.com/user-attachments/assets/93429b32-e692-46f3-8492-b49e796f1fe0" />

The elbow point suggests k = 3, balancing simplicity and interpretability.

**K-Means Clustering**
```python
kmeans = KMeans(n_clusters=3, random_state=42)
df_churned['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

df_churned['Cluster'].value_counts()
```
<img width="122" height="127" alt="image" src="https://github.com/user-attachments/assets/9eb335b2-fe98-4d3a-9ff6-6476b50b30f9" />


**Cluster Profiling**

```python
cluster_summary = (
    df_churned
    .groupby('Cluster')[cluster_features]
    .mean()
    .round(2)
)

cluster_summary
```
<img width="657" height="155" alt="image" src="https://github.com/user-attachments/assets/23f2789f-ad9e-4ea4-8202-949d62dbf1d1" />

### 🧩 Segment Interpretation

Based on cluster profiling, **three distinct churn segments** emerge:

---

#### 🔹 Cluster 0 – Early Drop-off Customers
- **Short tenure** and **low engagement**
- Churn shortly after onboarding  

**Interpretation:**  
Customers fail to build initial engagement or perceive early value.

---

#### 🔹 Cluster 1 – Disengaged Long-Tenure Customers
- **Longer tenure** with **increasing inactivity**
- Higher **days since last order**  

**Interpretation:**  
Previously engaged customers gradually disengage, often due to **declining experience** or **unresolved issues**.

---

#### 🔹 Cluster 2 – Logistics-Sensitive Customers
- Higher **warehouse-to-home distance**
- Moderate engagement but higher churn risk tied to **fulfillment experience**  

**Interpretation:**  
Churn is influenced more by **delivery experience and logistics constraints** than by engagement alone.

---

### 🏢 Business & Operational Implications

**Early Drop-off Customers**
- Improve **onboarding**, **early incentives**, and **first-order experience**

**Disengaged Long-Tenure Customers**
- Prioritize **re-engagement campaigns** and **proactive service outreach**

**Logistics-Sensitive Customers**
- Review **last-mile delivery strategies** and set **clearer delivery expectations**

---

Segmentation highlights that **churn is not a single behavior**, and effective retention strategies must be **segment-specific**.

## 8. 📈 Business Recommendations & Strategic Takeaways

This section consolidates insights from **exploratory analysis**, **churn prediction**, and **customer segmentation** into **actionable business and operational recommendations**.

The focus is on translating analytics into **decisions that reduce churn**, **stabilize demand**, and **improve customer experience**.

---

### 🧾 Summary of Key Findings

- Customer churn is strongly influenced by **customer lifecycle and engagement patterns**, particularly **tenure** and **recency**.
- **Service experience**, especially **customer complaints**, is a major driver of churn.
- **Logistics factors**, such as **warehouse-to-home distance**, contribute to churn risk and should not be overlooked.
- Churned customers are **not homogeneous** and can be grouped into **distinct behavioral segments**.

---

### 🧠 Strategic Business Recommendations

#### 1️⃣ Strengthen Early Customer Engagement
- Target customers with **short tenure** and **declining engagement** through onboarding improvements and early engagement campaigns.
- Improve **first-order experience** to reduce early-stage churn.

**Business Impact:**  
Reduces early customer drop-off and improves **customer lifetime value**.

---

#### 2️⃣ Use Churn Risk Signals for Proactive Retention
- Leverage churn prediction outputs to identify **high-risk customers** before disengagement becomes irreversible.
- **Prioritize recall** to minimize missed churned customers.

**Business Impact:**  
Enables **timely intervention** and reduces unexpected **demand loss**.

---

#### 3️⃣ Improve Service Recovery & Complaint Resolution
- Monitor **complaint activity** as a leading churn indicator.
- Implement **faster resolution** and **follow-up** for customers who submit complaints.

**Business Impact:**  
Builds **customer trust** and prevents **service-related churn**.

---

#### 4️⃣ Integrate Logistics Performance into Retention Strategy
- Analyze **delivery distance** and **fulfillment performance** for customers located farther from warehouses.
- Set **clearer delivery expectations** or optimize **last-mile delivery options** for logistics-sensitive customers.

**Business Impact:**  
Improves **customer experience** and reduces churn driven by **fulfillment issues**.

---

#### 5️⃣ Apply Segment-Specific Retention Actions
- **Early Drop-off Customers:** Focus on onboarding and early incentives.
- **Disengaged Long-Tenure Customers:** Use re-engagement campaigns and personalized outreach.
- **Logistics-Sensitive Customers:** Address delivery experience and operational constraints.

**Business Impact:**  
Improves efficiency by aligning retention actions with **customer behavior patterns**.

---

### 🎯 Strategic Takeaways

- Churn should be viewed as a **cross-functional problem**, not solely a marketing issue.
- Combining **predictive analytics** with **operational insights** enables more effective decision-making.
- Integrating churn insights into **planning processes** supports **demand stability**, **service capacity planning**, and **customer experience improvement**.


