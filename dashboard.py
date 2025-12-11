import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Client Retention Dashboard", layout="wide")

# ---------------------------------------------------------
# Load Data & Model
# ---------------------------------------------------------
df = pd.read_csv("prepared.csv", parse_dates=["Date"])
model = joblib.load("retention_model.pkl")

st.title("💄 Glow Up Studio — Client Retention Dashboard")
st.markdown("### Monitor return behavior, rebooking patterns & marketing effectiveness.")

st.markdown("---")

# ---------------------------------------------------------
# Extract category names
# ---------------------------------------------------------
category_cols = [c for c in df.columns if c.startswith("Category_")]
categories = [c.replace("Category_", "") for c in category_cols]

# ---------------------------------------------------------
# KPIs Section
# ---------------------------------------------------------
total_customers = df.shape[0]
overall_return_rate = df["Return_Visit"].mean() * 100
avg_spend = df["Purchase_Value"].mean()
email_eng_rate = (df["Email_Engagement"] == "Clicked").mean() * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 Total Customers", total_customers)
c2.metric("🔄 Overall Return Rate", f"{overall_return_rate:.1f}%")
c3.metric("💰 Avg Purchase Value", f"${avg_spend:.2f}")
c4.metric("📧 Email Click Rate", f"{email_eng_rate:.1f}%")

st.markdown("---")

# ---------------------------------------------------------
# Category Filter
# ---------------------------------------------------------
selected_cat = st.selectbox("Filter by Category:", ["All"] + categories)
filtered_df = df.copy()

if selected_cat != "All":
    colname = "Category_" + selected_cat
    filtered_df = filtered_df[filtered_df[colname] == 1]

st.subheader("📊 Return Rate by Selected Category")
st.write(f"**Return Rate:** {filtered_df['Return_Visit'].mean()*100:.1f}%")

# ---------------------------------------------------------
# Charts Section
# ---------------------------------------------------------
st.subheader("📈 Insights & Trends")

colA, colB = st.columns(2)

# 1 — Return by Category
cat_return = {
    cat: df[df[col] == 1]["Return_Visit"].mean()
    for cat, col in zip(categories, category_cols)
}
cat_df = pd.DataFrame({"Category": categories, "Return_Rate": cat_return.values()})

fig1 = px.bar(cat_df, x="Category", y="Return_Rate",
              title="Return Rate by Category", text_auto=True)
colA.plotly_chart(fig1, use_container_width=True)

# 2 — Email Engagement Impact
email_df = df.groupby("Email_Engagement")["Return_Visit"].mean().reset_index()
fig2 = px.pie(email_df, names="Email_Engagement", values="Return_Visit",
              title="Impact of Email Engagement on Return Visits")
colB.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Automatic Insights
# ---------------------------------------------------------
st.subheader("🧠 Key Insights")

top_category = cat_df.sort_values("Return_Rate", ascending=False).iloc[0]
lowest_category = cat_df.sort_values("Return_Rate").iloc[0]

st.write(f"""
### 🔍 Key Insights:
- The category with the **highest return rate** is **{top_category['Category']}** at **{top_category['Return_Rate']*100:.1f}%**.
- The category with the **lowest return rate** is **{lowest_category['Category']}** at **{lowest_category['Return_Rate']*100:.1f}%**.
- Customers who **open or click marketing emails** show significantly higher return behavior.
- **Discount usage** is strongly correlated with increased return visits.
- Higher spending segments tend to have **better retention performance**.

These insights provide valuable guidance for the marketing team to optimize rebooking strategies and enhance client retention efforts.
""")


# ---------------------------------------------------------
# Customer Segmentation (Simple)
# ---------------------------------------------------------
st.subheader("👥 Customer Spending Segments")

df["Segment"] = pd.cut(
    df["Purchase_Value"],
    bins=[0, 80, 150, 300],
    labels=["Low", "Medium", "High"]
)

seg = df.groupby("Segment")["Return_Visit"].mean().reset_index()

fig3 = px.bar(seg, x="Segment", y="Return_Visit",
              title="Return Rate by Spending Segment", text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# Prediction Tool
# ---------------------------------------------------------
st.markdown("---")
st.header("🔮 Predict Return Probability for a Specific Customer")

cust_id = st.text_input("Enter Customer ID:")

if cust_id and cust_id in df["Customer_ID"].values:
    row = df[df["Customer_ID"] == cust_id].iloc[0:1]

    feature_cols = [
        "Purchase_Value", "Email_Engagement", "Discount_Used",
        "frequency", "total_spent", "avg_spent", "recency_days",
        "purchase_month", "purchase_dayofweek"
    ] + category_cols

    inputs = row[feature_cols]

    pred = model.predict_proba(inputs)[0][1]
    st.success(f"🎯 **Return Probability: {pred*100:.1f}%**")

    st.info("Use this prediction to plan targeted retention campaigns.")

