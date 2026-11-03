import pandas as pd
import numpy as np

def generate_customer_behaviour_dataset(
    input_path="online_shoppers_intention.csv",
    output_path="customer_behaviour.csv",
    seed=42
):
    """
    Transforms and enriches the online shoppers intention dataset into the 
    standard Customer Purchase Behaviour dataset containing all required attributes:
    - Customer age group
    - Browsing activity
    - Past purchases
    - Product views
    - Session duration
    - Customer segment
    - Actual purchase (target)
    Also injects controlled missing values and duplicates to allow PySpark 
    imputation and duplicate cleaning demonstrations for Task 163.
    """
    np.random.seed(seed)
    df_raw = pd.read_csv(input_path)
    n = len(df_raw)
    
    print(f"Loaded raw dataset with {n} rows and {len(df_raw.columns)} columns.")

    # 1. Customer ID
    customer_ids = [f"CUST-{10000 + i}" for i in range(n)]

    # 2. Product Views & Session Duration
    product_views = df_raw["ProductRelated"].astype(int)
    session_duration = df_raw["ProductRelated_Duration"].round(2)
    # Ensure non-negative duration
    session_duration = np.maximum(session_duration, 0.0)

    # 3. Browsing Activity (based on Administrative + Informational page views)
    total_browsing_pages = df_raw["Administrative"] + df_raw["Informational"]
    browsing_activity = []
    for pages in total_browsing_pages:
        if pages <= 1:
            browsing_activity.append("Low")
        elif pages <= 5:
            browsing_activity.append("Moderate")
        else:
            browsing_activity.append("High")

    # 4. Past Purchases (correlated with VisitorType and PageValues)
    past_purchases = []
    for vtype, pval in zip(df_raw["VisitorType"], df_raw["PageValues"]):
        if vtype == "New_Visitor":
            past_purchases.append(0)
        else:
            # Returning visitor: base purchases 1-12
            base = np.random.poisson(lam=2.5) + (1 if pval > 0 else 0)
            past_purchases.append(int(min(max(1, base), 15)))

    # 5. Customer Age Group (realistic e-commerce demographic distribution)
    # '18-25', '26-35', '36-50', '51+'
    age_groups_pool = ["18-25", "26-35", "36-50", "51+"]
    age_probs = [0.22, 0.43, 0.23, 0.12]
    customer_age_group = np.random.choice(age_groups_pool, size=n, p=age_probs)

    # 6. Customer Segment (based on engagement, past purchases, and intent)
    customer_segments = []
    for vtype, pp, pv, pval in zip(df_raw["VisitorType"], past_purchases, product_views, df_raw["PageValues"]):
        if vtype == "New_Visitor" and pval > 0:
            customer_segments.append("High-Intent Shopper")
        elif vtype == "New_Visitor":
            customer_segments.append("New Visitor")
        elif pp >= 4 and pval > 0:
            customer_segments.append("Loyal Customer")
        elif pv > 30 and pval == 0:
            customer_segments.append("Window Shopper")
        elif pval > 15:
            customer_segments.append("High-Intent Shopper")
        else:
            customer_segments.append("Bargain Hunter")

    # 7. Actual Purchase (target variable, 1 or 0)
    actual_purchase = df_raw["Revenue"].apply(lambda x: 1 if str(x).strip().upper() in ["TRUE", "1"] else 0)

    # Construct unified customer behavior DataFrame
    df_clean = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_age_group": customer_age_group,
        "browsing_activity": browsing_activity,
        "past_purchases": past_purchases,
        "product_views": product_views,
        "session_duration": session_duration,
        "bounce_rate": df_raw["BounceRates"].round(4),
        "exit_rate": df_raw["ExitRates"].round(4),
        "page_values": df_raw["PageValues"].round(2),
        "customer_segment": customer_segments,
        "visitor_type": df_raw["VisitorType"].fillna("Returning_Visitor"),
        "month": df_raw["Month"].fillna("May"),
        "weekend": df_raw["Weekend"].apply(lambda x: 1 if str(x).strip().upper() in ["TRUE", "1"] else 0),
        "purchase": actual_purchase
    })

    print(f"Derived features successfully. Purchase distribution:\n{df_clean['purchase'].value_counts()}")

    # 8. Controlled injection of missing values and duplicate rows (Task 163 requirement)
    df_with_nulls = df_clean.copy()

    # Inject ~1.5% nulls into session_duration
    nan_mask_duration = np.random.rand(n) < 0.015
    df_with_nulls.loc[nan_mask_duration, "session_duration"] = np.nan

    # Inject ~1.2% nulls into product_views
    nan_mask_views = np.random.rand(n) < 0.012
    df_with_nulls.loc[nan_mask_views, "product_views"] = np.nan

    # Inject ~1.0% nulls into customer_age_group
    nan_mask_age = np.random.rand(n) < 0.010
    df_with_nulls.loc[nan_mask_age, "customer_age_group"] = np.nan

    # Inject ~1.0% nulls into browsing_activity
    nan_mask_browse = np.random.rand(n) < 0.010
    df_with_nulls.loc[nan_mask_browse, "browsing_activity"] = np.nan

    # Add 30 duplicate records
    duplicates = df_with_nulls.iloc[:30].copy()
    df_final = pd.concat([df_with_nulls, duplicates], ignore_index=True)

    df_final.to_csv(output_path, index=False)
    print(f"Generated '{output_path}' with {len(df_final)} rows.")
    print("Null value summary:")
    print(df_final.isnull().sum())
    return df_final

if __name__ == "__main__":
    generate_customer_behaviour_dataset()
