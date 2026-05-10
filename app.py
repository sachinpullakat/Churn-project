import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from predict import predict

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2E86AB;
    }
    .dashboard-bg {
        background-color: black;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .dashboard-bg .stMetric {
        background-color: black;
        color: white;
    }
    .dashboard-bg h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("dataset.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop(columns=[col for col in df.columns if "id" in col.lower()], errors='ignore')
    return df

# File uploader
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    st.sidebar.info("Ensure your CSV has columns like: gender, Partner, Dependents, tenure, PhoneService, etc., and a 'Churn' target column.")
else:
    st.sidebar.info("Using default dataset.csv")

df = load_data(uploaded_file)

# Sidebar filters
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect("Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
gender_filter = st.sidebar.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
internet_filter = st.sidebar.multiselect("Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())

# Apply filters
filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['gender'].isin(gender_filter)) &
    (df['InternetService'].isin(internet_filter))
]

st.markdown("""
# 📈 Customer Churn Predictor

Use this predictive app to evaluate whether a customer is likely to churn. Enter the customer's details and get an instant recommendation.
""")

with st.expander("Why this app?"):
    st.write(
        "This interface uses your trained churn model to make fast predictions and provide actionable suggestions to reduce customer turnover."
    )

st.markdown("---")

st.markdown('<div class="dashboard-bg">', unsafe_allow_html=True)
st.subheader("Dashboard overview")

churn_counts = filtered_df['Churn'].value_counts()
contract_counts = filtered_df['Contract'].value_counts()
service_counts = filtered_df['InternetService'].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(filtered_df):,}")
col2.metric("Churn rate", f"{(churn_counts.get('Yes', 0) / len(filtered_df) * 100):.1f}%" if len(filtered_df) > 0 else "0%")
col3.metric("Avg tenure", f"{filtered_df['tenure'].mean():.1f} months" if len(filtered_df) > 0 else "0 months")
col4.metric("Avg monthly charge", f"${filtered_df['MonthlyCharges'].mean():.2f}" if len(filtered_df) > 0 else "$0")

with st.container():
    chart1, chart2, chart3 = st.columns(3)

    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=["#ff6b6b", "#4CAF50"],
        startangle=140,
    )
    ax1.set_title("Churn distribution")
    chart1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.pie(
        contract_counts,
        labels=contract_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax2.set_title("Contract types")
    chart2.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.pie(
        service_counts,
        labels=service_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax3.set_title("Internet service mix")
    chart3.pyplot(fig3)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.subheader("Churn insights")

churn_by_contract = (
    filtered_df.groupby(['Contract', 'Churn'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

st.markdown("**Churn by contract type**")
fig4, ax4 = plt.subplots(figsize=(8, 4))
churn_by_contract.set_index('Contract').plot(kind='bar', stacked=True, ax=ax4, color=['#4CAF50', '#ff6b6b'])
ax4.set_ylabel('Customers')
ax4.set_title('Churn by contract')
ax4.legend(title='Churn')
ax4.grid(axis='y', linestyle='--', alpha=0.4)
st.pyplot(fig4)

st.markdown("**Monthly charges and tenure distribution**")
charge_col1, charge_col2 = st.columns(2)

fig5, ax5 = plt.subplots(figsize=(6, 4))
ax5.hist(filtered_df['MonthlyCharges'], bins=25, color="#4daf4a", edgecolor="black")
ax5.set_title('Monthly charges distribution')
ax5.set_xlabel('Monthly Charges')
ax5.set_ylabel('Customers')
charge_col1.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(6, 4))
ax6.hist(filtered_df['tenure'], bins=25, color="#377eb8", edgecolor="black")
ax6.set_title('Tenure distribution')
ax6.set_xlabel('Tenure (months)')
ax6.set_ylabel('Customers')
charge_col2.pyplot(fig6)

st.markdown("---")

st.subheader("Interactive correlations")

st.markdown("**Correlation heatmap**")
numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
corr_df = filtered_df[numeric_cols].corr()

fig7, ax7 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax7, fmt='.2f')
ax7.set_title('Feature Correlations')
st.pyplot(fig7)

st.markdown("---")

with st.expander("Customer profile predictor"):
    with st.form("predict_form"):
        st.subheader("Customer profile")

        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0,
            format="%.2f",
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=850.0,
            step=1.0,
            format="%.2f",
        )

        gender = st.selectbox("Gender", ["Female", "Male"])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])

        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        OnlineSecurity = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )
        OnlineBackup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )
        DeviceProtection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )
        TechSupport = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )
        StreamingTV = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )
        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

        Contract = st.selectbox(
            "Contract type",
            ["Month-to-month", "One year", "Two year"],
        )
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        submitted = st.form_submit_button("Predict churn")

if submitted:
    customer_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
    }

    churn_prediction, suggestions = predict(customer_data)

    if churn_prediction == "YES":
        st.markdown("### 🔔 Churn Risk: **High**")
        st.error("This customer is likely to churn.")
    else:
        st.markdown("### ✅ Churn Risk: **Low**")
        st.success("This customer is likely to stay.")

    st.markdown("### Suggested actions")
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    st.divider()
    st.write(
        "Use the section above to test different customer profiles and see which features may improve retention."
    )

st.markdown("---")

with st.expander("How it works"):
    st.write(
        "This app converts form inputs into the same feature format the model expects, then loads your trained model from `model.pkl` and predicts whether the customer will churn. Suggestions are generated from the numeric fields in the input."
    )
