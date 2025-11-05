import streamlit as st
from rag_agent import query_agent
import pandas as pd
import matplotlib.pyplot as plt

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Business Analyst Dashboard",
    layout="wide",
)

# ==================== CUSTOM STYLE ====================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0f7fa, #ede7f6);
    font-family: 'Poppins', sans-serif;
}
.stButton>button {
    background-color: #6a1b9a;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: 500;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #4a148c;
    transform: scale(1.05);
}
h1, h2, h3 {
    color: #4a148c;
}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.title("AI Business Analyst Dashboard")
st.markdown("Your intelligent assistant for analyzing sales, expenses, and marketing trends.")

# ==================== FILE UPLOAD ====================
uploaded = st.file_uploader(" Upload a CSV file (optional)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(" New dataset uploaded successfully!")
else:
    df = pd.read_csv("data/business_data.csv")

# ==================== SIDEBAR METRICS ====================
with st.sidebar:
    st.header(" Business Summary")

    # Basic metrics
    df["Profit"] = df["Sales (INR)"] - df["Expenses (INR)"]
    avg_sales = df["Sales (INR)"].mean()
    avg_expenses = df["Expenses (INR)"].mean()
    avg_profit = df["Profit"].mean()
    roi = ((df["Profit"] / df["Marketing Spend (INR)"]) * 100).mean()
    best_month = df.loc[df["Profit"].idxmax(), "Month"]

    st.metric(" Average Sales (₹)", f"{avg_sales:,.0f}")
    st.metric(" Average Expenses (₹)", f"{avg_expenses:,.0f}")
    st.metric(" Average Profit (₹)", f"{avg_profit:,.0f}")
    st.metric(" ROI (%)", f"{roi:.2f}")
    st.metric(" Best Month", best_month)

    st.markdown("---")
    st.info(" Tip: Try asking questions like *'Show sales trends'* or *'Which month had highest profit?'*")

# ==================== AUTO SUMMARY ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if len(st.session_state.chat_history) == 0:
    st.markdown("###  Quick Overview")
    st.success(
        f"Your dataset covers **{len(df)} months**. "
        f"The highest profit was in **{best_month}**, "
        f"with an average ROI of **{roi:.2f}%**."
    )

# ==================== CHARTS SECTION ====================
st.markdown("## Monthly Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sales vs Expenses")
    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Sales (INR)"], label="Sales", marker="o", color="#6a1b9a")
    ax.plot(df["Month"], df["Expenses (INR)"], label="Expenses", marker="o", color="#ce93d8")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

with col2:
    st.markdown("### Monthly Profit")
    st.bar_chart(df.set_index("Month")[["Profit"]])

st.markdown("---")

# ==================== AI Q&A CHAT ====================
st.header(" Ask the AI Analyst")

query = st.text_input("Enter your business question:")
ask = st.button(" Analyze")

if ask and query:
    with st.spinner("Analyzing your business data..."):
        answer = query_agent(query)
        st.session_state.chat_history.append((query, answer))
        st.success(answer)

    # Auto chart trigger if user asks about trends
    if any(x in query.lower() for x in ["trend", "chart", "plot", "graph", "compare"]):
        st.info(" Here's a visual representation of your data:")
        st.line_chart(df.set_index("Month")[["Sales (INR)", "Expenses (INR)"]])

# ==================== CHAT HISTORY ====================
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("###  Conversation History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"** You:** {q}")
        st.markdown(f"** Analyst:** {a}")
