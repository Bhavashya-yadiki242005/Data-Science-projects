import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📉 Bankruptcy Prediction System")
st.write("Upload a CSV file with financial details to assess bankruptcy risk.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop the 'class' column if it exists
    if "class" in df.columns:
        df = df.drop(columns=["class"])

    st.write("🔍 **Preview of Uploaded Data:**")
    st.dataframe(df.head())

    if st.button("⚡ Predict"):
        try:
            input_array = scaler.transform(df)  # Transform input
            predictions = model.predict(input_array)
            risk_scores = model.predict_proba(input_array)[:, 1] * 100

            # Add predictions to DataFrame
            df["Prediction"] = predictions
            df["Risk Score (%)"] = risk_scores

            # Categorize risk levels
            df["Risk Category"] = pd.cut(df["Risk Score (%)"], bins=[0, 30, 70, 100], 
                                         labels=["Low Risk", "Medium Risk", "High Risk"])

            # Display results
            st.write("📊 **Prediction Results:**")
            st.dataframe(df)

            # Summary Statistics
            low_risk_count = (df["Risk Category"] == "Low Risk").sum()
            medium_risk_count = (df["Risk Category"] == "Medium Risk").sum()
            high_risk_count = (df["Risk Category"] == "High Risk").sum()

            st.subheader("📌 Bankruptcy Risk Analysis")
            st.write(f"✅ **Low Risk Cases:** {low_risk_count}")
            st.write(f"⚠️ **Medium Risk Cases:** {medium_risk_count}")
            st.write(f"🚨 **High Risk Cases:** {high_risk_count}")

            # Show risk distribution
            st.subheader("📊 Risk Score Distribution")
            fig, ax = plt.subplots()
            df["Risk Score (%)"].hist(bins=20, edgecolor='black', ax=ax)
            plt.xlabel("Risk Score (%)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Risk Scores")
            st.pyplot(fig)

            # Pie Chart of Risk Categories
            st.subheader("📊 Risk Category Breakdown")
            fig, ax = plt.subplots()
            df["Risk Category"].value_counts().plot.pie(autopct='%1.1f%%', colors=["green", "orange", "red"], ax=ax)
            plt.title("Proportion of Risk Categories")
            st.pyplot(fig)

            # Recommendations
            st.subheader("📌 Recommendations")
            if high_risk_count > 0:
                st.warning("🚨 **High Risk Detected!** Consider financial restructuring and cost-cutting measures.")
                st.write("""
                - **Review financial statements** for areas of concern.  
                - **Increase revenue sources** through diversification.  
                - **Consult financial advisors** for restructuring options.  
                """)
            elif medium_risk_count > 0:
                st.warning("⚠️ **Medium Risk Detected!** Monitor finances closely and take preventive actions.")
                st.write("""
                - **Reduce operational costs** where possible.  
                - **Improve cash flow management.**  
                - **Maintain a financial buffer** to avoid risk escalation.  
                """)
            else:
                st.success("🎉 **No High Bankruptcy Risk Detected!** Your financial status seems stable.")
                st.write("""
                - Continue with **strong financial planning.**  
                - Keep track of **market trends** for future stability.  
                - **Monitor expenses** regularly to maintain a healthy financial position.  
                """)

        except Exception as e:
            st.error(f"❌ Error: {e}")

