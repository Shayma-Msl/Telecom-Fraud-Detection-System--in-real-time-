import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime
import time
import plotly.express as px
# Cache for SHAP explanations

# PAGE CONFIG
st.set_page_config(page_title="📞 Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
# Custom CSS to style the sidebar navigation
# 🌟 Modern sidebar styling
st.markdown("""
<style>
/* Keep original sidebar background */
section[data-testid="stSidebar"] {
    padding: 2rem 1rem;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1, .css-1d391kg > div > div:nth-child(1) > div > div > div {
    font-size: 24px !important;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
    letter-spacing: 0.5px;
}

/* Radio group vertical layout and spacing */
.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
}

/* Radio labels */
.stRadio div[role="radiogroup"] > label {
    font-size: 18px;
    font-weight: 500;
    background-color: transparent;  /* Transparent to inherit Streamlit bg */
    color: #eee;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    transition: all 0.3s ease-in-out;
}

/* Hover effect */
.stRadio div[role="radiogroup"] > label:hover {
    background-color: #262730;
    transform: scale(1.02);
    border-color: #555;
}

/* Selected radio button */
.stRadio div[role="radiogroup"] > label[data-selected="true"] {
    background: linear-gradient(to right, #0052cc, #0073e6);
    color: #fff !important;
    font-weight: 600;
    border: none;
    box-shadow: 0 0 0 2px #0073e660;
}
</style>
""", unsafe_allow_html=True)



# DATABASE CONNECTION
try:
    conn = mysql.connector.connect(
        host="mysql", user="root", password="root", database="frauddetection1"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM fraud_results ORDER BY detected_at DESC")
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    df["detected_at"] = pd.to_datetime(df["detected_at"])
except Exception as e:
    st.error(f"❌ Database error: {e}")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["📈 Dashboard", "📊 Analytics", "🔎 Fraud Logs & Search", "🛠️ Simulate Calls"])

# === PAGE 1: DASHBOARD ===
if page == "📈 Dashboard":
    st.title("📞 Real-Time Fraud Detection Dashboard")

    # Welcome Section
    st.markdown("""
    <div style="padding: 1rem; border-radius: 8px; background-color: #0e1117;
                 border-left: 6px solid #0066cc; margin-bottom: 1.5rem;">
        <h3 style="color:#ffffff;">👋 Welcome to the Fraud Detection Dashboard</h3>
        <p style="color:#cccccc;">This platform monitors telecom activity in real time to detect Wangiri and IPBX hacking using GAT-COBO and LightGBM models. Navigate the sidebar for analytics, logs, and simulation tools.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    gat_html = """<div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Accuracy</div><div style="font-weight:bold;font-size:20px;color:#fff;">95.78%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">F1 Score</div><div style="font-weight:bold;font-size:20px;color:#fff;">89.00%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Precision</div><div style="font-weight:bold;font-size:20px;color:#fff;">95.00%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Recall</div><div style="font-weight:bold;font-size:20px;color:#fff;">84.61%</div></div>"""

    lgbm_html = """<div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Accuracy</div><div style="font-weight:bold;font-size:20px;color:#fff;">98%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">F1 Score</div><div style="font-weight:bold;font-size:20px;color:#fff;">97%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Precision</div><div style="font-weight:bold;font-size:20px;color:#fff;">98%</div></div>
    <div style="display:inline-block;margin:6px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #444;text-align:center;min-width:140px;"><div style="font-size:14px;color:#bbb;">Recall</div><div style="font-weight:bold;font-size:20px;color:#fff;">99%</div></div>"""

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔷 GAT-COBO")
        st.markdown(gat_html, unsafe_allow_html=True)
    with col2:
        st.markdown("### 🔷 LightGBM")
        st.markdown(lgbm_html, unsafe_allow_html=True)

    # Daily Summary
    st.markdown("### 📊 Today's Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("📞 Total Fraud Records", f"{len(df):,}")
    col2.metric("🚨 Frauds Today", df[df["detected_at"].dt.date == datetime.today().date()].shape[0])
    col3.metric("⏱ Last Fraud Detected", df["detected_at"].max().strftime("%Y-%m-%d %H:%M:%S"))

    # Highlighted Caller
    st.markdown("### 📌 Highlighted Caller")
    fraud_today = df[df["detected_at"].dt.date == datetime.today().date()]
    top_today = fraud_today["CallingNumber"].value_counts().head(1)
    if not top_today.empty:
        number, count = top_today.index[0], top_today.iloc[0]
        st.warning(f"🔔 {number} flagged {count} time(s) today.")
    else:
        st.success("✅ No frauds detected yet today.")

# === PAGE 2: ANALYTICS ===
elif page == "📊 Analytics":
    st.title("📊 Fraud Analytics")

    st.subheader("📋 Total Logs by Fraud Type")
    fraud_type_counts = df["fraud_type"].value_counts().reset_index()
    fraud_type_counts.columns = ["Fraud Type", "Total Logs"]
    st.table(fraud_type_counts)

    
    st.subheader("📊 Fraud Type Distribution")
    st.plotly_chart(px.pie(df, names="fraud_type", title="Fraud Types"))

    st.subheader("📈 Daily Fraud Trends")
    start_date = pd.to_datetime("2024-10-01")
    daily_fraud_df = df[df["detected_at"] >= start_date]
    st.line_chart(daily_fraud_df.groupby(daily_fraud_df["detected_at"].dt.date).size())


    st.subheader("🏆 Top 10 Fraudulent Callers")
    st.bar_chart(df["CallingNumber"].value_counts().head(10))

    if "Callduration" in df.columns:
        st.subheader("⏱ Average Fraud Call Duration")
        fraud_calls = df[df["fraud_type"].notnull()]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", round(fraud_calls["Callduration"].mean(), 2))
        col2.metric("Median", round(fraud_calls["Callduration"].median(), 2))
        col3.metric("Max", round(fraud_calls["Callduration"].max(), 2))
        col4.metric("Min", round(fraud_calls["Callduration"].min(), 2))

    st.subheader("📅 Heatmap of Fraud by Hour/Day")
    df["hour"] = df["detected_at"].dt.hour
    df["day_of_week"] = df["detected_at"].dt.day_name()
    heatmap = df.groupby(["day_of_week", "hour"]).size().unstack().fillna(0)
    heatmap = heatmap.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    st.plotly_chart(px.imshow(heatmap, labels=dict(x="Hour", y="Day", color="Fraud Count"), color_continuous_scale="Reds"))

# === PAGE 3: SEARCH LOGS ===
elif page == "🔎 Fraud Logs & Search":
    st.title("🔍 Explore & Search Fraud Records")

    st.markdown("### 🔎 Check if a Number Has Been Involved in Fraud")
    search_number = st.text_input("Enter a phone number:")
    if search_number:
        match = df[df["CallingNumber"]==(search_number)]
        if not match.empty:
            st.warning(f"⚠️ {len(match)} fraud record(s) found for this number.")
            st.dataframe(match)
        else:
            st.success("✅ This number has no fraud record.")

    st.markdown("### 📋 All Detected Fraud Logs")
    filter_input = st.text_input("Filter fraud logs by phone number:")
    filtered_df = df[df["CallingNumber"]==(filter_input)] if filter_input else df
    st.dataframe(filtered_df, use_container_width=True, height=500)
    st.download_button("📥 Download CSV", data=filtered_df.to_csv(index=False).encode(), file_name="fraud_results.csv")

# === PAGE 4: SIMULATION ===
elif page == "🛠️ Simulate Calls":
    st.title("🛠️ Real-Time Call Simulation")

    # Initialize session state for all form fields
    for field in [
        "CallHour", "CallMinute", "CallSecond",
        "CallingNumber", "CalledNumber", "Callduration",
        "callDay", "callMonth", "callYear",
        "IntrunkTT", "OuttrunkTT", "InSwitch_IGW_TUN", "OutSwitch_IGW_TUN",
        "Intrunk_enc", "Outtrunk_enc"
    ]:
        st.session_state.setdefault(field, "")




    with st.form("new_call_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        CallHour = c1.text_input("Hour", value=st.session_state["CallHour"], key="CallHour")
        CallMinute = c2.text_input("Minute", value=st.session_state["CallMinute"], key="CallMinute")
        CallSecond = c3.text_input("Second", value=st.session_state["CallSecond"], key="CallSecond")

        c4, c5, c6 = st.columns(3)
        CallingNumber = c4.text_input("Calling Number", value=st.session_state["CallingNumber"], key="CallingNumber")
        CalledNumber = c5.text_input("Called Number", value=st.session_state["CalledNumber"], key="CalledNumber")
        Callduration = c6.text_input("Duration", value=st.session_state["Callduration"], key="Callduration")

        c7, c8, c9 = st.columns(3)
        callDay = c7.text_input("Day", value=st.session_state["callDay"], key="callDay")
        callMonth = c8.text_input("Month", value=st.session_state["callMonth"], key="callMonth")
        callYear = c9.text_input("Year", value=st.session_state["callYear"], key="callYear")

        c10, c11 = st.columns(2)
        IntrunkTT = c10.text_input("IntrunkTT", value=st.session_state["IntrunkTT"], key="IntrunkTT")
        OuttrunkTT = c11.text_input("OuttrunkTT", value=st.session_state["OuttrunkTT"], key="OuttrunkTT")

        c12, c13 = st.columns(2)
        InSwitch_IGW_TUN = c12.text_input("InSwitch IGW", value=st.session_state["InSwitch_IGW_TUN"], key="InSwitch_IGW_TUN")
        OutSwitch_IGW_TUN = c13.text_input("OutSwitch IGW", value=st.session_state["OutSwitch_IGW_TUN"], key="OutSwitch_IGW_TUN")

        c14, c15 = st.columns(2)
        Intrunk_enc = c14.text_input("Intrunk", value=st.session_state["Intrunk_enc"], key="Intrunk_enc")
        Outtrunk_enc = c15.text_input("Outtrunk", value=st.session_state["Outtrunk_enc"], key="Outtrunk_enc")

        col_submit, col_clear = st.columns([5, 1])
        with col_submit:
            submit_clicked = st.form_submit_button("➕ Submit")
        with col_clear:
            clear_clicked = st.form_submit_button("🗑️ Clear Form")

        if submit_clicked:
            line = ",".join([
                CallHour, CallMinute, CallSecond,
                CallingNumber, CalledNumber, Callduration,
                callDay, callMonth, callYear,
                IntrunkTT, OuttrunkTT, InSwitch_IGW_TUN, OutSwitch_IGW_TUN,
                Intrunk_enc, Outtrunk_enc
            ])
            with open("new_calls.csv", "a") as f:
                f.write(line + "\n")

            # === Fraud detection wait logic ===
            import datetime

            calling_number = st.session_state["CallingNumber"].strip()
            called_number = st.session_state["CalledNumber"].strip()

            def query_latest_fraud(calling_number, called_number):
                try:
                    temp_conn = mysql.connector.connect(
                        host="mysql", user="root", password="root", database="frauddetection1"
                    )
                    temp_cursor = temp_conn.cursor(dictionary=True)
                    temp_cursor.execute(
                        """
                        SELECT * FROM fraud_results
                        WHERE CallingNumber = %s AND CalledNumber = %s
                        ORDER BY detected_at DESC LIMIT 1
                        """,
                        (float(calling_number), float(called_number))
                    )
                    result = temp_cursor.fetchone()
                    temp_cursor.close()
                    temp_conn.close()
                    return result
                except Exception as e:
                    st.error(f"❌ DB query failed: {e}")
                    return None

            found = False
            result = None
            start_time = datetime.datetime.now()
            timeout = datetime.timedelta(seconds=5)
            progress_bar = st.progress(0)

            with st.spinner("⌛ Waiting for backend to detect fraud..."):
                while not found and datetime.datetime.now() - start_time < timeout:
                    result = query_latest_fraud(calling_number, called_number)

                    if result:
                        found = True
                    else:
                        time.sleep(1)
                        elapsed = (datetime.datetime.now() - start_time).seconds
                        progress_bar.progress(min(elapsed / timeout.seconds, 1.0))

            if found:
                st.success(f"🚨 Fraud Detected: {result['fraud_type']} at {result['detected_at']}")



                # Textual SHAP breakdown
                st.markdown("### 🔍 Why was this flagged as fraud?")


                import plotly.graph_objects as go

                def plot_shap_bar(explanation):
                    features, values = zip(*explanation)
                    features = features[::-1]
                    values = values[::-1]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker=dict(color=values, colorscale='Sunset', line=dict(color='rgba(58, 71, 80, 1.0)', width=1)),
                        hoverinfo='x+y',
                        text=[f"{v:.3f}" for v in values],
                        textposition="auto",
                    ))

                    fig.update_layout(
                        title="🔍 Top SHAP Feature Impacts",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        margin=dict(l=80, r=20, t=50, b=40),
                        height=450,
                    )
                    return fig

                import json
                import os

                key = calling_number
                path = f"shap_explanations/{key}.0.json"
                st.markdown(f"#### 🔬 SHAP Explanation for: `{key}`")

                if os.path.exists(path):
                    with open(path, "r") as f:
                        explanation = json.load(f)
                    fig = plot_shap_bar(explanation)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ SHAP explanation not found yet.")




            else:
                st.info("🟢 Call not flagged as fraudulent (yet). Try again later.")


        if clear_clicked:
            for field in [
                "CallHour", "CallMinute", "CallSecond",
                "CallingNumber", "CalledNumber", "Callduration",
                "callDay", "callMonth", "callYear",
                "IntrunkTT", "OuttrunkTT", "InSwitch_IGW_TUN", "OutSwitch_IGW_TUN",
                "Intrunk_enc", "Outtrunk_enc"
            ]:
                st.session_state[field] = ""


# === CLOSE DB ===
conn.close()
