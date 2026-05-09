"""
streamlit_app.py — Customer Churn Prediction Dashboard
Run: streamlit run streamlit_app.py
Requires api.py running at localhost:8000
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import csv
import os
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL       = "http://localhost:8000"
FEEDBACK_FILE = "feedback_log.csv"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { background:#fee2e2; border-left:4px solid #ef4444;
                   padding:12px 16px; border-radius:8px; color:#7f1d1d; font-weight:600; }
    .risk-medium { background:#fef9c3; border-left:4px solid #eab308;
                   padding:12px 16px; border-radius:8px; color:#713f12; font-weight:600; }
    .risk-low    { background:#dcfce7; border-left:4px solid #22c55e;
                   padding:12px 16px; border-radius:8px; color:#14532d; font-weight:600; }
    .section-title { font-size:1.1rem; font-weight:600; margin-bottom:8px; color:#1e293b; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def risk_label(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.10: return "High"
    if prob >= threshold - 0.15: return "Medium"
    return "Low"


def gauge_chart(prob: float, threshold: float) -> go.Figure:
    color = "#ef4444" if prob >= threshold + 0.10 else \
            "#eab308" if prob >= threshold - 0.15 else "#22c55e"
    lo = max(0, (threshold - 0.15) * 100)
    hi = min(100, (threshold + 0.10) * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,   lo],  "color": "#dcfce7"},
                {"range": [lo,  hi],  "color": "#fef9c3"},
                {"range": [hi, 100],  "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100
            }
        },
        title={"text": f"Churn Probability (cutoff {threshold:.0%})", "font": {"size": 14}}
    ))
    fig.update_layout(height=270, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def shap_bar_chart(shap_features: list) -> go.Figure:
    df = pd.DataFrame(shap_features)
    df = df.sort_values('shap_value')
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in df['shap_value']]
    def shorten(name):
        return name.replace("cat__","").replace("num__","") \
                   .replace("_Yes","").replace("_No","")[:35]
    df['label'] = df['feature'].apply(shorten)
    fig = go.Figure(go.Bar(
        x=df['shap_value'], y=df['label'],
        orientation='h', marker_color=colors,
        text=[f"{v:+.3f}" for v in df['shap_value']],
        textposition='outside'
    ))
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value (impact on churn prediction)",
        height=360, margin=dict(l=10, r=60, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(zeroline=True, zerolinecolor='#94a3b8', zerolinewidth=1.5)
    )
    return fig


def save_feedback(record: dict):
    exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def load_feedback() -> pd.DataFrame:
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()
    return pd.read_csv(FEEDBACK_FILE)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn Predictor")
    st.caption("Powered by XGBoost + SHAP")
    st.divider()

    tab_choice = st.radio("Mode", ["🔍 Single Customer", "📦 Bulk Analysis", "📊 Feedback Log"])
    st.divider()

    # ── Threshold slider ───────────────────────────────────────────────────
    st.markdown("**⚙️ Decision Threshold**")
    threshold = st.slider(
        "Churn cutoff probability",
        min_value=0.20, max_value=0.80,
        value=0.50, step=0.05,
        help="Lower = catch more churners (higher recall). Higher = fewer false alarms (higher precision)."
    )
    if threshold < 0.40:
        st.caption("⚠️ High recall mode — catches more churners, more false positives")
    elif threshold > 0.60:
        st.caption("🎯 High precision mode — fewer false alarms, may miss some churners")
    else:
        st.caption("⚖️ Balanced mode")

    st.divider()
    st.caption("API endpoint")
    api_input = st.text_input("API URL", value=API_URL)
    if st.button("🔌 Check Health"):
        try:
            r = requests.get(f"{api_input}/health", timeout=3)
            st.success(f"Online ✓  ({r.json()['model']})")
        except:
            st.error("API offline — start uvicorn first")


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Single Customer
# ══════════════════════════════════════════════════════════════════════════
if "Single" in tab_choice:
    st.header("🔍 Single Customer Churn Prediction")
    st.caption("Fill in customer details and click **Predict** to get churn probability + SHAP explanation.")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner    = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)
            phone       = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security    = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup      = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device      = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech        = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv          = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies      = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with c3:
            st.markdown('<div class="section-title">Billing</div>', unsafe_allow_html=True)
            contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                      value=float(monthly * max(tenure, 1)), step=1.0)

        submitted = st.form_submit_button("⚡ Predict Churn", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
        }
        with st.spinner("Predicting..."):
            try:
                resp = requests.post(f"{api_input}/predict", json=payload, timeout=15)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        prob  = result['churn_probability']
        shaps = result['shap_top_features']
        pred  = prob >= threshold
        risk  = risk_label(prob, threshold)

        cls  = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}[risk]
        icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk]
        st.markdown(
            f'<div class="{cls}">{icon} {risk} Risk — '
            f'{"Likely to churn" if pred else "Likely to stay"} '
            f'(cutoff: {threshold:.0%})</div>',
            unsafe_allow_html=True
        )
        st.write("")

        col_gauge, col_shap = st.columns([1, 1.6])
        with col_gauge:
            st.plotly_chart(gauge_chart(prob, threshold), use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("Probability", f"{prob:.1%}")
            m2.metric("Risk Level", risk)
        with col_shap:
            st.plotly_chart(shap_bar_chart(shaps), use_container_width=True)

        # ── Annotation & Feedback ─────────────────────────────────────────
        st.divider()
        st.subheader("📝 Annotation & Feedback")
        st.caption("Correct the prediction or add context — saved to `feedback_log.csv` for monitoring.")

        fb1, fb2 = st.columns([1.3, 1])
        with fb1:
            note = st.text_area(
                "📌 Notes / Annotation",
                placeholder="e.g. 'VIP account — escalate before acting' or 'Customer complained about billing last week'",
                height=110, key="note_single"
            )
        with fb2:
            st.markdown("**Was the prediction correct?**")
            actual = st.radio(
                "Actual churn outcome (if known)",
                ["Unknown", "Yes — did churn", "No — did not churn"],
                key="actual_single"
            )
            action = st.selectbox(
                "Action taken",
                ["None", "Sent retention offer", "Called customer", "Escalated to team", "Other"],
                key="action_single"
            )

        if st.button("💾 Save Annotation", use_container_width=True):
            save_feedback({
                "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode":              "single",
                "churn_probability": prob,
                "threshold_used":    threshold,
                "predicted_churn":   pred,
                "risk_level":        risk,
                "actual_outcome":    actual,
                "action_taken":      action,
                "notes":             note,
                "tenure":            tenure,
                "contract":          contract,
                "monthly_charges":   monthly,
                "internet_service":  internet,
            })
            st.success("Saved to feedback_log.csv ✓  →  View in **📊 Feedback Log** tab.")

        with st.expander("📋 Raw API Response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Bulk Analysis
# ══════════════════════════════════════════════════════════════════════════
elif "Bulk" in tab_choice:
    st.header("📦 Bulk Customer Analysis")
    st.caption(f"Active threshold: **{threshold:.0%}** — adjust in sidebar.")

    template_cols = [
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ]
    example_row = [
        "Female",0,"Yes","No",5,"Yes","No","Fiber optic","No","No",
        "No","No","No","No","Month-to-month","Yes","Electronic check",70.7,151.65
    ]
    buf = io.StringIO()
    pd.DataFrame([example_row], columns=template_cols).to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV Template", data=buf.getvalue(),
                       file_name="churn_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write(f"**{len(df_upload)} customers loaded**")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("⚡ Run Bulk Prediction", type="primary"):
            with st.spinner(f"Predicting {len(df_upload)} customers..."):
                try:
                    resp = requests.post(
                        f"{api_input}/predict/bulk",
                        json={"customers": df_upload.to_dict(orient='records')},
                        timeout=60
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            preds       = result['predictions']
            probs       = [p['churn_probability'] for p in preds]
            pred_labels = [p >= threshold for p in probs]
            risks       = [risk_label(p, threshold) for p in probs]

            # Summary
            st.subheader("📊 Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",           len(preds))
            m2.metric("Predicted Churn", sum(pred_labels))
            m3.metric("High Risk",       sum(r == "High" for r in risks))
            m4.metric("Avg Probability", f"{np.mean(probs):.1%}")

            # Charts
            col_hist, col_pie = st.columns(2)
            with col_hist:
                fig_hist = px.histogram(
                    x=probs, nbins=20, color_discrete_sequence=['#4C72B0'],
                    title="Churn Probability Distribution",
                    labels={"x": "Churn Probability", "y": "Count"}
                )
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                                   annotation_text=f"Cutoff {threshold:.0%}")
                fig_hist.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_pie:
                rc = pd.Series(risks).value_counts()
                fig_pie = px.pie(
                    values=rc.values, names=rc.index,
                    title="Risk Level Distribution",
                    color=rc.index,
                    color_discrete_map={"High":"#ef4444","Medium":"#eab308","Low":"#22c55e"}
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Results + annotation
            st.subheader("📋 Results & Annotation")
            results_df = df_upload.copy()
            results_df['churn_probability'] = [round(p, 4) for p in probs]
            results_df['churn_prediction']  = pred_labels
            results_df['risk_level']        = risks
            results_df['threshold_used']    = threshold
            results_df['run_timestamp']     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results_df = results_df.sort_values('churn_probability', ascending=False)

            batch_note = st.text_area(
                "📌 Batch annotation (added to all rows in export)",
                placeholder="e.g. 'Q2 2024 retention campaign — priority: High risk segment'",
                height=70
            )
            results_df['batch_note'] = batch_note

            st.dataframe(results_df, use_container_width=True)

            ts = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                "⬇️ Download Results + Annotations",
                data=results_df.to_csv(index=False),
                file_name=f"churn_predictions_{ts}.csv",
                mime="text/csv"
            )

            if st.button("💾 Log This Bulk Run to Feedback"):
                save_feedback({
                    "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode":              "bulk",
                    "churn_probability": round(float(np.mean(probs)), 4),
                    "threshold_used":    threshold,
                    "predicted_churn":   sum(pred_labels),
                    "risk_level":        f"{sum(r=='High' for r in risks)} high-risk",
                    "actual_outcome":    "Unknown",
                    "action_taken":      "Bulk export",
                    "notes":             batch_note or f"Bulk: {len(preds)} customers",
                    "tenure": "—", "contract": "—",
                    "monthly_charges": "—", "internet_service": "—",
                })
                st.success("Bulk run logged ✓")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Feedback Log
# ══════════════════════════════════════════════════════════════════════════
else:
    st.header("📊 Feedback & Annotation Log")
    st.caption("All saved predictions with human labels and notes — for monitoring and future retraining.")

    df_fb = load_feedback()

    if df_fb.empty:
        st.info("No feedback yet. Run predictions and click **Save Annotation** to start logging.")
    else:
        # Accuracy metrics
        known   = df_fb[df_fb['actual_outcome'] != 'Unknown']
        if len(known) > 0:
            correct = known[
                ((known['actual_outcome'] == 'Yes — did churn')    &  known['predicted_churn'].astype(bool)) |
                ((known['actual_outcome'] == 'No — did not churn') & ~known['predicted_churn'].astype(bool))
            ]
            acc = len(correct) / len(known)
        else:
            correct, acc = pd.DataFrame(), None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Logged",         len(df_fb))
        m2.metric("With Known Outcome",   len(known))
        m3.metric("Correct Predictions",  len(correct) if len(known) > 0 else "—")
        m4.metric("Observed Accuracy",    f"{acc:.1%}" if acc is not None else "—")

        st.divider()

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            fmode = st.multiselect("Mode", ["single","bulk"], default=["single","bulk"])
        with fc2:
            frisk = st.multiselect("Risk level", ["Low","Medium","High"],
                                   default=["Low","Medium","High"])

        mask = df_fb['mode'].isin(fmode) & \
               df_fb['risk_level'].str.contains('|'.join(frisk), na=False)
        st.dataframe(df_fb[mask], use_container_width=True)

        # Running accuracy chart
        if len(known) >= 3:
            known_s = known.copy().reset_index(drop=True)
            known_s['correct'] = (
                ((known_s['actual_outcome'] == 'Yes — did churn')    &  known_s['predicted_churn'].astype(bool)) |
                ((known_s['actual_outcome'] == 'No — did not churn') & ~known_s['predicted_churn'].astype(bool))
            ).astype(int)
            known_s['rolling_acc'] = known_s['correct'].expanding().mean() * 100
            fig = px.line(known_s, x=known_s.index, y='rolling_acc', markers=True,
                          title="Running Accuracy Over Labeled Predictions",
                          labels={"x":"Prediction #","rolling_acc":"Accuracy (%)"})
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        # Export + clear
        st.download_button("⬇️ Export Full Log", data=df_fb.to_csv(index=False),
                           file_name="feedback_log_export.csv", mime="text/csv")
        st.divider()
        if st.button("🗑️ Clear Feedback Log", type="secondary"):
            if os.path.exists(FEEDBACK_FILE): os.remove(FEEDBACK_FILE)
            st.success("Log cleared.")
            st.rerun()
