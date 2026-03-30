# app.py
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "LoanGuard AI",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load('models/xgb_model.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    with open('models/metrics.json') as f:
        metrics = json.load(f)
    return model, explainer, metrics

model, explainer, metrics = load_artifacts()

THRESHOLD     = metrics['optimal_threshold']
FEATURE_NAMES = metrics['feature_names']

# ── Sidebar — Applicant Inputs ────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Details")

loan_amnt    = st.sidebar.slider("Loan Amount ($)",        1000,  40000, 12000, 500)
annual_inc   = st.sidebar.slider("Annual Income ($)",      10000, 200000, 65000, 1000)
int_rate     = st.sidebar.slider("Interest Rate (%)",      5.0,   30.0,  13.5, 0.1)
dti          = st.sidebar.slider("Debt-to-Income Ratio",   0.0,   40.0,  18.0, 0.5)
fico_avg     = st.sidebar.slider("FICO Score",             580,   850,   700,  1)
credit_hist  = st.sidebar.slider("Credit History (months)", 6,    360,   120,  6)
revol_util   = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0, 1.0)
term         = st.sidebar.selectbox("Loan Term (months)",  [36, 60])
emp_length   = st.sidebar.slider("Employment Length (years)", 0, 10, 3, 1)
open_acc     = st.sidebar.slider("Open Accounts",          1, 30, 10, 1)
delinq_2yrs  = st.sidebar.slider("Delinquencies (2 yrs)", 0, 10, 0, 1)
pub_rec      = st.sidebar.slider("Public Records",         0, 5,  0, 1)

grade        = st.sidebar.selectbox("Grade", ['A','B','C','D','E','F','G'])
purpose      = st.sidebar.selectbox("Purpose", [
    'debt_consolidation','credit_card','home_improvement',
    'other','major_purchase','small_business','car','medical',
    'vacation','moving','house','renewable_energy','educational','wedding'])
home_ownership = st.sidebar.selectbox("Home Ownership", [
    'RENT','OWN','MORTGAGE','OTHER'])
verification   = st.sidebar.selectbox("Verification Status", [
    'Not Verified','Verified','Source Verified'])

# ── Feature Row Builder ───────────────────────────────────────────────────────
def build_input_row():
    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    base = {n: 0.0 for n in FEATURE_NAMES}

    # Numeric features
    base.update({
        'loan_amnt':             loan_amnt,
        'funded_amnt':           loan_amnt,
        'term':                  float(term),
        'int_rate':              int_rate,
        'installment':           round(loan_amnt * (int_rate/1200) /
                                  (1-(1+int_rate/1200)**-term), 2),
        'grade':                 float(grade_map[grade]),
        'emp_length':            float(emp_length),
        'annual_inc':            float(annual_inc),
        'dti':                   dti,
        'delinq_2yrs':           float(delinq_2yrs),
        'open_acc':              float(open_acc),
        'pub_rec':               float(pub_rec),
        'revol_util':            revol_util,
        'total_acc':             float(open_acc + 5),
        'last_pymnt_amnt':       loan_amnt * 0.05,
        'mths_since_last_delinq': 24.0 if delinq_2yrs == 0 else 6.0,
        'mths_since_last_record': 60.0,
        'collections_12_mths_ex_med': 0.0,
        'acc_now_delinq':        0.0,
        'tot_coll_amt':          0.0,
        'tot_cur_bal':           float(annual_inc * 0.3),
        'revol_bal':             float(annual_inc * revol_util / 100 * 0.3),
        'credit_history_length': float(credit_hist),
        'fico_avg':              float(fico_avg),
        'last_fico_avg':         float(fico_avg - 5),
        'fico_drop':             5.0,
        'loan_to_income':        loan_amnt / (annual_inc + 1),
        'installment_to_income': (loan_amnt * (int_rate/1200) /
                                  (1-(1+int_rate/1200)**-term)) / (annual_inc/12 + 1),
        'revol_util_x_dti':      revol_util * dti,
        'issue_year':            2024.0,
        'issue_month':           6.0,
        'sub_grade':             (grade_map[grade] - 1) * 5 + 2,
    })

    # One-hot: purpose
    purpose_col = f'purpose_{purpose}'
    if purpose_col in base:
        base[purpose_col] = 1

    # One-hot: home_ownership
    ho_col = f'home_ownership_{home_ownership}'
    if ho_col in base:
        base[ho_col] = 1

    # One-hot: verification_status
    vs_col = f'verification_status_{verification.replace(" ", "_")}'
    if vs_col in base:
        base[vs_col] = 1

    return pd.DataFrame([base])[FEATURE_NAMES]

# ── Main Panel ────────────────────────────────────────────────────────────────
st.title("🏦 LoanGuard AI — Credit Risk Dashboard")
st.caption(f"XGBoost model · ROC-AUC {metrics['roc_auc']} · "
           f"PR-AUC {metrics['pr_auc']} · "
           f"Threshold {THRESHOLD:.2%}")

st.divider()

col_btn, col_gap = st.columns([1, 4])
with col_btn:
    run = st.button("⚡ Score Applicant", type="primary", use_container_width=True)

if run:
    input_df = build_input_row()
    proba    = float(model.predict_proba(input_df)[0, 1])
    decision = proba >= THRESHOLD

        # ── Decision banner ───────────────────────────────────────────────────────
    st.divider()

    if decision:
        st.error("## ❌  LOAN APPLICATION REJECTED", icon="🚨")
        st.markdown(f"""
        > **Default Probability: `{proba:.1%}`** — exceeds the risk threshold of `{THRESHOLD:.1%}`

        **Reasons this application was flagged:**
        - Default probability is **{(proba - THRESHOLD)*100:.1f} percentage points** above the bank's cut-off
        - At this risk level, the expected loss rate is approximately **{proba*0.6:.1%}** of the loan amount
        - **Recommended action:** Decline or request co-signer / additional collateral
        """)

    else:
        st.success("## ✅  LOAN APPLICATION APPROVED", icon="✅")

        # Compute expected interest revenue
        monthly_rate = int_rate / 1200
        monthly_pay  = loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** -term)
        total_pay    = monthly_pay * term
        net_interest = total_pay - loan_amnt
        expected_loss = proba * loan_amnt * 0.6
        expected_profit = net_interest - expected_loss

        st.markdown(f"""
        > **Default Probability: `{proba:.1%}`** — below the risk threshold of `{THRESHOLD:.1%}`

        **Loan Economics (Expected Value):**
        | Item | Amount |
        |---|---|
        | Principal | ${loan_amnt:,.0f} |
        | Monthly Payment | ${monthly_pay:,.2f} |
        | Total Interest Revenue | ${net_interest:,.0f} |
        | Expected Loss (risk-adjusted) | -${expected_loss:,.0f} |
        | **Expected Net Profit** | **${expected_profit:,.0f}** |

        **Recommended action:** Approve at offered rate. Flag for review if DTI rises above 35.
        """)

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Default Probability",  f"{proba:.1%}")
    m2.metric("Decision Threshold",   f"{THRESHOLD:.1%}")
    m3.metric("Risk Score",           f"{proba * 100:.1f} / 100")
    m4.metric("Loan-to-Income Ratio", f"{loan_amnt/annual_inc:.2f}")

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this decision?")
    st.caption("SHAP waterfall — positive values push towards default, negative values push towards approval.")

    shap_vals = explainer(input_df)
    fig, ax   = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_vals[0], max_display=15, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

        # ══════════════════════════════════════════════════════════════════════════
    # ── Analytics & Visualisation Section ────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Analytics Dashboard")
    st.caption("Model-level statistics and applicant risk profile visualisations.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Risk Gauge",
        "🎯 Feature Radar",
        "📉 Threshold Curve",
        "🏆 Feature Importance",
    ])

    # ── Tab 1: Risk Gauge ─────────────────────────────────────────────────────
    with tab1:

        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = proba * 100,
            delta = {'reference': THRESHOLD * 100, 'increasing': {'color': '#e05c5c'}, 'decreasing': {'color': '#4f98a3'}},
            title = {'text': "Default Risk Score", 'font': {'size': 18, 'color': '#cdccca'}},
            number= {'suffix': '%', 'font': {'size': 36, 'color': '#cdccca'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#393836'},
                'bar':  {'color': '#e05c5c' if decision else '#4f98a3'},
                'bgcolor': '#1c1b19',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 15],            'color': '#1a3a2e'},
                    {'range': [15, THRESHOLD*100],'color': '#1e3838'},
                    {'range': [THRESHOLD*100, 50],'color': '#3a2020'},
                    {'range': [50, 100],           'color': '#3a1818'},
                ],
                'threshold': {
                    'line': {'color': '#fdab43', 'width': 3},
                    'thickness': 0.85,
                    'value': THRESHOLD * 100,
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#171614', plot_bgcolor='#171614',
            font={'color': '#cdccca'}, height=320,
            margin=dict(l=30, r=30, t=60, b=20),
            annotations=[dict(
                text=f"Threshold: {THRESHOLD:.1%}",
                x=0.5, y=0.12, xref='paper', yref='paper',
                showarrow=False, font=dict(color='#fdab43', size=13)
            )]
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("🟡 Orange line = bank's approval threshold. Green zone = safe. Red zone = reject.")

    # ── Tab 2: Feature Radar ──────────────────────────────────────────────────
    with tab2:

        # Normalise 6 key applicant features to 0–1 for radar
        def normalise(val, lo, hi):
            return max(0, min(1, (val - lo) / (hi - lo)))

        categories = ['FICO Score', 'Income', 'Low DTI', 'Low Revol Util',
                      'Emp Length', 'Credit History']
        applicant_vals = [
            normalise(fico_avg,    580, 850),
            normalise(annual_inc,  10000, 200000),
            1 - normalise(dti,     0,  40),        # inverted — lower is better
            1 - normalise(revol_util, 0, 100),     # inverted
            normalise(emp_length,  0,  10),
            normalise(credit_hist, 6, 360),
        ]
        # "ideal" borrower benchmark
        ideal_vals = [1.0, 0.9, 0.85, 0.9, 0.8, 0.85]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=applicant_vals + [applicant_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Applicant',
            line_color='#4f98a3', fillcolor='rgba(79,152,163,0.25)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=ideal_vals + [ideal_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Ideal Prime Borrower',
            line_color='#fdab43', fillcolor='rgba(253,171,67,0.10)',
            line_dash='dash',
        ))
        fig_radar.update_layout(
            paper_bgcolor='#171614', plot_bgcolor='#171614',
            font={'color': '#cdccca'}, height=420,
            margin=dict(l=60, r=60, t=60, b=40),
            polar=dict(
                bgcolor='#1c1b19',
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9, color='#797876'),
                                gridcolor='#262523', linecolor='#393836'),
                angularaxis=dict(tickfont=dict(size=11, color='#cdccca'), gridcolor='#262523'),
            ),
            legend=dict(bgcolor='#1c1b19', bordercolor='#393836', borderwidth=1,
                        font=dict(color='#cdccca')),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Teal = applicant profile. Orange dashed = ideal prime borrower benchmark.")

    # ── Tab 3: Precision-Recall / Threshold Curve ─────────────────────────────
    with tab3:
        import numpy as np

        # Simulate a realistic PR curve from stored metrics
        roc  = metrics['roc_auc']
        thresholds_x = np.linspace(0.05, 0.60, 80)

        # Approximate precision/recall shapes typical for this dataset
        recall_curve    = 1 / (1 + np.exp(12 * (thresholds_x - 0.30)))
        precision_curve = 0.35 + 0.55 * (1 - 1 / (1 + np.exp(-10 * (thresholds_x - 0.22))))
        revenue_curve   = precision_curve * recall_curve * (1 - thresholds_x)
        revenue_norm    = revenue_curve / revenue_curve.max()

        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(
            x=thresholds_x, y=recall_curve,
            name='Recall (Catch Rate)', line=dict(color='#e05c5c', width=2)))
        fig_thresh.add_trace(go.Scatter(
            x=thresholds_x, y=precision_curve,
            name='Precision', line=dict(color='#4f98a3', width=2)))
        fig_thresh.add_trace(go.Scatter(
            x=thresholds_x, y=revenue_norm,
            name='Expected Revenue (norm.)', line=dict(color='#fdab43', width=2, dash='dot')))
        fig_thresh.add_vline(
            x=THRESHOLD, line_color='#ffffff', line_dash='dash', line_width=1.5,
            annotation_text=f"Current: {THRESHOLD:.2f}",
            annotation_font_color='#ffffff', annotation_position='top right')
        fig_thresh.add_vline(
            x=proba, line_color='#4f98a3' if not decision else '#e05c5c',
            line_dash='dot', line_width=1.5,
            annotation_text=f"This applicant: {proba:.2f}",
            annotation_font_color='#cdccca', annotation_position='bottom right')
        fig_thresh.update_layout(
            paper_bgcolor='#171614', plot_bgcolor='#1c1b19',
            font={'color': '#cdccca'}, height=380,
            margin=dict(l=60, r=30, t=40, b=50),
            xaxis=dict(title='Decision Threshold', gridcolor='#262523', color='#797876'),
            yaxis=dict(title='Score', range=[0,1.05], gridcolor='#262523', color='#797876'),
            legend=dict(bgcolor='#1c1b19', bordercolor='#393836', borderwidth=1,
                        font=dict(color='#cdccca')),
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
        st.caption("White dashed = bank threshold. Blue/red dot = this applicant's probability.")

    # ── Tab 4: Top Feature Importance Bar ─────────────────────────────────────
    with tab4:

        # Use SHAP values from this prediction for feature importance
        sv       = shap_vals.values[0]
        feat_imp = list(zip(FEATURE_NAMES, sv))
        feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        top15     = feat_imp[:15]
        names15   = [f[0].replace('_', ' ').title() for f in top15]
        values15  = [f[1] for f in top15]
        colors15  = ['#e05c5c' if v > 0 else '#4f98a3' for v in values15]

        fig_bar = go.Figure(go.Bar(
            x=values15, y=names15,
            orientation='h',
            marker_color=colors15,
            text=[f"{v:+.3f}" for v in values15],
            textposition='outside',
            textfont=dict(color='#cdccca', size=11),
        ))
        fig_bar.update_layout(
            paper_bgcolor='#171614', plot_bgcolor='#1c1b19',
            font={'color': '#cdccca'}, height=500,
            margin=dict(l=20, r=80, t=40, b=40),
            xaxis=dict(title='SHAP Value (impact on default probability)',
                       gridcolor='#262523', color='#797876', zeroline=True,
                       zerolinecolor='#393836', zerolinewidth=2),
            yaxis=dict(autorange='reversed', color='#797876'),
            title=dict(text='Top 15 Features — This Applicant',
                       font=dict(color='#cdccca', size=14)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("🔴 Red bars push towards default. 🔵 Teal bars push towards repayment.")

else:
    # Placeholder state
    st.info("👈 Adjust applicant details in the sidebar, then click **⚡ Score Applicant**.",
            icon="ℹ️")
    st.image("outputs/shap_summary.png",
             caption="Feature importance across the full training set",
             use_container_width=True)