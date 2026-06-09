import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce BI Platform",
    page_icon="📦",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    kpis         = pd.read_csv(os.path.join(BASE, 'data', 'kpis.csv'))
    monthly      = pd.read_csv(os.path.join(BASE, 'data', 'monthly_revenue.csv'))
    by_state     = pd.read_csv(os.path.join(BASE, 'data', 'revenue_by_state.csv'))
    category     = pd.read_csv(os.path.join(BASE, 'data', 'category_performance.csv'))
    delivery     = pd.read_csv(os.path.join(BASE, 'data', 'delivery_performance.csv'))
    payments     = pd.read_csv(os.path.join(BASE, 'data', 'payment_analysis.csv'))
    del_review   = pd.read_csv(os.path.join(BASE, 'data', 'review_vs_delivery.csv'))
    by_day       = pd.read_csv(os.path.join(BASE, 'data', 'orders_by_day.csv'))
    by_hour      = pd.read_csv(os.path.join(BASE, 'data', 'orders_by_hour.csv'))
    rfm_summary  = pd.read_csv(os.path.join(BASE, 'data', 'rfm_summary.csv'))
    seller_score = pd.read_csv(os.path.join(BASE, 'data', 'seller_scorecard.csv'))
    return (kpis, monthly, by_state, category, delivery,
            payments, del_review, by_day, by_hour, rfm_summary, seller_score)

@st.cache_resource
def load_models():
    m = os.path.join(BASE, 'models')
    with open(os.path.join(m, 'review_predictor.pkl'), 'rb') as f:
        review_model = pickle.load(f)
    with open(os.path.join(m, 'delay_predictor.pkl'), 'rb') as f:
        delay_model = pickle.load(f)
    with open(os.path.join(m, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    states     = pd.read_csv(os.path.join(m, 'states.csv'))['state'].tolist()
    categories = pd.read_csv(os.path.join(m, 'categories.csv'))['category'].tolist()
    payments   = pd.read_csv(os.path.join(m, 'payments.csv'))['payment'].tolist()
    return review_model, delay_model, encoders, states, categories, payments

(kpis, monthly, by_state, category, delivery,
 payments, del_review, by_day, by_hour,
 rfm_summary, seller_score) = load_data()

review_model, delay_model, encoders, states, categories, pay_types = load_models()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shop.png", width=60)
st.sidebar.title("Filters")

selected_state = st.sidebar.multiselect(
    "Filter by State",
    options=by_state['state'].tolist(),
    default=[]
)

top_n = st.sidebar.slider("Top N States to show", 5, 27, 10)
top_n_cat = st.sidebar.slider("Top N Categories to show", 5, 20, 10)

st.sidebar.divider()
st.sidebar.markdown("**About**")
st.sidebar.markdown("Built by **Anuj Sherekar**")
st.sidebar.markdown("Python · SQL · Scikit-learn · Streamlit")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
filtered_state = by_state[
    by_state['state'].isin(selected_state)
] if selected_state else by_state

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📦 E-Commerce BI & Predictive Analytics Platform")
st.markdown("**100,000+ real orders · Olist Brazil · 2016–2018 · Python · SQL · ML**")
st.divider()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k = kpis.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue",     f"R$ {k['total_revenue']:,.0f}",    "100k+ orders")
c2.metric("📦 Total Orders",      f"{int(k['total_orders']):,}",       "2016–2018")
c3.metric("⭐ Avg Review Score",  f"{k['avg_review']} / 5.0",          "Customer satisfaction")
c4.metric("🚚 Avg Delivery Days", f"{k['avg_delivery_days']} days",    "End-to-end")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sales Overview",
    "🗺️ Regional Analysis",
    "🤖 ML Predictions",
    "👥 Customer Segments",
    "🏆 Seller Scorecard"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Monthly Revenue Trend")
    fig1 = px.bar(monthly, x='year_month', y='revenue',
        labels={'year_month': 'Month', 'revenue': 'Revenue (R$)'},
        color='revenue', color_continuous_scale='Blues')
    fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders by Day of Week")
        fig_day = px.bar(by_day, x='day', y='orders',
            color='orders', color_continuous_scale='Blues')
        fig_day.update_layout(showlegend=False)
        st.plotly_chart(fig_day, use_container_width=True)

    with col2:
        st.subheader("Orders by Hour of Day")
        fig_hr = px.line(by_hour, x='hour', y='orders', markers=True)
        fig_hr.update_traces(line_color='#1F3864')
        st.plotly_chart(fig_hr, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Payment Methods")
        fig_pay = px.pie(payments, names='payment_type', values='total_orders',
            color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_pay.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pay, use_container_width=True)

    with col2:
        st.subheader("Delivery Time vs Review Score")
        fig_dr = px.bar(del_review, x='review_score', y='avg_delivery_days',
            color='avg_delivery_days', color_continuous_scale='RdYlGn_r',
            labels={'review_score': 'Review Score',
                    'avg_delivery_days': 'Avg Delivery Days'})
        fig_dr.update_layout(showlegend=False)
        st.plotly_chart(fig_dr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Top {top_n} States by Revenue")
    display_state = filtered_state.head(top_n) if not selected_state else filtered_state

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(display_state.head(top_n), x='state', y='total_revenue',
            color='total_revenue', color_continuous_scale='Teal',
            labels={'state': 'State', 'total_revenue': 'Revenue (R$)'})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(display_state.head(top_n), x='state', y='total_orders',
            color='total_orders', color_continuous_scale='Blues',
            labels={'state': 'State', 'total_orders': 'Orders'})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Delivery Performance by State")
    col1, col2 = st.columns(2)
    with col1:
        fig5 = px.bar(delivery.head(top_n), x='state', y='actual_delivery_days',
            color='actual_delivery_days', color_continuous_scale='Reds',
            labels={'state': 'State', 'actual_delivery_days': 'Avg Days'})
        fig5.update_layout(showlegend=False, title="Slowest States")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig6 = px.bar(delivery.tail(top_n), x='state', y='actual_delivery_days',
            color='actual_delivery_days', color_continuous_scale='Greens',
            labels={'state': 'State', 'actual_delivery_days': 'Avg Days'})
        fig6.update_layout(showlegend=False, title="Fastest States")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Category Performance")
    fig4 = px.scatter(category.head(top_n_cat),
        x='avg_review_score', y='total_revenue',
        size='total_orders', color='avg_review_score',
        hover_name='category',
        labels={'avg_review_score': 'Avg Review Score',
                'total_revenue': 'Revenue (R$)'},
        color_continuous_scale='RdYlGn')
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 ML-Powered Order Predictions")
    st.markdown("Fill in order details below to get instant predictions powered by Random Forest models trained on 100,000+ orders.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⭐ Predict Review Score")
        st.caption("Predicts what review score this order is likely to receive")

        p_state    = st.selectbox("Customer State", states, key='r_state')
        p_category = st.selectbox("Product Category", categories, key='r_cat')
        p_payment  = st.selectbox("Payment Type", pay_types, key='r_pay')
        p_days     = st.slider("Expected Delivery Days", 1, 60, 12, key='r_days')
        p_value    = st.number_input("Order Value (R$)", 10.0, 5000.0, 150.0, key='r_val')
        p_install  = st.slider("Instalments", 1, 24, 1, key='r_inst')

        if st.button("Predict Review Score", type="primary"):
            state_enc    = encoders['state'].transform([p_state])[0]
            category_enc = encoders['category'].transform([p_category])[0]
            payment_enc  = encoders['payment'].transform([p_payment])[0]

            features = np.array([[p_days, p_value, p_install,
                                   state_enc, category_enc, payment_enc]])
            pred = review_model.predict(features)[0]
            proba = review_model.predict_proba(features)[0]
            confidence = round(max(proba) * 100, 1)

            stars = "⭐" * pred
            color = "green" if pred >= 4 else "orange" if pred == 3 else "red"
            st.markdown(f"### Predicted Score: :{color}[{stars} ({pred}/5)]")
            st.markdown(f"**Confidence:** {confidence}%")

            if pred <= 2:
                st.error("⚠️ High risk of bad review — consider faster delivery or discount.")
            elif pred == 3:
                st.warning("🟡 Average review likely — check delivery time and category quality.")
            else:
                st.success("✅ Positive review likely — order looks good!")

    with col2:
        st.markdown("#### 🚚 Predict Delivery Delay Risk")
        st.caption("Predicts whether this order is at risk of taking more than 15 days")

        d_state    = st.selectbox("Customer State", states, key='d_state')
        d_category = st.selectbox("Product Category", categories, key='d_cat')
        d_payment  = st.selectbox("Payment Type", pay_types, key='d_pay')
        d_value    = st.number_input("Order Value (R$)", 10.0, 5000.0, 150.0, key='d_val')
        d_install  = st.slider("Instalments", 1, 24, 1, key='d_inst')

        if st.button("Predict Delay Risk", type="primary"):
            state_enc    = encoders['state'].transform([d_state])[0]
            category_enc = encoders['category'].transform([d_category])[0]
            payment_enc  = encoders['payment'].transform([d_payment])[0]

            features = np.array([[d_value, d_install,
                                   state_enc, category_enc, payment_enc]])
            pred  = delay_model.predict(features)[0]
            proba = delay_model.predict_proba(features)[0]
            delay_prob = round(proba[1] * 100, 1)

            if pred == 1:
                st.error(f"⚠️ HIGH DELAY RISK — {delay_prob}% probability of delay")
                st.markdown("**Recommendation:** Prioritise fulfillment or notify customer early.")
            else:
                st.success(f"✅ LOW DELAY RISK — {delay_prob}% probability of delay")
                st.markdown("**Recommendation:** Order is likely to arrive on time.")

            st.progress(int(delay_prob))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER SEGMENTS (RFM)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("👥 RFM Customer Segmentation")
    st.markdown("""
    **RFM Analysis** segments customers based on:
    - **Recency** — how recently did they order?
    - **Frequency** — how often do they order?
    - **Monetary** — how much do they spend?
    """)

    col1, col2 = st.columns(2)
    with col1:
        fig_rfm1 = px.bar(rfm_summary, x='segment', y='customers',
            color='customers', color_continuous_scale='Blues',
            title='Customers per Segment',
            labels={'segment': 'Segment', 'customers': 'Customers'})
        fig_rfm1.update_layout(showlegend=False)
        st.plotly_chart(fig_rfm1, use_container_width=True)

    with col2:
        fig_rfm2 = px.bar(rfm_summary, x='segment', y='avg_monetary',
            color='avg_monetary', color_continuous_scale='Greens',
            title='Avg Spend per Segment (R$)',
            labels={'segment': 'Segment', 'avg_monetary': 'Avg Spend (R$)'})
        fig_rfm2.update_layout(showlegend=False)
        st.plotly_chart(fig_rfm2, use_container_width=True)

    st.subheader("Segment Summary Table")
    st.dataframe(
        rfm_summary.style.format({
            'avg_monetary': 'R$ {:.2f}',
            'avg_recency': '{:.0f} days'
        }),
        use_container_width=True
    )

    st.subheader("💡 Business Recommendations by Segment")
    recs = {
        'Champion':  '🏆 Reward them. Offer early access, loyalty perks. These are your best customers.',
        'Loyal':     '💙 Upsell and cross-sell. They trust you — offer bundles and premium categories.',
        'Potential': '🌱 Engage with personalised offers. One good experience converts them to Loyal.',
        'At Risk':   '⚠️ Send win-back campaigns. Offer discounts before they churn completely.',
        'Lost':      '📧 Last attempt — heavy discount or survey to understand why they left.'
    }
    for seg, rec in recs.items():
        with st.expander(f"{seg} Customers"):
            st.markdown(rec)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SELLER SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🏆 Seller Performance Scorecard")
    st.markdown("Sellers ranked by composite score: **Revenue (40%) + Review Score (40%) + Delivery Speed (20%)**")

    grade_filter = st.multiselect(
        "Filter by Grade",
        options=['A', 'B', 'C', 'D'],
        default=['A', 'B']
    )

    display_sellers = seller_score[
        seller_score['grade'].isin(grade_filter)
    ].head(20)

    col1, col2 = st.columns(2)
    with col1:
        fig_s1 = px.bar(display_sellers.head(15),
            x='composite_score', y='seller_short',
            orientation='h',
            color='composite_score', color_continuous_scale='RdYlGn',
            labels={'composite_score': 'Score', 'seller_short': 'Seller'},
            title='Top Sellers by Composite Score')
        fig_s1.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_s1, use_container_width=True)

    with col2:
        fig_s2 = px.scatter(display_sellers,
            x='avg_review', y='avg_delivery',
            color='composite_score',
            size='total_orders',
            hover_name='seller_short',
            color_continuous_scale='RdYlGn',
            labels={'avg_review': 'Avg Review', 'avg_delivery': 'Avg Delivery Days'},
            title='Review Score vs Delivery Speed')
        st.plotly_chart(fig_s2, use_container_width=True)

    st.subheader("Grade Distribution")
    grade_dist = seller_score['grade'].value_counts().reset_index()
    grade_dist.columns = ['grade', 'count']
    fig_grade = px.pie(grade_dist, names='grade', values='count',
        color_discrete_sequence=['#2ecc71','#3498db','#f39c12','#e74c3c'])
    st.plotly_chart(fig_grade, use_container_width=True)

    st.subheader("Full Seller Table")
    st.dataframe(
        display_sellers[[
            'seller_short', 'composite_score', 'grade',
            'total_revenue', 'total_orders', 'avg_review', 'avg_delivery'
        ]].rename(columns={
            'seller_short': 'Seller ID',
            'composite_score': 'Score',
            'grade': 'Grade',
            'total_revenue': 'Revenue (R$)',
            'total_orders': 'Orders',
            'avg_review': 'Avg Review',
            'avg_delivery': 'Avg Delivery Days'
        }).style.format({
            'Revenue (R$)': 'R$ {:.0f}',
            'Score': '{:.1f}',
            'Avg Review': '{:.2f}',
            'Avg Delivery Days': '{:.1f}'
        }),
        use_container_width=True
    )

st.divider()
st.markdown("**Anuj Sherekar** · Data Analyst · Python · SQL · Scikit-learn · Streamlit · [GitHub](https://github.com/AnujS10)")