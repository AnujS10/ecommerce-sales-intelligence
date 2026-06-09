import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Sales Intelligence",
    page_icon="📦",
    layout="wide"
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    kpis       = pd.read_csv(os.path.join(BASE, 'data', 'kpis.csv'))
    monthly    = pd.read_csv(os.path.join(BASE, 'data', 'monthly_revenue.csv'))
    by_state   = pd.read_csv(os.path.join(BASE, 'data', 'revenue_by_state.csv'))
    category   = pd.read_csv(os.path.join(BASE, 'data', 'category_performance.csv'))
    delivery   = pd.read_csv(os.path.join(BASE, 'data', 'delivery_performance.csv'))
    payments   = pd.read_csv(os.path.join(BASE, 'data', 'payment_analysis.csv'))
    del_review = pd.read_csv(os.path.join(BASE, 'data', 'review_vs_delivery.csv'))
    by_day     = pd.read_csv(os.path.join(BASE, 'data', 'orders_by_day.csv'))
    by_hour    = pd.read_csv(os.path.join(BASE, 'data', 'orders_by_hour.csv'))
    return kpis, monthly, by_state, category, delivery, payments, del_review, by_day, by_hour

kpis, monthly, by_state, category, delivery, payments, del_review, by_day, by_hour = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📦 E-Commerce Sales Intelligence Dashboard")
st.markdown("**100,000+ real orders · Olist Brazil · 2016–2018**")
st.divider()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k = kpis.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue",     f"R$ {k['total_revenue']:,.0f}")
col2.metric("📦 Total Orders",      f"{int(k['total_orders']):,}")
col3.metric("⭐ Avg Review Score",  f"{k['avg_review']} / 5.0")
col4.metric("🚚 Avg Delivery Days", f"{k['avg_delivery_days']} days")

st.divider()

# ── SECTION 1: SALES TREND ────────────────────────────────────────────────────
st.subheader("📈 Monthly Revenue Trend")
fig1 = px.bar(monthly, x='year_month', y='revenue',
    labels={'year_month': 'Month', 'revenue': 'Revenue (R$)'},
    color='revenue', color_continuous_scale='Blues')
fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── SECTION 2: REGIONAL ANALYSIS ─────────────────────────────────────────────
st.subheader("🗺️ Revenue by State (Top 10)")
col1, col2 = st.columns(2)

with col1:
    fig2 = px.bar(by_state.head(10), x='state', y='total_revenue',
        labels={'state': 'State', 'total_revenue': 'Revenue (R$)'},
        color='total_revenue', color_continuous_scale='Teal')
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.bar(by_state.head(10), x='state', y='total_orders',
        labels={'state': 'State', 'total_orders': 'Total Orders'},
        color='total_orders', color_continuous_scale='Blues')
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── SECTION 3: CATEGORY PERFORMANCE ──────────────────────────────────────────
st.subheader("🛍️ Product Category Performance")
fig4 = px.scatter(category,
    x='avg_review_score', y='total_revenue',
    size='total_orders', color='avg_review_score',
    hover_name='category',
    labels={'avg_review_score': 'Avg Review Score', 'total_revenue': 'Revenue (R$)'},
    color_continuous_scale='RdYlGn')
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── SECTION 4: DELIVERY INTELLIGENCE ─────────────────────────────────────────
st.subheader("🚚 Delivery Intelligence")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Delivery Days by State (Slowest)**")
    fig5 = px.bar(delivery.head(15), x='state', y='actual_delivery_days',
        color='actual_delivery_days', color_continuous_scale='Reds',
        labels={'state': 'State', 'actual_delivery_days': 'Avg Days'})
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("**Delivery Time vs Review Score**")
    fig6 = px.bar(del_review, x='review_score', y='avg_delivery_days',
        color='avg_delivery_days', color_continuous_scale='RdYlGn_r',
        labels={'review_score': 'Review Score', 'avg_delivery_days': 'Avg Delivery Days'})
    fig6.update_layout(showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── SECTION 5: CUSTOMER BEHAVIOUR ─────────────────────────────────────────────
st.subheader("👥 Customer Behaviour")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Payment Methods**")
    fig7 = px.pie(payments, names='payment_type', values='total_orders',
        color_discrete_sequence=px.colors.sequential.Blues_r)
    fig7.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.markdown("**Orders by Day of Week**")
    fig8 = px.bar(by_day, x='day', y='orders',
        color='orders', color_continuous_scale='Blues')
    fig8.update_layout(showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

with col3:
    st.markdown("**Orders by Hour of Day**")
    fig9 = px.line(by_hour, x='hour', y='orders', markers=True)
    fig9.update_traces(line_color='#1F3864')
    st.plotly_chart(fig9, use_container_width=True)

st.divider()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("Built by **Anuj Sherekar** · Python · SQL · Pandas · Plotly · Streamlit")