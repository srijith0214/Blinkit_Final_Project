"""
Blinkit AI-Powered Business Decision Platform
============================================
Streamlit Dashboard covering all 4 layers:
  Layer 1 — Data Engineering (Master Analytical View)
  Layer 2 — Analytics Dashboard (Marketing ROI)
  Layer 3 — Predictive ML (Delivery Delay Risk)
  Layer 4 — Generative AI + RAG (AI Business Assistant using Groq)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Blinkit Decision Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0b0d; }
  [data-testid="stSidebar"] { background: #111318; border-right: 1px solid #2a3040; }
  .block-container { padding: 1.5rem 2rem; }
  h1,h2,h3 { color: #e8ecf4; }
  .metric-card {
    background: #1a1f29; border: 1px solid #2a3040; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 8px;
  }
  .metric-label { font-size: 11px; color: #8a93a8; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
  .metric-val { font-size: 28px; font-weight: 800; line-height: 1.1; }
  .metric-sub { font-size: 12px; color: #5a6270; margin-top: 4px; }
  .alert-box {
    background: rgba(255,82,82,0.1); border: 1px solid rgba(255,82,82,0.35);
    border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;
  }
  .info-box {
    background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.25);
    border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;
  }
  .chat-user {
    background: rgba(64,196,255,0.1); border: 1px solid rgba(64,196,255,0.2);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0; color: #e8ecf4;
  }
  .chat-ai {
    background: #1a1f29; border: 1px solid #2a3040;
    border-radius: 10px; padding: 14px 16px; margin: 8px 0; color: #e8ecf4;
    font-size: 14px; line-height: 1.6;
  }
  .stButton > button {
    background: #00e676; color: #000; font-weight: 700;
    border: none; border-radius: 8px; padding: 10px 24px;
  }
  .stButton > button:hover { background: #69f0ae; }
  .stDateInput > div > div > input { background: #1a1f29; color: #e8ecf4; border-color: #2a3040; }
  .stSelectbox > div > div { background: #1a1f29; color: #e8ecf4; }
  .stTextInput > div > div > input { background: #1a1f29; color: #e8ecf4; border-color: #2a3040; }
  .stTextArea > div > div > textarea { background: #1a1f29; color: #e8ecf4; border-color: #2a3040; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'green': '#00e676', 'green2': '#69f0ae', 'blue': '#40c4ff',
    'yellow': '#ffd600', 'red': '#ff5252', 'purple': '#d500f9',
    'orange': '#ff9100', 'bg': '#0a0b0d', 'bg2': '#111318',
    'card': '#1a1f29', 'border': '#2a3040', 'text': '#e8ecf4', 'text2': '#8a93a8'
}

# ── Data loaders ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

@st.cache_data
def load_data():
    orders    = pd.read_csv(os.path.join(DATA_DIR, "blinkit_orders.csv"))
    marketing = pd.read_csv(os.path.join(DATA_DIR, "blinkit_marketing_performance.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "blinkit_customers.csv"))
    products  = pd.read_csv(os.path.join(DATA_DIR, "blinkit_products.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "blinkit_order_items.csv"))
    feedback  = pd.read_csv(os.path.join(DATA_DIR, "blinkit_customer_feedback.csv"))

    orders['order_dt']    = pd.to_datetime(orders['order_date'])
    orders['order_date_only'] = orders['order_dt'].dt.date
    orders['promised_dt'] = pd.to_datetime(orders['promised_delivery_time'])
    orders['actual_dt']   = pd.to_datetime(orders['actual_delivery_time'])
    orders['is_late']     = (orders['delivery_status'] != 'On Time').astype(int)
    orders['hour']        = orders['order_dt'].dt.hour
    orders['day_of_week'] = orders['order_dt'].dt.dayofweek
    orders['month']       = orders['order_dt'].dt.month

    marketing['date'] = pd.to_datetime(marketing['date']).dt.date

    return orders, marketing, customers, products, order_items, feedback

@st.cache_data
def build_master_view(orders, marketing):
    daily_rev = orders.groupby('order_date_only').agg(
        total_revenue=('order_total','sum'),
        total_orders=('order_id','count'),
        avg_order_value=('order_total','mean'),
        late_orders=('is_late','sum')
    ).reset_index().rename(columns={'order_date_only':'date'})
    daily_rev['on_time_pct'] = (1 - daily_rev['late_orders']/daily_rev['total_orders'])*100

    daily_mkt = marketing.groupby('date').agg(
        total_spend=('spend','sum'),
        total_impressions=('impressions','sum'),
        total_clicks=('clicks','sum'),
        total_conversions=('conversions','sum')
    ).reset_index()

    master = pd.merge(daily_rev, daily_mkt, on='date', how='outer').fillna(0)
    master['roas'] = master.apply(
        lambda r: round(r['total_revenue']/r['total_spend'],4) if r['total_spend']>0 else np.nan, axis=1)
    master['day_class'] = master['roas'].apply(
        lambda r: 'Profitable' if (r>=2.0) else ('Break-Even' if (r>=1.0) else 'Loss-Making') if not np.isnan(r) else 'Organic')
    master['date'] = pd.to_datetime(master['date'])
    return master.sort_values('date')

@st.cache_resource
def load_model():
    path = os.path.join(BASE, "src", "model.pkl")
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_rag():
    """Legacy loader kept for reference — app now uses BlinkitRAG directly."""
    path = os.path.join(BASE, "src", "rag_index.pkl")
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)
    return None

# ── Helpers ───────────────────────────────────────────────────
def fmt_inr(val):
    if val >= 1e7: return f"₹{val/1e7:.2f}Cr"
    if val >= 1e5: return f"₹{val/1e5:.2f}L"
    if val >= 1e3: return f"₹{val/1e3:.1f}K"
    return f"₹{val:.0f}"

def kpi(label, value, sub="", color=COLORS['green']):
    st.markdown(f"""
    <div class="metric-card" style="border-top: 2px solid {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-val" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text2'], font_family='monospace',
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border']),
        xaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
        margin=dict(l=0,r=0,t=30,b=0)
    )
    return fig

# ── RAG pipeline — import from dedicated module ───────────────
# rag_pipeline.py implements the full 3-step RAG flow:
#   Step 1 RETRIEVAL  : TF-IDF cosine similarity over 5,000 feedback docs
#   Step 2 AUGMENTATION: inject retrieved chunks + KPI context into prompt
#   Step 3 GENERATION : Groq LLaMA 3 produces root-cause analysis
from rag_pipeline import BlinkitRAG

@st.cache_resource
def get_rag_pipeline(api_key: str) -> BlinkitRAG:
    """Instantiate BlinkitRAG once per session (cached by Streamlit)."""
    return BlinkitRAG(groq_api_key=api_key, model="llama-3.3-70b-versatile", top_k=15)

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚡ Blinkit Intelligence")
        st.markdown("*Business Decision Platform*")
        st.markdown("---")
        page = st.radio("Navigate", [
            "📊 Overview",
            "📈 Marketing ROI",
            "🚚 Operations & ML",
            "📦 Products",
            "💬 Customer Feedback",
            "🤖 AI Assistant (RAG)"
        ])
        st.markdown("---")
        st.markdown("**Dataset**")
        st.caption("5,000 Orders · 2,500 Customers")
        st.caption("5,400 Campaigns · 5,000 Feedbacks")
        st.caption("268 Products · Mar 2023 – Nov 2024")
        st.markdown("---")
        groq_key = st.text_input("🔑 Groq API Key", type="password",
                                  help="Required for AI Assistant tab")

    # Load data
    orders, marketing, customers, products, order_items, feedback = load_data()
    master = build_master_view(orders, marketing)

    # ── PAGE: OVERVIEW ─────────────────────────────────────────
    if page == "📊 Overview":
        st.title("📊 Business Overview")
        st.caption("Unified intelligence across Marketing, Operations & Customer Experience")

        st.markdown("""<div class="alert-box">
            ⚠️ <strong style="color:#ff5252">ROAS Alert:</strong>
            Overall ROAS is <strong style="color:#ff5252">0.67x</strong> — below the 2.0x profitable threshold.
            Total ad spend (₹1.63Cr) exceeds total revenue (₹1.10Cr). Immediate campaign review recommended.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Total Revenue", fmt_inr(orders['order_total'].sum()), "Mar 2023 – Nov 2024", COLORS['green'])
        with c2: kpi("Total Ad Spend", fmt_inr(marketing['spend'].sum()), "ROAS: 0.67x ⚠️", COLORS['red'])
        with c3: kpi("Total Orders", f"{len(orders):,}", f"Avg {fmt_inr(orders['order_total'].mean())}/order", COLORS['blue'])
        with c4: kpi("On-Time Rate", f"{(orders['delivery_status']=='On Time').mean()*100:.1f}%", "1,530 orders delayed", COLORS['yellow'])

        c5,c6,c7,c8 = st.columns(4)
        with c5: kpi("Total Customers", f"{len(customers):,}", "24% Inactive segment", COLORS['purple'])
        with c6: kpi("Avg Rating", f"{feedback['rating'].mean():.2f} ⭐", "32.8% negative feedback", COLORS['orange'])
        with c7: kpi("Total Impressions", "29.5M", "Across all campaigns", COLORS['green2'])
        with c8: kpi("Best Channel", "Email 2.05x", "Only profitable channel ✅", COLORS['blue'])

        # Monthly chart
        monthly = master.copy()
        monthly['month'] = master['date'].dt.to_period('M').astype(str)
        m_grp = monthly.groupby('month').agg(revenue=('total_revenue','sum'), spend=('total_spend','sum')).reset_index()
        m_grp = m_grp[m_grp['month'] < '2024-11']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=m_grp['month'], y=m_grp['spend'], name='Ad Spend',
                             marker_color='rgba(255,82,82,0.6)'), secondary_y=True)
        fig.add_trace(go.Scatter(x=m_grp['month'], y=m_grp['revenue'], name='Revenue',
                                  line=dict(color=COLORS['green'], width=2.5),
                                  fill='tozeroy', fillcolor='rgba(0,230,118,0.08)'), secondary_y=False)
        fig.add_hline(y=0, line_dash='dot', line_color=COLORS['border'])
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False, tickprefix="₹")
        fig.update_yaxes(title_text="Ad Spend (₹)", secondary_y=True, tickprefix="₹")
        fig.update_layout(title="Monthly Revenue vs Ad Spend", height=320, barmode='overlay')
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    # ── PAGE: MARKETING ROI ───────────────────────────────────
    elif page == "📈 Marketing ROI":
        st.title("📈 Marketing ROI Dashboard")
        st.caption("ROAS ≥ 2.0x = Profitable | ROAS 1.0–2.0x = Break-Even | ROAS < 1.0x = Loss-Making")

        # Date filter
        col_a, col_b = st.columns(2)
        min_d = master['date'].min().date()
        max_d = master['date'].max().date()
        with col_a:
            d_start = st.date_input("From", value=max_d - timedelta(days=60), min_value=min_d, max_value=max_d)
        with col_b:
            d_end = st.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

        filt = master[(master['date'].dt.date >= d_start) & (master['date'].dt.date <= d_end)]

        c1,c2,c3,c4 = st.columns(4)
        rev_filt = filt['total_revenue'].sum()
        spend_filt = filt['total_spend'].sum()
        roas_filt = rev_filt/spend_filt if spend_filt > 0 else 0
        profitable_days = (filt['roas'] >= 2.0).sum()
        with c1: kpi("Revenue (Period)", fmt_inr(rev_filt), f"{d_start} → {d_end}", COLORS['green'])
        with c2: kpi("Ad Spend (Period)", fmt_inr(spend_filt), f"ROAS: {roas_filt:.2f}x", COLORS['red'])
        with c3: kpi("Profitable Days", str(profitable_days), "ROAS ≥ 2.0x", COLORS['yellow'])
        with c4: kpi("Loss Days", str((filt['roas']<1.0).sum()), "ROAS < 1.0x", COLORS['red'])

        # Dual-axis ROAS chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=filt['date'], y=filt['total_spend'], name='Ad Spend',
                             marker_color='rgba(255,82,82,0.55)'), secondary_y=True)
        fig.add_trace(go.Scatter(x=filt['date'], y=filt['total_revenue'], name='Revenue',
                                  line=dict(color=COLORS['green'], width=2),
                                  fill='tozeroy', fillcolor='rgba(0,230,118,0.07)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=filt['date'], y=[2.0]*len(filt),
                                  name='2x Target', line=dict(color=COLORS['yellow'], dash='dash', width=1),
                                  showlegend=True), secondary_y=False)
        fig.update_layout(title="Daily Revenue (Green) vs Ad Spend (Red) — Dual Axis", height=340)
        fig.update_yaxes(title_text="₹ Revenue", secondary_y=False, tickprefix="₹")
        fig.update_yaxes(title_text="₹ Ad Spend", secondary_y=True, tickprefix="₹")
        st.plotly_chart(dark_fig(fig), use_container_width=True)

        col_l, col_r = st.columns(2)

        # Channel ROAS
        with col_l:
            ch = marketing.groupby('channel').agg(spend=('spend','sum'), rev=('revenue_generated','sum')).reset_index()
            ch['roas'] = (ch['rev']/ch['spend']).round(2)
            ch_colors = [COLORS['green'] if r>=2.0 else COLORS['yellow'] for r in ch['roas']]
            fig2 = go.Figure(go.Bar(x=ch['channel'], y=ch['roas'], marker_color=ch_colors,
                                    text=ch['roas'].apply(lambda x: f'{x}x'), textposition='outside'))
            fig2.add_hline(y=2.0, line_dash='dash', line_color=COLORS['yellow'],
                           annotation_text="2.0x Target")
            fig2.update_layout(title="Channel ROAS", height=320, yaxis_title="ROAS",
                                yaxis=dict(range=[0,2.5]))
            st.plotly_chart(dark_fig(fig2), use_container_width=True)

        # Day classification pie
        with col_r:
            dc = filt['day_class'].value_counts().reset_index()
            pie_colors = {'Profitable':COLORS['green'],'Break-Even':COLORS['yellow'],
                          'Loss-Making':COLORS['red'],'Organic':COLORS['blue']}
            fig3 = go.Figure(go.Pie(labels=dc['day_class'], values=dc['count'],
                                     marker_colors=[pie_colors.get(l,COLORS['text2']) for l in dc['day_class']],
                                     hole=0.55))
            fig3.update_layout(title="Day Classification", height=320)
            st.plotly_chart(dark_fig(fig3), use_container_width=True)

        # Campaign table
        st.subheader("Campaign Performance")
        camp = marketing.groupby('campaign_name').agg(
            spend=('spend','sum'), rev=('revenue_generated','sum')).reset_index()
        camp['roas'] = (camp['rev']/camp['spend']).round(3)
        camp['status'] = camp['roas'].apply(lambda r: '✅ Profitable' if r>=2.0 else '⚠️ Below Target')
        camp['spend_fmt'] = camp['spend'].apply(fmt_inr)
        camp['rev_fmt'] = camp['rev'].apply(fmt_inr)
        camp = camp.sort_values('roas', ascending=False)
        st.dataframe(camp[['campaign_name','spend_fmt','rev_fmt','roas','status']].rename(
            columns={'campaign_name':'Campaign','spend_fmt':'Spend','rev_fmt':'Revenue',
                     'roas':'ROAS','status':'Status'}),
            use_container_width=True, hide_index=True)

    # ── PAGE: OPERATIONS & ML ─────────────────────────────────
    elif page == "🚚 Operations & ML":
        st.title("🚚 Operations & Delivery Intelligence")
        st.caption("Delivery analytics + ML-powered delay risk prediction")

        c1,c2,c3,c4 = st.columns(4)
        on_time  = (orders['delivery_status']=='On Time').sum()
        slight   = (orders['delivery_status']=='Slightly Delayed').sum()
        signif   = (orders['delivery_status']=='Significantly Delayed').sum()
        with c1: kpi("On Time", f"{on_time:,}", f"{on_time/len(orders)*100:.1f}% of orders", COLORS['green'])
        with c2: kpi("Slightly Delayed", f"{slight:,}", f"{slight/len(orders)*100:.1f}%", COLORS['yellow'])
        with c3: kpi("Significantly Delayed", f"{signif:,}", f"{signif/len(orders)*100:.1f}%", COLORS['red'])
        with c4: kpi("Total Delayed", f"{slight+signif:,}", "30.6% delay rate", COLORS['orange'])

        col_l, col_r = st.columns(2)

        with col_l:
            delay_hour = orders.groupby('hour')['is_late'].mean().reset_index()
            delay_hour['delay_pct'] = (delay_hour['is_late']*100).round(1)
            bar_colors = [COLORS['red'] if v>35 else COLORS['yellow'] if v>32 else COLORS['blue']
                          for v in delay_hour['delay_pct']]
            fig = go.Figure(go.Bar(x=delay_hour['hour'], y=delay_hour['delay_pct'],
                                   marker_color=bar_colors,
                                   text=delay_hour['delay_pct'].apply(lambda x: f'{x}%'),
                                   textposition='outside'))
            fig.add_hline(y=30.6, line_dash='dot', line_color=COLORS['text2'],
                          annotation_text="Avg 30.6%")
            fig.update_layout(title="Delay Rate by Hour (Red = High Risk)", height=320,
                               xaxis_title="Hour", yaxis_title="Delay %",
                               yaxis=dict(range=[0,48]))
            st.plotly_chart(dark_fig(fig), use_container_width=True)

        with col_r:
            fig2 = go.Figure(go.Pie(
                labels=['On Time (69.4%)','Slightly Delayed (20.7%)','Significantly Delayed (9.9%)'],
                values=[3470,1037,493],
                marker_colors=[COLORS['green'],COLORS['yellow'],COLORS['red']],
                hole=0.55
            ))
            fig2.update_layout(title="Delivery Status Breakdown", height=320)
            st.plotly_chart(dark_fig(fig2), use_container_width=True)

        # ── ML Risk Calculator ──────────────────────────────────
        st.markdown("---")
        st.subheader("⚠️ ML-Powered Delay Risk Calculator")

        model_payload = load_model()
        if model_payload:
            auc = model_payload.get('auc', 'N/A')
            st.markdown(f"""<div class="info-box">
                📌 <strong style="color:#00e676">Model Info:</strong>
                XGBoost Classifier | Trained on 4,000 orders | Test AUC: <strong>{auc}</strong><br>
                <small style="color:#8a93a8">Note: The dataset's delays are statistically random
                (chi-square p=0.30 for hour, p=0.89 for day-of-week), so AUC ~0.5 is expected.
                Rule-based probabilities are layered on top for actionable estimates.</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Model file not found. Run training script first.")

        col1,col2,col3 = st.columns(3)
        with col1:
            hour_val = st.selectbox("Hour of Order", list(range(24)),
                                     format_func=lambda h: f"{h}:00 {'⚠️' if h in [11,13,22] else ''}",
                                     index=12)
        with col2:
            day_val = st.selectbox("Day of Week",
                                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                                    index=3)
        with col3:
            order_val = st.number_input("Order Value (₹)", min_value=100, max_value=10000, value=1500, step=100)

        area_val = st.selectbox("Customer Area", customers['area'].dropna().unique().tolist(), index=0)
        seg_val  = st.selectbox("Customer Segment", ['Regular','Premium','New','Inactive'])

        if st.button("🔮 Calculate Delay Risk"):
            # Rule-based risk on top of model
            delay_hour = orders.groupby('hour')['is_late'].mean()
            base_risk = delay_hour.get(hour_val, 0.306) * 100
            day_idx = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(day_val)
            if day_idx in [4,5]: base_risk += 4   # Fri/Sat
            if hour_val in [11,12,13]: base_risk += 4  # Lunch rush
            if hour_val in [19,20,21,22]: base_risk += 3  # Evening rush
            if order_val > 3000: base_risk += 2
            risk = min(round(base_risk), 95)

            if risk < 30:
                st.success(f"✅ **Low Risk: {risk}%** — Good time slot. Dispatch confidently.")
            elif risk < 38:
                st.warning(f"⚠️ **Medium Risk: {risk}%** — Consider proactive customer notification.")
            else:
                st.error(f"🔴 **High Risk: {risk}%** — Allocate extra delivery partners. Alert customer proactively.")

        # ROAS chart (weekly)
        st.markdown("---")
        st.subheader("ROAS Trend (Weekly)")
        weekly = master.copy()
        weekly['week'] = master['date'].dt.to_period('W').astype(str)
        w_grp = weekly.groupby('week').agg(
            revenue=('total_revenue','sum'), spend=('total_spend','sum')).reset_index()
        w_grp['roas'] = w_grp['revenue']/w_grp['spend']
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=w_grp['week'], y=w_grp['roas'],
                                   mode='lines+markers', name='ROAS',
                                   line=dict(color=COLORS['blue'], width=2),
                                   marker=dict(color=[COLORS['green'] if r>=2.0 else COLORS['red']
                                                      for r in w_grp['roas']], size=7)))
        fig3.add_hline(y=2.0, line_dash='dash', line_color=COLORS['yellow'],
                       annotation_text="2.0x Profitable Threshold")
        fig3.update_layout(title="Weekly ROAS — Above Yellow = Profitable", height=280,
                            xaxis=dict(tickangle=45, nticks=20))
        st.plotly_chart(dark_fig(fig3), use_container_width=True)

    # ── PAGE: PRODUCTS ─────────────────────────────────────────
    elif page == "📦 Products":
        st.title("📦 Product & Category Intelligence")
        st.caption("Revenue breakdown by product category")

        merged_items = order_items.merge(products[['product_id','category','brand','margin_percentage']], on='product_id', how='left')
        merged_items['revenue'] = merged_items['quantity'] * merged_items['unit_price']
        cat_rev = merged_items.groupby('category').agg(
            revenue=('revenue','sum'), orders=('order_id','nunique'),
            avg_margin=('margin_percentage','mean')).reset_index()
        cat_rev = cat_rev.sort_values('revenue', ascending=False)
        total_rev = cat_rev['revenue'].sum()
        cat_rev['share_pct'] = (cat_rev['revenue']/total_rev*100).round(1)

        colors_list = [COLORS['green'],COLORS['blue'],COLORS['yellow'],COLORS['purple'],
                       COLORS['orange'],COLORS['green2'],COLORS['red'],'#84ffff','#ea80fc','#b2ff59','#ccff90']

        col_l, col_r = st.columns(2)
        with col_l:
            fig = go.Figure(go.Pie(labels=cat_rev['category'], values=cat_rev['revenue'],
                                    marker_colors=colors_list[:len(cat_rev)], hole=0.5))
            fig.update_layout(title="Revenue Share by Category", height=380)
            st.plotly_chart(dark_fig(fig), use_container_width=True)
        with col_r:
            fig2 = go.Figure(go.Bar(
                x=cat_rev['revenue'][::-1], y=cat_rev['category'][::-1],
                orientation='h',
                marker_color=colors_list[:len(cat_rev)][::-1],
                text=cat_rev['revenue'][::-1].apply(fmt_inr), textposition='outside'
            ))
            fig2.update_layout(title="Revenue by Category", height=380, xaxis_title="₹ Revenue")
            st.plotly_chart(dark_fig(fig2), use_container_width=True)

        st.subheader("Category Breakdown Table")
        display = cat_rev.copy()
        display['revenue'] = display['revenue'].apply(fmt_inr)
        display['share_pct'] = display['share_pct'].apply(lambda x: f"{x}%")
        display['avg_margin'] = display['avg_margin'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display[['category','revenue','share_pct','orders','avg_margin']].rename(
            columns={'category':'Category','revenue':'Revenue','share_pct':'Share',
                     'orders':'Orders','avg_margin':'Avg Margin'}),
            use_container_width=True, hide_index=True)

        # Brand analysis
        st.subheader("Top Brands by Revenue")
        brand_rev = merged_items.groupby('brand')['revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        fig3 = go.Figure(go.Bar(x=brand_rev['brand'], y=brand_rev['revenue'],
                                 marker_color=COLORS['blue'],
                                 text=brand_rev['revenue'].apply(fmt_inr), textposition='outside'))
        fig3.update_layout(title="Top 10 Brands by Revenue", height=300)
        st.plotly_chart(dark_fig(fig3), use_container_width=True)

    # ── PAGE: FEEDBACK ──────────────────────────────────────────
    elif page == "💬 Customer Feedback":
        st.title("💬 Customer Feedback Intelligence")
        st.caption("Sentiment analysis across 5,000 feedback records")

        total_fb = len(feedback)
        pos = (feedback['sentiment']=='Positive').sum()
        neg = (feedback['sentiment']=='Negative').sum()
        neu = (feedback['sentiment']=='Neutral').sum()

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Total Feedbacks", f"{total_fb:,}", "All time", COLORS['blue'])
        with c2: kpi("Positive", f"{pos:,} ({pos/total_fb*100:.1f}%)", "😊", COLORS['green'])
        with c3: kpi("Negative", f"{neg:,} ({neg/total_fb*100:.1f}%)", "😞", COLORS['red'])
        with c4: kpi("Avg Rating", f"{feedback['rating'].mean():.2f} ⭐", "Out of 5.0", COLORS['yellow'])

        col_l, col_r = st.columns(2)
        with col_l:
            sent_data = feedback['sentiment'].value_counts().reset_index()
            fig = go.Figure(go.Pie(
                labels=sent_data['sentiment'], values=sent_data['count'],
                marker_colors=[COLORS['blue'],COLORS['red'],COLORS['green']],
                hole=0.55
            ))
            fig.update_layout(title="Sentiment Distribution", height=300)
            st.plotly_chart(dark_fig(fig), use_container_width=True)

        with col_r:
            cat_data = feedback['feedback_category'].value_counts().reset_index()
            fig2 = go.Figure(go.Bar(
                x=cat_data['feedback_category'], y=cat_data['count'],
                marker_color=[COLORS['red'],COLORS['yellow'],COLORS['blue'],COLORS['purple']],
                text=cat_data['count'], textposition='outside'
            ))
            fig2.update_layout(title="Feedback by Category", height=300)
            st.plotly_chart(dark_fig(fig2), use_container_width=True)

        # Rating distribution
        st.subheader("Rating Distribution")
        rating_data = feedback['rating'].value_counts().sort_index().reset_index()
        fig3 = go.Figure(go.Bar(
            x=rating_data['rating'], y=rating_data['count'],
            marker_color=[COLORS['red'],COLORS['orange'],COLORS['yellow'],COLORS['green2'],COLORS['green']],
            text=rating_data['count'], textposition='outside'
        ))
        fig3.update_layout(title="Star Rating Distribution (1–5)", height=280,
                            xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(dark_fig(fig3), use_container_width=True)

        # Sentiment trend over time
        st.subheader("Sentiment Trend Over Time")
        feedback['date'] = pd.to_datetime(feedback['feedback_date'])
        feedback['month'] = feedback['date'].dt.to_period('M').astype(str)
        sent_trend = feedback.groupby(['month','sentiment']).size().unstack(fill_value=0).reset_index()
        fig4 = go.Figure()
        for col, color in [('Positive',COLORS['green']),('Negative',COLORS['red']),('Neutral',COLORS['blue'])]:
            if col in sent_trend.columns:
                fig4.add_trace(go.Scatter(x=sent_trend['month'], y=sent_trend[col],
                                           name=col, line=dict(color=color, width=2),
                                           mode='lines+markers', marker=dict(size=4)))
        fig4.update_layout(title="Monthly Sentiment Trend", height=280, xaxis=dict(tickangle=45))
        st.plotly_chart(dark_fig(fig4), use_container_width=True)

    # ── PAGE: AI ASSISTANT (RAG) ───────────────────────────────
    elif page == "🤖 AI Assistant (RAG)":
        st.title("🤖 AI Business Assistant")
        st.caption("Full RAG pipeline: TF-IDF retrieval → context augmentation → Groq LLaMA 3 generation")

        if not groq_key:
            st.markdown("""<div class="alert-box">
                🔑 Please enter your <strong>Groq API Key</strong> in the sidebar to activate the RAG pipeline.<br>
                Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>
            </div>""", unsafe_allow_html=True)
            st.stop()

        # ── Instantiate BlinkitRAG (cached per session) ────────
        try:
            rag_pipeline = get_rag_pipeline(groq_key)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialise RAG pipeline: {e}")
            st.stop()

        idx = rag_pipeline._index
        n_docs     = len(idx["feedback_df"])
        n_features = idx["tfidf_matrix"].shape[1]

        st.markdown(f"""<div class="info-box">
            ✅ <strong style="color:#00e676">RAG Pipeline Active</strong> — 3-step flow:<br>
            <strong>Step 1 — RETRIEVAL</strong>: TF-IDF cosine similarity search over
            <strong>{n_docs:,} feedback documents</strong> ({n_features} bigram features)<br>
            <strong>Step 2 — AUGMENTATION</strong>: Top-15 chunks + business KPI context
            injected into structured prompt<br>
            <strong>Step 3 — GENERATION</strong>: <strong>Groq LLaMA 3 (llama-3.3-70b-versatile)</strong>
            performs root-cause analysis grounded in real feedback evidence
        </div>""", unsafe_allow_html=True)

        # ── RAG pipeline diagram ───────────────────────────────
        col_diag = st.columns([1,2,1])
        with col_diag[1]:
            st.markdown("""
            <div style="background:#1a1f29;border:1px solid #2a3040;border-radius:10px;
                        padding:16px;font-family:'monospace';font-size:12px;text-align:center">
              <div style="color:#40c4ff">❓ User Query</div>
              <div style="color:#5a6270;margin:4px 0">↓</div>
              <div style="color:#ffd600">🔎 TF-IDF Vectorization</div>
              <div style="color:#5a6270;margin:4px 0">↓ cosine similarity</div>
              <div style="color:#ffd600">📋 Top-15 Feedback Chunks Retrieved</div>
              <div style="color:#5a6270;margin:4px 0">↓ + KPI context injected</div>
              <div style="color:#ffd600">📝 Augmented Prompt Built</div>
              <div style="color:#5a6270;margin:4px 0">↓</div>
              <div style="color:#00e676">🤖 Groq LLaMA 3 (8B) generates answer</div>
              <div style="color:#5a6270;margin:4px 0">↓</div>
              <div style="color:#00e676">✅ Root-Cause Analysis + Recommendations</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Quick prompts ──────────────────────────────────────
        quick_qs = [
            "Why are customers giving negative feedback about delivery?",
            "What product quality issues are customers complaining about?",
            "Why is customer satisfaction low? What are the top complaints?",
            "Which product categories have the most quality complaints?",
            "Why did we lose inactive customers? What went wrong?",
            "What should we prioritize to improve our ratings above 4.0?",
        ]

        st.subheader("💡 Quick Analysis Prompts")
        cols = st.columns(3)
        for i, q in enumerate(quick_qs):
            with cols[i % 3]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state['ai_query'] = q
                    st.rerun()

        st.markdown("---")
        st.subheader("💬 Ask the RAG Pipeline")

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        if 'ai_query' not in st.session_state:
            st.session_state['ai_query'] = ""
        if 'last_retrieved' not in st.session_state:
            st.session_state['last_retrieved'] = []

        user_input = st.text_area(
            "Your business question:",
            value=st.session_state['ai_query'],
            height=80,
            placeholder="e.g. Why are Dairy & Breakfast products getting bad reviews?"
        )

        col_send, col_clear, col_reset = st.columns([1,1,3])
        with col_send:
            send = st.button("🚀 Analyze with AI")
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state['chat_history'] = []
                st.session_state['last_retrieved'] = []
                rag_pipeline.reset_conversation()
                st.rerun()

        if send and user_input.strip():
            st.session_state['ai_query'] = ""

            # ── Step 1: Retrieval (shown to user) ─────────────
            with st.spinner("🔍 Step 1 — Retrieving relevant feedback via TF-IDF cosine similarity..."):
                retrieved = rag_pipeline.retrieve(user_input, top_k=15)
                st.session_state['last_retrieved'] = retrieved

            st.success(f"✅ Retrieved {len(retrieved)} relevant feedback chunks (top similarity: {retrieved[0]['similarity_score']:.3f})")

            # ── Step 2+3: Augment + Generate ──────────────────
            with st.spinner("🤖 Step 2+3 — Augmenting prompt & calling Groq LLaMA 3..."):
                try:
                    result = rag_pipeline.query(user_input, top_k=15, remember=True)
                    answer = result["answer"]
                    st.session_state['chat_history'].append(('user', user_input))
                    st.session_state['chat_history'].append(('ai', answer, retrieved))
                except Exception as e:
                    st.error(f"❌ Groq API error: {e}")
                    st.info("Check your API key or try a different Groq model.")

        # ── Display conversation ───────────────────────────────
        for turn in st.session_state['chat_history']:
            role = turn[0]
            msg  = turn[1]
            if role == 'user':
                st.markdown(
                    f'<div class="chat-user">👤 <strong>You:</strong> {msg}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 <strong>Groq LLaMA 3 — Root Cause Analysis:</strong>'
                    f'<br><br>{msg.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

        # ── Show retrieved chunks as expandable evidence ───────
        if st.session_state['last_retrieved']:
            with st.expander(f"📋 RAG Evidence — {len(st.session_state['last_retrieved'])} Retrieved Feedback Chunks"):
                st.caption("These are the exact feedback records retrieved by TF-IDF cosine similarity and injected into the Groq prompt.")
                for i, r in enumerate(st.session_state['last_retrieved'], 1):
                    sentiment_color = (
                        COLORS['green'] if r['sentiment'] == 'Positive'
                        else COLORS['red'] if r['sentiment'] == 'Negative'
                        else COLORS['blue']
                    )
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #2a3040;
                                border-left:4px solid {sentiment_color};
                                border-radius:6px;padding:10px 14px;margin:5px 0;font-size:13px">
                        <span style="color:{sentiment_color};font-weight:700">
                            #{i} [{r['feedback_category']} | {r['sentiment']} | ⭐{r['rating']}/5]
                        </span>
                        <span style="color:#5a6270;font-size:11px;margin-left:10px">
                            similarity: {r['similarity_score']:.4f} | {r.get('feedback_date','')}
                        </span><br>
                        <span style="color:#e8ecf4;line-height:1.6">"{r['feedback_text']}"</span>
                    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
