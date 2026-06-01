import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration

st.set_page_config(
    page_title="CLV Engine",
    layout="wide"
)

# Theme styling

PLOT_BG = "#111111"
PAPER_BG = "#111111"
FONT_COLOR = "white"
GRID_COLOR = "#444444"

# Segment color palette

segment_colors = {
    "Champions": "#00BFFF",
    "High Value": "#1F77B4",
    "Medium Value": "#FF7F0E",
    "Low Value": "#D62728"
}

# Streamlit custom styling

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.stMetric {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3, h4 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Load data

df = pd.read_csv(
    "data/features/clv_scored.csv",
    index_col=0
)

seg_summary = pd.read_csv(
    "outputs/reports/segment_summary.csv",
    index_col=0
)

# Sort segment order

segment_order = [
    "Champions",
    "High Value",
    "Medium Value",
    "Low Value"
]

df["clv_segment"] = pd.Categorical(
    df["clv_segment"],
    categories=segment_order,
    ordered=True
)

seg_summary["clv_segment"] = pd.Categorical(
    seg_summary.index,
    categories=segment_order,
    ordered=True
)

seg_summary = (
    seg_summary
    .reset_index(drop=True)
    .sort_values("clv_segment")
)

# Title section

st.title("Customer Lifetime Value Engine")

st.caption(
    "BG/NBD + Gamma-Gamma probabilistic CLV model"
)

# KPI row

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Customers Modelled",
    f"{len(df):,}"
)

c2.metric(
    "Total Forecast CLV (12m)",
    f"₹{df['clv_12m'].sum():,.0f}"
)

c3.metric(
    "Champions",
    f"{(df['clv_segment']=='Champions').sum():,}"
)

c4.metric(
    "At-Risk High Value",
    f"{((df['clv_segment'].isin(['Champions','High Value'])) & (df['prob_alive'] < 0.5)).sum():,}"
)

st.markdown("---")

# Charts section

col1, col2 = st.columns(2)

# CLV by segment

with col1:

    st.subheader(
        "Forecast Customer Value by Segment"
    )

    fig1 = px.bar(

        seg_summary,

        x="clv_segment",

        y="total_clv",

        color="clv_segment",

        text="clv_share_pct",

        color_discrete_map=segment_colors,

        labels={
            "clv_segment": "Customer Segment",
            "total_clv": "Forecast CLV"
        }
    )

    fig1.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    fig1.update_layout(

        plot_bgcolor=PLOT_BG,

        paper_bgcolor=PAPER_BG,

        font=dict(
            color=FONT_COLOR,
            size=14
        ),

        xaxis=dict(
            title="Customer Segment",
            gridcolor=GRID_COLOR,
            tickfont=dict(color=FONT_COLOR)
        ),

        yaxis=dict(
            title="Forecast CLV (₹)",
            gridcolor=GRID_COLOR,
            tickfont=dict(color=FONT_COLOR)
        ),

        showlegend=False
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# Probability alive boxplot

with col2:

    st.subheader(
        "Probability Customer Is Still Active"
    )

    fig2 = px.box(

        df,

        x="clv_segment",

        y="prob_alive",

        color="clv_segment",

        color_discrete_map=segment_colors,

        labels={
            "prob_alive": "Probability Alive"
        }
    )

    fig2.update_layout(

        plot_bgcolor=PLOT_BG,

        paper_bgcolor=PAPER_BG,

        font=dict(
            color=FONT_COLOR,
            size=14
        ),

        xaxis=dict(
            title="Customer Segment",
            gridcolor=GRID_COLOR,
            tickfont=dict(color=FONT_COLOR)
        ),

        yaxis=dict(
            title="Probability Alive",
            gridcolor=GRID_COLOR,
            tickfont=dict(color=FONT_COLOR)
        ),

        showlegend=False
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

# Scatter plot

st.subheader(
    "Customer Lifetime Value vs Purchase Frequency"
)

fig3 = px.scatter(

    df,

    x="frequency",

    y="clv_12m",

    color="clv_segment",

    color_discrete_map=segment_colors,

    hover_data=[
        "prob_alive",
        "monetary_value",
        "predicted_purchases_12w"
    ],

    labels={
        "frequency": "Repeat Purchases",
        "clv_12m": "12-Month CLV"
    }
)

fig3.update_traces(
    marker=dict(
        size=7,
        opacity=0.7
    )
)

fig3.update_layout(

    plot_bgcolor=PLOT_BG,

    paper_bgcolor=PAPER_BG,

    font=dict(
        color=FONT_COLOR,
        size=14
    ),

    xaxis=dict(
        title="Repeat Purchases",
        gridcolor=GRID_COLOR,
        tickfont=dict(color=FONT_COLOR)
    ),

    yaxis=dict(
        title="12-Month CLV",
        gridcolor=GRID_COLOR,
        tickfont=dict(color=FONT_COLOR),
        type="log"
    ),

    legend=dict(
        font=dict(color=FONT_COLOR)
    )
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.markdown("---")

# At-risk customers

st.subheader(
    "High-Value Customers at Risk of Churn"
)

at_risk = df[
    (
        df["clv_segment"].isin(
            ["Champions", "High Value"]
        )
    )
    &
    (
        df["prob_alive"] < 0.5
    )
]

st.dataframe(

    at_risk[
        [
            "clv_12m",
            "prob_alive",
            "frequency",
            "monetary_value",
            "predicted_purchases_12w"
        ]
    ]
    .sort_values(
        by="clv_12m",
        ascending=False
    )
    .head(20),

    use_container_width=True
)

st.markdown("---")

# ROI simulator

st.subheader(
    "Campaign ROI Simulator"
)

st.caption(
    "Estimate expected return from retention campaigns"
)

col1, col2, col3 = st.columns(3)

with col1:

    target_segment = st.selectbox(

        "Target Segment",

        [
            "Champions",
            "High Value",
            "Champions + High Value"
        ]
    )

with col2:

    cost_per_customer = st.slider(

        "Campaign Cost per Customer (₹)",

        50,
        5000,
        500,
        step=50
    )

with col3:

    retention_lift = st.slider(

        "Expected Retention Lift (%)",

        5,
        50,
        20
    )

# Segment filtering

if target_segment == "Champions + High Value":

    target = df[
        df["clv_segment"].isin(
            ["Champions", "High Value"]
        )
    ]

else:

    target = df[
        df["clv_segment"] == target_segment
    ]

# ROI calculations

n_customers = len(target)

total_campaign_cost = (
    n_customers * cost_per_customer
)

retained_extra = (
    target[
        target["prob_alive"] < 0.7
    ]["clv_12m"].sum()
)

incremental_revenue = (
    retained_extra *
    (retention_lift / 100)
)

net_roi = (
    incremental_revenue -
    total_campaign_cost
)

roi_pct = (
    (net_roi / total_campaign_cost) * 100
    if total_campaign_cost > 0
    else 0
)

# ROI metrics

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Customers Targeted",
    f"{n_customers:,}"
)

r2.metric(
    "Campaign Cost",
    f"₹{total_campaign_cost:,.0f}"
)

r3.metric(
    "Incremental Revenue",
    f"₹{incremental_revenue:,.0f}"
)

r4.metric(

    "Net ROI",

    f"₹{net_roi:,.0f}",

    delta=f"{roi_pct:.1f}%",

    delta_color=(
        "normal"
        if roi_pct > 0
        else "inverse"
    )
)

if roi_pct > 0:

    st.success(
        f"Projected positive ROI: {roi_pct:.1f}%"
    )

else:

    st.warning(
        "Campaign cost exceeds projected incremental revenue."
    )

st.markdown("---")

# Business insights

st.subheader(
    "Key Business Insights"
)

top_20pct = int(len(df) * 0.2)

df_sorted = df.sort_values(
    "clv_12m",
    ascending=False
)

top_share = (
    df_sorted.iloc[:top_20pct]["clv_12m"].sum()
    /
    df["clv_12m"].sum()
    * 100
)

st.info(

    f"Top 20% of customers contribute "
    f"{top_share:.1f}% of total forecast CLV. "
    f"High-value customers with low alive probability "
    f"represent the primary retention opportunity."
)
