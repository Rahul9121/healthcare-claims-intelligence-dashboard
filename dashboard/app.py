from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Healthcare Claims Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    return {
        "utilization": pd.read_csv(base_dir / "dashboard_utilization_monthly.csv"),
        "cost_drivers": pd.read_csv(base_dir / "dashboard_cost_drivers.csv"),
        "provider": pd.read_csv(base_dir / "dashboard_provider_performance.csv"),
        "cohort": pd.read_csv(base_dir / "dashboard_cohort_analysis.csv"),
        "member_rollup": pd.read_csv(base_dir / "dashboard_member_rollup.csv"),
    }


try:
    data = load_dashboard_data()
except FileNotFoundError:
    st.error("Dashboard datasets not found. Run: `py -m src.pipeline`")
    st.stop()

utilization = data["utilization"]
cost_drivers = data["cost_drivers"]
provider = data["provider"]
cohort = data["cohort"]
member_rollup = data["member_rollup"]

st.title("Healthcare Claims Intelligence Dashboard")
st.caption(
    "Synthetic claims analytics for utilization, cost drivers, provider performance, and cohort risk."
)

latest_month = utilization.sort_values("service_month").iloc[-1]
total_paid_all = float(utilization["total_paid"].sum())
total_claims_all = int(utilization["total_claims"].sum())
overall_readmission = float(member_rollup["readmission_rate"].mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Paid (All Months)", f"${total_paid_all:,.0f}")
col2.metric("Total Claims", f"{total_claims_all:,}")
col3.metric("Avg Readmission Rate", f"{overall_readmission:.2f}%")
col4.metric("Latest Month", str(latest_month["service_month"]))

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Utilization Trend",
        "Cost Drivers",
        "Provider Performance",
        "Cohort Intelligence",
    ]
)

with tab1:
    left, right = st.columns(2)
    with left:
        fig_paid = px.line(
            utilization,
            x="service_month",
            y="total_paid",
            title="Monthly Paid Amount Trend",
            markers=True,
            labels={"service_month": "Service Month", "total_paid": "Total Paid"},
        )
        st.plotly_chart(fig_paid, use_container_width=True)
    with right:
        fig_readmit = px.line(
            utilization,
            x="service_month",
            y="readmission_rate",
            title="Monthly Readmission Rate Trend",
            markers=True,
            labels={"service_month": "Service Month", "readmission_rate": "Readmission Rate (%)"},
        )
        st.plotly_chart(fig_readmit, use_container_width=True)

with tab2:
    top_cost = cost_drivers.sort_values("total_paid", ascending=False).head(10)
    fig_cost = px.bar(
        top_cost,
        x="diagnosis_group",
        y="total_paid",
        color="readmission_rate",
        title="Top Cost Drivers by Diagnosis Group",
        labels={
            "diagnosis_group": "Diagnosis Group",
            "total_paid": "Total Paid",
            "readmission_rate": "Readmission Rate (%)",
        },
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.dataframe(
        top_cost[
            ["diagnosis_group", "claim_count", "avg_paid_per_claim", "readmission_rate", "total_paid"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    fig_provider = px.scatter(
        provider,
        x="avg_paid_per_claim",
        y="readmission_rate",
        size="claim_count",
        color="provider_specialty",
        hover_data=["provider_name"],
        title="Provider Cost vs Readmission Performance",
        labels={
            "avg_paid_per_claim": "Average Paid per Claim",
            "readmission_rate": "Readmission Rate (%)",
            "claim_count": "Claim Count",
            "provider_specialty": "Specialty",
        },
    )
    st.plotly_chart(fig_provider, use_container_width=True)
    st.dataframe(
        provider.sort_values("readmission_rate").head(20),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    cohort_matrix = (
        cohort.groupby(["plan_type", "chronic_band"], as_index=False)
        .agg(avg_readmission_rate=("avg_readmission_rate", "mean"), members=("members", "sum"))
    )
    fig_heat = px.density_heatmap(
        cohort_matrix,
        x="plan_type",
        y="chronic_band",
        z="avg_readmission_rate",
        text_auto=".2f",
        title="Cohort Readmission Heatmap (Plan vs Chronic Burden)",
        labels={
            "plan_type": "Plan Type",
            "chronic_band": "Chronic Burden Band",
            "avg_readmission_rate": "Avg Readmission Rate (%)",
        },
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(
        cohort.sort_values("avg_readmission_rate", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )
