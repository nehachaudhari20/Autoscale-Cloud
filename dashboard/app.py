import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation.baselines import run_static_scaling, run_threshold_scaling
from evaluation.run_rl_policy import run_rl_policy
from evaluation.metrics_summary import compute_metrics


st.set_page_config(
    page_title="RL Cloud Autoscaling Platform",
    layout="wide"
)

st.title("Distributed RL Platform for Cloud Auto-Scaling")

st.markdown(
"""
This platform simulates **cloud autoscaling policies** and evaluates
Reinforcement Learning against traditional scaling strategies.
"""
)

# Sidebar
st.sidebar.header("Controls")

run_exp = st.sidebar.button("Run Experiment")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Latency",
    "Instance Scaling",
    "Cost Analysis"
])

if run_exp:

    st.sidebar.success("Running experiments...")

    static_lat, static_inst, static_cost = run_static_scaling()
    th_lat, th_inst, th_cost = run_threshold_scaling()
    rl_lat, rl_inst, rl_cost = run_rl_policy()

    # Metrics
    static_metrics = compute_metrics(static_lat, static_inst, static_cost)
    th_metrics = compute_metrics(th_lat, th_inst, th_cost)
    rl_metrics = compute_metrics(rl_lat, rl_inst, rl_cost)

    df_metrics = pd.DataFrame({
        "Strategy": ["Static", "Threshold", "RL"],
        "Avg Latency": [static_metrics[0], th_metrics[0], rl_metrics[0]],
        "Avg Instances": [static_metrics[1], th_metrics[1], rl_metrics[1]],
        "Total Cost": [static_metrics[2], th_metrics[2], rl_metrics[2]]
    })

    st.subheader("Performance Summary")

    st.dataframe(df_metrics)

    # Latency graph
    with tab1:

        df = pd.DataFrame({
            "Static": static_lat,
            "Threshold": th_lat,
            "RL": rl_lat
        })

        fig = px.line(df, title="Latency Comparison")

        st.plotly_chart(fig, use_container_width=True)

    # Instances graph
    with tab2:

        df = pd.DataFrame({
            "Static": static_inst,
            "Threshold": th_inst,
            "RL": rl_inst
        })

        fig = px.line(df, title="Instance Scaling")

        st.plotly_chart(fig, use_container_width=True)

    # Cost graph
    with tab3:

        df = pd.DataFrame({
            "Static": static_cost,
            "Threshold": th_cost,
            "RL": rl_cost
        })

        fig = px.line(df, title="Infrastructure Cost")

        st.plotly_chart(fig, use_container_width=True)

else:

    st.info("Click 'Run Experiment' in the sidebar to start.")