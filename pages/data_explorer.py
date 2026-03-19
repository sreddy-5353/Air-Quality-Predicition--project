import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

def show():
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the air quality dataset — distributions, trends, and feature correlations.")

    df = load_data()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", f"{len(df.columns) - 1}")
    c3.metric("Avg AQI", f"{df['AQI'].mean():.1f}")
    c4.metric("Max AQI", f"{df['AQI'].max():.0f}")

    st.markdown("---")
    tabs = st.tabs(["📋 Raw Data", "📉 Distributions", "🔥 Correlation", "📈 Trends"])

    with tabs[0]:
        st.dataframe(df.head(100), use_container_width=True, height=380)
        st.caption(f"Showing first 100 of {len(df):,} rows")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Statistics**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Feature", "Missing Count"]
            st.dataframe(missing, use_container_width=True)

    with tabs[1]:
        feature = st.selectbox("Select feature", df.columns.tolist())
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=feature, nbins=40, color_discrete_sequence=["#2563eb"],
                               title=f"Distribution of {feature}")
            fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df, y=feature, color_discrete_sequence=["#1e3a8a"],
                          title=f"Box Plot — {feature}")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                        title="Feature Correlation Heatmap", aspect="auto")
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with AQI
        if "AQI" in corr:
            st.markdown("**Top Features Correlated with AQI**")
            top = corr["AQI"].drop("AQI").abs().sort_values(ascending=False).head(6)
            fig2 = px.bar(x=top.index, y=top.values, color=top.values,
                          color_continuous_scale="Blues", labels={"x": "Feature", "y": "|Correlation|"})
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        if "Date" in df.columns or "date" in df.columns:
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            fig = px.line(df.sort_values(date_col), x=date_col, y="AQI",
                          title="AQI Over Time", color_discrete_sequence=["#2563eb"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Rolling AQI trend using index
            df_sorted = df.copy()
            df_sorted["Index"] = range(len(df_sorted))
            df_sorted["Rolling_AQI"] = df_sorted["AQI"].rolling(20).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sorted["Index"], y=df_sorted["AQI"],
                                     mode="lines", name="AQI", line=dict(color="#93c5fd", width=1)))
            fig.add_trace(go.Scatter(x=df_sorted["Index"], y=df_sorted["Rolling_AQI"],
                                     mode="lines", name="20-pt Rolling Avg", line=dict(color="#1e3a8a", width=2)))
            fig.update_layout(title="AQI Trend (Rolling Average)", plot_bgcolor="white",
                               paper_bgcolor="white", xaxis_title="Record Index", yaxis_title="AQI")
            st.plotly_chart(fig, use_container_width=True)

        # AQI category distribution
        bins = [0, 50, 100, 150, 200, 300, 500]
        labels = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
        colors = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7", "#7f1d1d"]
        df["AQI_Category"] = pd.cut(df["AQI"], bins=bins, labels=labels)
        cat_counts = df["AQI_Category"].value_counts().reindex(labels).dropna()
        fig3 = px.pie(values=cat_counts.values, names=cat_counts.index,
                      color_discrete_sequence=colors, title="AQI Category Distribution")
        fig3.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
