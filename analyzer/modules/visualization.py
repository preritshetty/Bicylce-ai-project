import json5
import pandas as pd
import plotly.express as px
import streamlit as st


def extract_json(response: str):
    """Extract and validate JSON block from LLM response text"""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            return json5.loads(response[start:end])
    except Exception:
        return None
    return None


def prepare_dataframe(df: pd.DataFrame, x: str, y: str):
    """
    Prepare DataFrame for plotting if requested columns exist
    or if the query involves common derived metrics (counts, averages, sums, min, max, cancellations, time trends).
    """

    if "Departure_Date" in df.columns:
        df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")

    # ✅ Case 1: direct match
    if x in df.columns and y in df.columns:
        return df, x, y

    # ✅ Case 2: Counts
    if x in df.columns and any(term in y.lower() for term in ["count", "booking", "flight", "total"]):
        agg_df = df.groupby(x).size().reset_index(name="Booking_Count")
        return agg_df, x, "Booking_Count"

    # ✅ Case 3: Averages
    if x in df.columns:
        if "fare" in y.lower() and "avg" in y.lower():
            agg_df = df.groupby(x)["Fare_Amount"].mean().reset_index(name="Avg_Fare")
            return agg_df, x, "Avg_Fare"
        if "duration" in y.lower() and "avg" in y.lower():
            agg_df = df.groupby(x)["Duration_Hours"].mean().reset_index(name="Avg_Duration")
            return agg_df, x, "Avg_Duration"
        if "loyalty" in y.lower() and "avg" in y.lower():
            agg_df = df.groupby(x)["Loyalty_Points"].mean().reset_index(name="Avg_Loyalty")
            return agg_df, x, "Avg_Loyalty"

    # ✅ Case 4: Sums
    if x in df.columns:
        if "fare" in y.lower() and "sum" in y.lower():
            agg_df = df.groupby(x)["Fare_Amount"].sum().reset_index(name="Total_Fare")
            return agg_df, x, "Total_Fare"
        if "loyalty" in y.lower() and "sum" in y.lower():
            agg_df = df.groupby(x)["Loyalty_Points"].sum().reset_index(name="Total_Loyalty")
            return agg_df, x, "Total_Loyalty"

    # ✅ Case 5: Min / Max
    if x in df.columns:
        if "fare" in y.lower():
            if "max" in y.lower():
                agg_df = df.groupby(x)["Fare_Amount"].max().reset_index(name="Max_Fare")
                return agg_df, x, "Max_Fare"
            if "min" in y.lower():
                agg_df = df.groupby(x)["Fare_Amount"].min().reset_index(name="Min_Fare")
                return agg_df, x, "Min_Fare"
        if "duration" in y.lower():
            if "max" in y.lower():
                agg_df = df.groupby(x)["Duration_Hours"].max().reset_index(name="Max_Duration")
                return agg_df, x, "Max_Duration"
            if "min" in y.lower():
                agg_df = df.groupby(x)["Duration_Hours"].min().reset_index(name="Min_Duration")
                return agg_df, x, "Min_Duration"

    # ✅ Case 6: Cancellation Rate
    if x in df.columns and "cancel" in y.lower():
        total = df.groupby(x).size()
        cancelled = df[df["Booking_Status"] == "Cancelled"].groupby(x).size()
        rates = (cancelled / total * 100).fillna(0).reset_index(name="Cancellation_Rate")
        return rates, x, "Cancellation_Rate"

    # ✅ Case 7: Time-based trends
    if "month" in y.lower() or "month" in x.lower():
        df["Month"] = df["Departure_Date"].dt.to_period("M").astype(str)
        agg_df = df.groupby("Month").size().reset_index(name="Booking_Count")
        return agg_df, "Month", "Booking_Count"

    if "day" in y.lower() or "day" in x.lower():
        df["Day"] = df["Departure_Date"].dt.date
        agg_df = df.groupby("Day").size().reset_index(name="Booking_Count")
        return agg_df, "Day", "Booking_Count"

    if "week" in y.lower():
        df["Week"] = df["Departure_Date"].dt.to_period("W").astype(str)
        agg_df = df.groupby("Week").size().reset_index(name="Booking_Count")
        return agg_df, "Week", "Booking_Count"

    if "year" in y.lower():
        df["Year"] = df["Departure_Date"].dt.year
        agg_df = df.groupby("Year").size().reset_index(name="Booking_Count")
        return agg_df, "Year", "Booking_Count"

    return None, x, y


def recommend_chart_type(query: str) -> str:
    """Heuristic to pick a good default chart type based on query text."""
    q = query.lower()
    if any(word in q for word in ["trend", "time", "month", "day", "year", "over time"]):
        return "Line Chart"
    if any(word in q for word in ["distribution", "spread", "variance", "outlier"]):
        return "Box Plot"
    if any(word in q for word in ["share", "proportion", "percentage", "rate"]):
        return "Pie Chart"
    return "Bar Chart"


def plot_chart(spec: dict, df: pd.DataFrame, query: str = ""):
    """Render chart(s) with proof table and let user choose visualization."""
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    title = spec.get("title", "Visualization")

    # Prepare aggregated df
    df_to_plot, x, y = prepare_dataframe(df, x, y)
    if df_to_plot is None:
        st.warning(f"⚠️ The agent requested [{x}, {y}] which does not exist "
                   "and cannot be derived safely. Showing table only.")
        st.dataframe(df.head(15))
        return

    # Show proof table
    st.markdown("### 📑 Proof: Data used for visualization")
    try:
        st.dataframe(df_to_plot[[x, y]].head(15))
    except Exception:
        st.dataframe(df_to_plot.head(15))

    # Smart default
    default_chart = recommend_chart_type(query)

    chart_type = st.selectbox(
        "📊 Choose visualization type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"],
        index=["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"].index(default_chart),
        key=f"chart_selector_{query}",  # ✅ unique per query
    )

    # Render chart
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df_to_plot, x=x, y=y, color=color, title=title)
        elif chart_type == "Line Chart":
            fig = px.line(df_to_plot, x=x, y=y, color=color, title=title)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df_to_plot, x=x, y=y, color=color, title=title)
        elif chart_type == "Pie Chart":
            fig = px.pie(df_to_plot, names=x, values=y, title=title)
        elif chart_type == "Box Plot":
            fig = px.box(df_to_plot, x=x, y=y, color=color, title=title)
        else:
            st.warning("⚠️ Unsupported chart type.")
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart rendering failed: {str(e)}")
