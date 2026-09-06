"""
visualization.py
----------------
Plotly chart builders for the Streamlit UI.
All functions return Plotly figures that Streamlit renders with st.plotly_chart().

Kept simple: 3-4 core charts that add genuine insight, not decoration.
"""

import plotly.graph_objects as go
import plotly.express as px
import json
from src.utils import load_metrics


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

PHISHING_RED = "#FF4B4B"
LEGITIMATE_GREEN = "#00CC66"
WARNING_ORANGE = "#FFA500"
NEUTRAL_BLUE = "#4B8BFF"
CHART_BG = "rgba(0,0,0,0)"  # transparent background for Streamlit dark mode


def make_gauge_chart(score: float, title: str, color: str = None) -> go.Figure:
    """
    Gauge/speedometer chart showing a 0-100 score.
    Used for both the ML confidence and heuristic risk score.
    """
    if color is None:
        if score >= 70:
            color = PHISHING_RED
        elif score >= 40:
            color = WARNING_ORANGE
        else:
            color = LEGITIMATE_GREEN

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 15], "color": "#e8f5e9"},
                {"range": [15, 40], "color": "#fff9c4"},
                {"range": [40, 70], "color": "#ffe0b2"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=CHART_BG,
    )
    return fig


def make_ml_probability_chart(phishing_prob: float, legitimate_prob: float) -> go.Figure:
    """
    Horizontal bar chart comparing ML phishing vs legitimate probability.
    Simple and intuitive for a demo.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Phishing",
        x=[round(phishing_prob * 100, 1)],
        y=["ML Classifier"],
        orientation="h",
        marker_color=PHISHING_RED,
        text=[f"{phishing_prob:.1%}"],
        textposition="inside",
        insidetextanchor="middle",
    ))

    fig.add_trace(go.Bar(
        name="Legitimate",
        x=[round(legitimate_prob * 100, 1)],
        y=["ML Classifier"],
        orientation="h",
        marker_color=LEGITIMATE_GREEN,
        text=[f"{legitimate_prob:.1%}"],
        textposition="inside",
        insidetextanchor="middle",
    ))

    fig.update_layout(
        title="ML Classifier Probability Breakdown",
        barmode="stack",
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        height=140,
        margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        legend=dict(orientation="h", y=-0.3),
        showlegend=True,
    )
    return fig


def make_indicator_category_chart(text_categories: dict, url_categories: dict) -> go.Figure:
    """
    Grouped bar chart showing heuristic indicator scores by category.
    Helps visualize WHERE the risk is coming from.
    """
    # Friendly labels for category keys
    category_labels = {
        "urgency": "Urgency/Pressure",
        "credential": "Credential Theft",
        "reward": "Reward/Bait",
        "threat": "Threat Language",
        "cta": "Suspicious CTA",
        "formatting": "Formatting Anomaly",
        "structure": "URL Structure",
        "domain": "Domain Anomaly",
        "keywords": "Suspicious Keywords",
    }

    all_cats = set(list(text_categories.keys()) + list(url_categories.keys()))

    if not all_cats:
        fig = go.Figure()
        fig.add_annotation(
            text="No indicators triggered",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=200, paper_bgcolor=CHART_BG)
        return fig

    cats = sorted(all_cats)
    labels = [category_labels.get(c, c.title()) for c in cats]
    text_vals = [text_categories.get(c, 0) for c in cats]
    url_vals = [url_categories.get(c, 0) for c in cats]

    fig = go.Figure()

    if any(v > 0 for v in text_vals):
        fig.add_trace(go.Bar(
            name="Text Indicators",
            x=labels,
            y=text_vals,
            marker_color=WARNING_ORANGE,
        ))

    if any(v > 0 for v in url_vals):
        fig.add_trace(go.Bar(
            name="URL Indicators",
            x=labels,
            y=url_vals,
            marker_color=PHISHING_RED,
        ))

    fig.update_layout(
        title="Heuristic Indicator Breakdown by Category",
        barmode="group",
        xaxis_title="Indicator Category",
        yaxis_title="Score Points",
        height=300,
        margin=dict(l=10, r=10, t=50, b=60),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        legend=dict(orientation="h", y=-0.35),
    )
    return fig


def make_score_comparison_chart(ml_score: float, heuristic_score: float, final_score: float) -> go.Figure:
    """
    Radar / bar chart showing the three scores side by side.
    Visualises how the two layers combine into the final verdict.
    """
    fig = go.Figure()

    categories = ["ML Score", "Heuristic Score", "Final Score"]
    values = [ml_score, heuristic_score, final_score]
    colors = []
    for v in values:
        if v >= 70:
            colors.append(PHISHING_RED)
        elif v >= 40:
            colors.append(WARNING_ORANGE)
        else:
            colors.append(LEGITIMATE_GREEN)

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
    ))

    # 50-point threshold line
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#333",
        annotation_text="Decision Threshold (50)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Score Comparison: ML vs Heuristics vs Final",
        yaxis=dict(range=[0, 110], title="Score (0-100)"),
        height=300,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        showlegend=False,
    )
    return fig


def make_confusion_matrix_chart(metrics: dict) -> go.Figure:
    """
    Display saved confusion matrix from model training.
    Only shown in the 'Model Info' section.
    """
    cm = metrics.get("confusion_matrix")
    if not cm:
        fig = go.Figure()
        fig.add_annotation(
            text="No confusion matrix available. Train the model first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="gray"),
        )
        fig.update_layout(height=200, paper_bgcolor=CHART_BG)
        return fig

    # cm is [[TN, FP], [FN, TP]]
    z = cm
    x_labels = ["Predicted: Legitimate", "Predicted: Phishing"]
    y_labels = ["Actual: Legitimate", "Actual: Phishing"]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#e8f5e9"], [1, "#FF4B4B"]],
        text=z,
        texttemplate="%{text}",
        showscale=False,
    ))

    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        height=280,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor=CHART_BG,
    )
    return fig
