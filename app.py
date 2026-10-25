import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import streamlit as st
from predict import predict_with_probability

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Visual system
# -------------------------------------------------------------------
BG = "#0A0F1C"
PANEL = "#111827"
PANEL_ALT = "#151F30"
BORDER = "#263449"
TEXT = "#F3F6FC"
MUTED = "#9AA9BF"
CYAN = "#67E8F9"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#FB7185"
BLUE = "#60A5FA"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {{
      --bg: {BG};
      --panel: {PANEL};
      --panel-alt: {PANEL_ALT};
      --border: {BORDER};
      --text: {TEXT};
      --muted: {MUTED};
      --cyan: {CYAN};
    }}
    html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
    }}
    .stApp {{
      background:
        radial-gradient(ellipse at 80% 0%, rgba(34, 211, 238, 0.07), transparent 32%),
        var(--bg);
      color: var(--text);
    }}
    [data-testid="stHeader"] {{
      background: rgba(10, 15, 28, 0.86);
    }}
    [data-testid="stSidebar"] {{
      background: #0D1422;
      border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] > div:first-child {{
      padding-top: 1.5rem;
    }}
    h1, h2, h3, h4 {{
      font-family: 'Space Grotesk', sans-serif !important;
      color: var(--text) !important;
      letter-spacing: -0.035em;
    }}
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {{
      color: var(--text);
    }}
    .block-container {{
      max-width: 1500px;
      padding-top: 2.3rem;
      padding-bottom: 4rem;
    }}
    [data-testid="stMetric"] {{
      background: linear-gradient(145deg, #172236, #111827);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px 20px;
      min-height: 122px;
      box-shadow: 0 12px 28px rgba(0,0,0,.12);
    }}
    [data-testid="stMetricLabel"] p {{
      color: #A9B7CB !important;
      font-size: .78rem !important;
      font-weight: 600 !important;
      letter-spacing: .07em;
      text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{
      color: #F8FAFC !important;
      font-family: 'Space Grotesk', sans-serif;
      font-size: clamp(1.5rem, 2vw, 2rem) !important;
      font-weight: 600 !important;
    }}
    [data-testid="stMetricDelta"] {{
      font-size: .78rem !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
      border-color: var(--border);
      border-radius: 16px;
    }}
    div.stButton > button, div.stFormSubmitButton > button {{
      border-radius: 10px;
      border: 1px solid rgba(103,232,249,.35);
      background: linear-gradient(135deg, #22D3EE, #67E8F9);
      color: #07111D;
      font-weight: 700;
      padding: .65rem 1.1rem;
      transition: transform .15s ease, filter .15s ease;
    }}
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
      filter: brightness(1.07);
      transform: translateY(-1px);
      border-color: var(--cyan);
      color: #07111D;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {{
      background: #111827;
      border-color: #34445C;
      border-radius: 10px;
    }}
    [data-testid="stFileUploader"] {{
      background: #111827;
      border: 1px dashed #34445C;
      border-radius: 12px;
      padding: 10px;
    }}
    [data-testid="stExpander"] {{
      background: rgba(17,24,39,.7);
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    [data-testid="stDataFrame"] {{
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    hr {{
      border-color: var(--border);
      margin: 1.6rem 0;
    }}
    .eyebrow {{
      color: {CYAN};
      text-transform: uppercase;
      letter-spacing: .18em;
      font-size: .72rem;
      font-weight: 700;
      margin-bottom: .5rem;
    }}
    .hero-title {{
      font-family: 'Space Grotesk', sans-serif;
      font-size: clamp(2.2rem, 4vw, 3.7rem);
      line-height: 1.04;
      letter-spacing: -.055em;
      font-weight: 700;
      color: #F8FAFC;
      margin: 0;
    }}
    .hero-title span {{ color: {CYAN}; }}
    .hero-copy {{
      color: #A9B7CB;
      font-size: 1rem;
      line-height: 1.7;
      max-width: 740px;
      margin-top: .9rem;
    }}
    .section-kicker {{
      color: {MUTED};
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .16em;
      font-weight: 700;
      margin-bottom: .3rem;
    }}
    .insight-card {{
      background: linear-gradient(145deg, #151F30, #101725);
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 17px 18px;
      min-height: 112px;
    }}
    .insight-label {{
      color: {MUTED};
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .1em;
      font-weight: 700;
    }}
    .insight-value {{
      color: {TEXT};
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.5rem;
      font-weight: 600;
      margin-top: .45rem;
    }}
    .insight-note {{
      color: {MUTED};
      font-size: .8rem;
      margin-top: .2rem;
    }}
    .status-pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(103,232,249,.25);
      background: rgba(103,232,249,.08);
      color: {CYAN};
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
    }}
    .footer-note {{
      color: #73839B;
      font-size: .78rem;
      text-align: center;
      padding-top: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Data loading and validation
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes=None, file_name=None):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_csv("dataset.csv")

    df.columns = [str(col).strip() for col in df.columns]

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.drop(columns=[col for col in df.columns if "id" in col.lower()], errors="ignore")

    required = {"Churn", "Contract", "gender", "InternetService", "tenure", "MonthlyCharges"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))

    df = df.dropna(subset=["Churn", "Contract", "gender", "InternetService", "tenure", "MonthlyCharges"])
    return df


def style_chart(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(PANEL)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, labelpad=8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(axis="y", color=BORDER, linestyle="-", linewidth=.7, alpha=.7)
    ax.set_axisbelow(True)


def render_fig(fig, container=None):
    (container or st).pyplot(fig, use_container_width=True)
    plt.close(fig)


# -------------------------------------------------------------------
# Sidebar: data and filters
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:0 0 1rem 0;">
          <div style="font-size:.68rem;letter-spacing:.18em;color:{CYAN};font-weight:700;">CUSTOMER INTELLIGENCE</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;color:{TEXT};">Churn<span style="color:{CYAN};">IQ</span></div>
          <div style="color:{MUTED};font-size:.82rem;">Retention analytics workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Data source")
    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"], help="Use a CSV containing the same feature columns expected by this app.")
    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
    else:
        file_bytes = None
        file_name = None
        st.caption("Using the bundled dataset.csv")

try:
    df = load_data(file_bytes, file_name)
except FileNotFoundError:
    st.error("Could not find `dataset.csv`. Place it in the project folder or upload a CSV from the sidebar.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load this dataset: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("#### Segment filters")
    contract_options = sorted(df["Contract"].dropna().astype(str).unique().tolist())
    gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist())
    internet_options = sorted(df["InternetService"].dropna().astype(str).unique().tolist())

    contract_filter = st.multiselect("Contract type", options=contract_options, default=contract_options)
    gender_filter = st.multiselect("Gender", options=gender_options, default=gender_options)
    internet_filter = st.multiselect("Internet service", options=internet_options, default=internet_options)

    st.markdown("---")
    st.caption("Filters update the analytics across this workspace.")
    reset_filters = st.button("Reset filters", use_container_width=True)
    if reset_filters:
        st.rerun()

filtered_df = df[
    df["Contract"].astype(str).isin(contract_filter)
    & df["gender"].astype(str).isin(gender_filter)
    & df["InternetService"].astype(str).isin(internet_filter)
].copy()

# -------------------------------------------------------------------
# Hero
# -------------------------------------------------------------------
hero_left, hero_right = st.columns([4, 1])
with hero_left:
    st.markdown('<div class="eyebrow">CUSTOMER RETENTION ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Understand churn.<br><span>Act before customers leave.</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Explore customer segments, identify patterns associated with churn, and test individual customer profiles with your trained machine-learning model.</div>',
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="status-pill">● DATA CONNECTED</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='color:{MUTED};font-size:.78rem;margin-top:.6rem'>{len(df):,} rows loaded</div>", unsafe_allow_html=True)

st.markdown("")

# -------------------------------------------------------------------
# Overview metrics
# -------------------------------------------------------------------
st.markdown('<div class="section-kicker">01 / Portfolio overview</div>', unsafe_allow_html=True)
st.markdown("### Customer snapshot")

if filtered_df.empty:
    st.warning("No customers match the current filters. Adjust the sidebar filters to continue.")
    st.stop()

churn_counts = filtered_df["Churn"].astype(str).value_counts()
total_customers = len(filtered_df)
churned = int(churn_counts.get("Yes", 0))
churn_rate = churned / total_customers * 100 if total_customers else 0
avg_tenure = filtered_df["tenure"].mean()
avg_monthly = filtered_df["MonthlyCharges"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers analyzed", f"{total_customers:,}", help="Customers remaining after the selected filters.")
m2.metric("Observed churn rate", f"{churn_rate:.1f}%", help="Share of filtered records where Churn is Yes.")
m3.metric("Average tenure", f"{avg_tenure:.1f} mo", help="Mean customer tenure in months.")
m4.metric("Avg. monthly charge", f"${avg_monthly:,.2f}", help="Mean monthly charge in the filtered dataset.")

st.markdown("")

# -------------------------------------------------------------------
# Overview charts
# -------------------------------------------------------------------
st.markdown('<div class="section-kicker">02 / Customer composition</div>', unsafe_allow_html=True)
st.markdown("### Where churn and customers cluster")

chart1, chart2 = st.columns([1, 1.25])

with chart1:
    fig, ax = plt.subplots(figsize=(5, 3.7))
    labels = ["Stayed", "Churned"]
    values = [int(churn_counts.get("No", 0)), int(churn_counts.get("Yes", 0))]
    if sum(values):
        wedges, _, _ = ax.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=[GREEN, RED],
            autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
            pctdistance=.68,
            wedgeprops={"width": .32, "edgecolor": PANEL, "linewidth": 3},
            textprops={"color": TEXT, "fontsize": 9, "weight": "bold"},
        )
        ax.text(0, .06, f"{churn_rate:.1f}%", ha="center", va="center", color=TEXT, fontsize=20, fontweight="bold")
        ax.text(0, -.17, "churn rate", ha="center", va="center", color=MUTED, fontsize=8)
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, labelcolor=MUTED, fontsize=9)
    style_chart(ax, "Churn distribution")
    ax.set_aspect("equal")
    render_fig(fig)

with chart2:
    contract_churn = pd.crosstab(filtered_df["Contract"], filtered_df["Churn"], normalize="index") * 100
    for col in ["No", "Yes"]:
        if col not in contract_churn.columns:
            contract_churn[col] = 0
    contract_churn = contract_churn[["No", "Yes"]]
    fig, ax = plt.subplots(figsize=(6, 3.7))
    contract_churn.plot(kind="barh", stacked=True, ax=ax, color=[GREEN, RED], width=.58, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of customers (%)", color=MUTED)
    ax.set_ylabel("")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(["Stayed", "Churned"], frameon=False, labelcolor=MUTED, loc="lower center", bbox_to_anchor=(.5, -.28), ncol=2)
    style_chart(ax, "Churn mix by contract", xlabel="Share of customers (%)")
    ax.grid(axis="x", color=BORDER, linewidth=.7)
    ax.grid(axis="y", visible=False)
    render_fig(fig)

# -------------------------------------------------------------------
# Segment insights
# -------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="section-kicker">03 / Retention signals</div>', unsafe_allow_html=True)
st.markdown("### Explore the customer base")

seg1, seg2 = st.columns(2)
with seg1:
    contract_counts = filtered_df["Contract"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(contract_counts.index.astype(str), contract_counts.values, color=BLUE, height=.58)
    ax.bar_label(bars, labels=[f"{v:,}" for v in contract_counts.values], padding=5, color=TEXT, fontsize=9)
    style_chart(ax, "Customers by contract", xlabel="Customer count")
    ax.grid(axis="x", color=BORDER)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(contract_counts.values) * 1.18 if len(contract_counts) else 1)
    render_fig(fig)

with seg2:
    service_counts = filtered_df["InternetService"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(service_counts.index.astype(str), service_counts.values, color=[CYAN, BLUE, AMBER][:len(service_counts)], height=.58)
    ax.bar_label(bars, labels=[f"{v:,}" for v in service_counts.values], padding=5, color=TEXT, fontsize=9)
    style_chart(ax, "Internet service mix", xlabel="Customer count")
    ax.grid(axis="x", color=BORDER)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(service_counts.values) * 1.18 if len(service_counts) else 1)
    render_fig(fig)

dist1, dist2 = st.columns(2)
with dist1:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(filtered_df["MonthlyCharges"].dropna(), bins=24, color=CYAN, alpha=.88, edgecolor=PANEL, linewidth=.6)
    style_chart(ax, "Monthly charge distribution", xlabel="Monthly charges ($)", ylabel="Customers")
    render_fig(fig)

with dist2:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(filtered_df["tenure"].dropna(), bins=24, color=BLUE, alpha=.88, edgecolor=PANEL, linewidth=.6)
    style_chart(ax, "Tenure distribution", xlabel="Tenure (months)", ylabel="Customers")
    render_fig(fig)

# -------------------------------------------------------------------
# Correlations
# -------------------------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) >= 2:
    with st.expander("Explore numeric feature correlations", expanded=False):
        corr_df = filtered_df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(7, len(numeric_cols) * .85), max(4.5, len(numeric_cols) * .65)))
        sns.heatmap(
            corr_df,
            annot=len(numeric_cols) <= 10,
            fmt=".2f",
            cmap=sns.diverging_palette(210, 5, as_cmap=True),
            center=0,
            linewidths=.5,
            linecolor=PANEL,
            ax=ax,
            cbar_kws={"shrink": .8},
        )
        ax.set_title("Feature correlation matrix", color=TEXT, loc="left", pad=14, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.figure.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        render_fig(fig)

# -------------------------------------------------------------------
# Model performance centre
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-kicker">04 / Model diagnostics</div>',
    unsafe_allow_html=True,
)
st.markdown("### Model performance centre")
st.markdown(
    "Held-out evaluation results from the most recent training run."
)

metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
importance_path = Path(__file__).resolve().parent / "feature_importance.csv"

if metrics_path.exists():
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        metrics = None
        st.warning(f"Could not read model evaluation results: {exc}")

    if metrics is not None:
        st.caption(
            f"Model: {metrics.get('model', 'Unknown')} · "
            f"{metrics.get('test_rows', 0):,} held-out test rows · "
            f"{metrics.get('feature_count', 0)} features"
        )

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        p2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        p3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        p4.metric("F1-score", f"{metrics.get('f1_score', 0):.3f}")

        roc_auc = metrics.get("roc_auc")
        p5.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")

        left, right = st.columns([1, 1.15])

        with left:
            st.markdown("#### Confusion matrix")
            cm = metrics.get("confusion_matrix", [])

            if len(cm) == 2 and all(len(row) == 2 for row in cm):
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap=sns.color_palette(
                        ["#172236", "#164E63", "#0E7490", "#67E8F9"],
                        as_cmap=True,
                    ),
                    linewidths=1,
                    linecolor=PANEL,
                    cbar=False,
                    xticklabels=["Predicted stay", "Predicted churn"],
                    yticklabels=["Actual stay", "Actual churn"],
                    annot_kws={"size": 13, "weight": "bold"},
                    ax=ax,
                )
                ax.set_xlabel("Model prediction", color=MUTED)
                ax.set_ylabel("Actual outcome", color=MUTED)
                ax.set_facecolor(PANEL)
                fig.set_facecolor(PANEL)
                ax.tick_params(colors=MUTED)
                render_fig(fig)

                tn, fp = cm[0]
                fn, tp = cm[1]
                st.caption(
                    f"Correctly identified churners: {tp:,} · "
                    f"Missed churners: {fn:,} · "
                    f"False alarms: {fp:,} · "
                    f"Correctly identified stayers: {tn:,}"
                )
            else:
                st.warning("The saved confusion matrix has an unexpected format.")

        with right:
            st.markdown("#### Most influential features")

            if importance_path.exists():
                try:
                    importance_df = pd.read_csv(importance_path)
                except (OSError, pd.errors.ParserError) as exc:
                    importance_df = pd.DataFrame()
                    st.warning(f"Could not read feature importances: {exc}")

                if {"Feature", "Importance"}.issubset(importance_df.columns):
                    top = importance_df.nlargest(10, "Importance").sort_values(
                        "Importance"
                    )

                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    bars = ax.barh(
                        top["Feature"],
                        top["Importance"],
                        color=CYAN,
                        height=0.65,
                    )
                    ax.bar_label(
                        bars,
                        fmt="%.3f",
                        padding=4,
                        color=TEXT,
                        fontsize=8,
                    )
                    style_chart(ax, "Global Random Forest feature importance")
                    ax.grid(axis="x", color=BORDER)
                    ax.grid(axis="y", visible=False)
                    ax.set_xlim(0, max(float(top["Importance"].max()) * 1.2, 0.01))
                    render_fig(fig)

                    st.caption(
                        "Feature importance describes the model's overall use of "
                        "features; it does not establish causation or explain an "
                        "individual prediction."
                    )
                else:
                    st.info(
                        "Feature importance data is missing or has an unexpected format. "
                        "Run train.py to regenerate feature_importance.csv."
                    )
            else:
                st.info("Run train.py to generate feature_importance.csv.")

        with st.expander("Interpret these metrics"):
            st.markdown(
                """
                - **Accuracy:** proportion of test predictions that were correct.
                - **Precision:** proportion of predicted churners who actually churned.
                - **Recall:** proportion of actual churners detected by the model.
                - **F1-score:** harmonic mean of precision and recall.
                - **ROC-AUC:** ability to rank positive cases above negative cases
                  across classification thresholds.

                These metrics describe the saved held-out evaluation, not guaranteed
                future performance. Feature importance is not causal evidence.
                """
            )
else:
    st.info(
        "Model evaluation results are not available yet. Run "
        "`python train.py` to generate model_metrics.json."
    )

# -------------------------------------------------------------------
# Threshold optimization and held-out evaluation curves
# -------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-kicker">05 / Threshold optimization</div>', unsafe_allow_html=True)
st.markdown("### Threshold optimization lab")
st.markdown(
    "Explore how the decision threshold changes precision, recall, F1-score, and the number of missed churners. "
    "All curves below use the saved held-out test predictions generated by `train.py`."
)

predictions_path = Path(__file__).resolve().parent / "test_predictions.csv"
threshold_metrics_path = Path(__file__).resolve().parent / "threshold_metrics.csv"

if predictions_path.exists():
    try:
        evaluation_df = pd.read_csv(predictions_path)
        required_eval_columns = {"actual_churn", "churn_probability"}
        if not required_eval_columns.issubset(evaluation_df.columns):
            raise ValueError("test_predictions.csv is missing required columns.")
        evaluation_df = evaluation_df.dropna(subset=["actual_churn", "churn_probability"])
        evaluation_df["actual_churn"] = evaluation_df["actual_churn"].astype(int)
        evaluation_df["churn_probability"] = evaluation_df["churn_probability"].astype(float).clip(0, 1)

        if len(evaluation_df) and evaluation_df["actual_churn"].nunique() == 2:
            threshold = st.slider(
                "Evaluation threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="evaluation_threshold_slider",
                help="Predicted churn when estimated churn probability is greater than or equal to this threshold.",
                format="%.2f",
            )
            actual = evaluation_df["actual_churn"].to_numpy()
            probabilities = evaluation_df["churn_probability"].to_numpy()
            predicted = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            precision = precision_score(actual, predicted, zero_division=0)
            recall = recall_score(actual, predicted, zero_division=0)
            f1 = f1_score(actual, predicted, zero_division=0)
            accuracy = accuracy_score(actual, predicted)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy:.1%}")
            m2.metric("Precision", f"{precision:.1%}")
            m3.metric("Recall", f"{recall:.1%}")
            m4.metric("F1-score", f"{f1:.3f}")

            left, right = st.columns(2)
            with left:
                st.markdown("#### Confusion matrix at selected threshold")
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.heatmap(
                    [[tn, fp], [fn, tp]], annot=True, fmt="d", cbar=False,
                    cmap=sns.color_palette(["#172236", "#164E63", "#0E7490", "#67E8F9"], as_cmap=True),
                    linewidths=1, linecolor=PANEL,
                    xticklabels=["Predicted stay", "Predicted churn"],
                    yticklabels=["Actual stay", "Actual churn"], ax=ax,
                    annot_kws={"size": 13, "weight": "bold"},
                )
                style_chart(ax, f"Held-out test set · threshold {threshold:.0%}")
                render_fig(fig)
                st.caption(f"Detected churners: {tp:,} · Missed churners: {fn:,} · False alarms: {fp:,} · Correct stay predictions: {tn:,}")

            with right:
                st.markdown("#### Precision vs recall by threshold")
                if threshold_metrics_path.exists():
                    curve_df = pd.read_csv(threshold_metrics_path)
                    if {"threshold", "precision", "recall", "f1_score"}.issubset(curve_df.columns):
                        fig, ax = plt.subplots(figsize=(5.5, 4.5))
                        ax.plot(curve_df["threshold"], curve_df["precision"], marker="o", label="Precision", linewidth=2, color=CYAN)
                        ax.plot(curve_df["threshold"], curve_df["recall"], marker="o", label="Recall", linewidth=2, color=AMBER)
                        ax.plot(curve_df["threshold"], curve_df["f1_score"], marker="o", label="F1-score", linewidth=2, color=GREEN)
                        ax.axvline(threshold, linestyle="--", linewidth=1.5, color=TEXT, label=f"Selected {threshold:.0%}")
                        ax.set_ylim(0, 1.05)
                        ax.set_xlabel("Decision threshold")
                        ax.set_ylabel("Score")
                        ax.legend(frameon=False, fontsize=8)
                        style_chart(ax, "Metric trade-offs across thresholds")
                        render_fig(fig)
                    else:
                        st.info("Threshold metrics file has an unexpected format. Rerun train.py.")
                else:
                    st.info("Run train.py to create threshold_metrics.csv.")

            with st.expander("ROC curve and interpretation"):
                fpr, tpr, _ = roc_curve(actual, probabilities)
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(fpr, tpr, linewidth=2.5, color=CYAN, label="ROC curve")
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=MUTED, label="Random ranking baseline")
                auc_value = metrics.get("roc_auc") if metrics_path.exists() and "metrics" in locals() and metrics else None
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate (recall)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.legend(frameon=False)
                style_chart(ax, f"ROC curve{f' · AUC {auc_value:.3f}' if isinstance(auc_value, (int, float)) else ''}")
                render_fig(fig)
                st.markdown(
                    "The threshold changes the predicted class, not the saved probability. Lower thresholds usually increase recall and may increase false positives; higher thresholds usually reduce false positives and may miss more churners. "
                    "Choose a threshold based on retention-team capacity and the relative costs of missed churners versus unnecessary outreach."
                )

            st.download_button(
                "Download held-out predictions",
                data=evaluation_df.to_csv(index=False).encode("utf-8"),
                file_name="churniq_test_predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("The saved test predictions need both churn and stay examples. Rerun train.py with a suitable dataset.")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load evaluation predictions: {exc}")
else:
    st.info("Run `python train.py` to generate held-out predictions and threshold evaluation files.")

# -------------------------------------------------------------------
# Customer prediction lab
# -------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-kicker">06 / Individual customer analysis</div>', unsafe_allow_html=True)
st.markdown("### Customer risk lab")
st.markdown(
    "Build a customer profile and evaluate it with the project's existing trained model. The result is a model prediction, not a guarantee of future behavior."
)

with st.container(border=True):
    with st.form("predict_form"):
        st.markdown("#### Customer profile")
        st.caption("Adjust the profile fields and decision threshold, then run the model.")

        threshold = st.slider(
            "Churn decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help=(
                "The model flags churn when its estimated churn probability is at least "
                "this threshold. Lower thresholds usually catch more churners but can "
                "create more false alarms."
            ),
            format="%.2f",
        )

        a, b, c = st.columns(3)
        with a:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0, format="%.2f")
            total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=850.0, step=10.0, format="%.2f")
            gender = st.selectbox("Gender", ["Female", "Male"])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
        with b:
            PhoneService = st.selectbox("Phone service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c:
            TechSupport = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        submitted = st.form_submit_button("Run churn prediction", use_container_width=True)

if submitted:
    customer_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
    }

    try:
        churn_prediction, suggestions, churn_probability = predict_with_probability(customer_data, threshold=threshold)
        st.markdown("#### Prediction result")
        result_col, probability_col, threshold_col = st.columns(3)
        with result_col:
            if str(churn_prediction).upper() == "YES":
                st.error("**Higher churn risk**\n\nThe selected threshold has been met.")
            else:
                st.success("**Lower churn risk**\n\nThe selected threshold has not been met.")
        with probability_col:
            st.metric("Estimated churn probability", f"{churn_probability:.1%}")
            st.progress(float(max(0.0, min(1.0, churn_probability))))
        with threshold_col:
            st.metric("Decision threshold", f"{threshold:.0%}")
            st.caption(
                "Predicted churn when probability ≥ threshold."
            )

        if churn_probability >= threshold:
            st.warning(
                "This profile crosses the current threshold. Consider reviewing "
                "contract, tenure, support and security-service options before deciding "
                "on a retention action."
            )
        else:
            st.info(
                "This profile is below the current threshold. That does not guarantee "
                "the customer will stay; use the probability and business context together."
            )

        with st.expander("See how the threshold changes the decision"):
            threshold_rows = []
            for candidate_threshold in [0.25, 0.35, 0.50, 0.65, 0.75]:
                threshold_rows.append({
                    "Threshold": f"{candidate_threshold:.0%}",
                    "Decision": "Flag churn" if churn_probability >= candidate_threshold else "Do not flag",
                })
            st.dataframe(
                pd.DataFrame(threshold_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "This comparison changes only the decision threshold for this customer. "
                "It does not recalculate test-set precision or recall."
            )

        st.markdown("#### Suggested retention actions")
        if suggestions:
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.info("The model did not return any suggested actions for this profile.")
        with st.expander("View submitted customer profile"):
            st.json(customer_data)
        st.caption(
            "Probability is the model's estimated score, not necessarily a calibrated "
            "real-world probability. Threshold selection should be validated against "
            "held-out data and business costs."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that model.pkl, columns.pkl, and the expected feature columns are present and compatible with predict.py.")

# -------------------------------------------------------------------
# Data explorer and footer
# -------------------------------------------------------------------
with st.expander("Inspect filtered customer data", expanded=False):
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} loaded records after applying filters.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_filtered_data.csv",
        mime="text/csv",
    )

with st.expander("How this application works", expanded=False):
    st.markdown(
        """
        1. Load the bundled dataset or upload a compatible CSV.
        2. Filter the customer base by contract, gender, and internet service.
        3. Explore churn distribution, customer segments, tenure, charges, and numeric correlations.
        4. Submit an individual customer profile to the existing trained prediction model.
        5. Explore held-out evaluation curves and compare threshold trade-offs.
        6. Review an individual customer probability, threshold-based prediction, and suggested actions.

        **Model limitation:** this dashboard displays the prediction output provided by the current model interface. It does not claim a calibrated probability, causal explanation, or business impact estimate unless those are separately implemented and validated.
        """
    )

st.markdown(
    '<div class="footer-note">ChurnIQ · Customer retention analytics · Predictions are decision support, not guarantees.</div>',
    unsafe_allow_html=True,
)
