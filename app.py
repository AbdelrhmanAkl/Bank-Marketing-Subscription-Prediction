import io
import math

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ============================================================
# BRANDING  (غيّر الاسم واللوجو حسب العميل)
# ============================================================
BRAND_NAME = "TermWise"
BRAND_TAGLINE = "Term deposit lead scoring"
SUPPORT_EMAIL = "support@yourcompany.com"

MODEL_REPO = "AbdelrahmanAkl/bank-marketing-subscription-model"
MODEL_FILE = "bank_marketing_final_model.joblib"
DEFAULT_THRESHOLD = 0.44
TEST_F1 = "47.86%"
TEST_AUC = "78.32%"

FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome",
]

JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

PRESETS = {
    "Custom": None,
    "Young professional": dict(age=29, job="technician", marital="single", education="tertiary",
                               default="no", balance=2400, housing="no", loan="no",
                               contact="cellular", day=12, month="mar", duration=320,
                               campaign=1, pdays=-1, previous=0, poutcome="unknown"),
    "Retired saver": dict(age=67, job="retired", marital="married", education="secondary",
                          default="no", balance=9800, housing="no", loan="no",
                          contact="telephone", day=5, month="oct", duration=540,
                          campaign=2, pdays=90, previous=2, poutcome="success"),
    "Indebted customer": dict(age=41, job="blue-collar", marital="married", education="primary",
                              default="yes", balance=-300, housing="yes", loan="yes",
                              contact="unknown", day=20, month="may", duration=60,
                              campaign=5, pdays=-1, previous=0, poutcome="unknown"),
}

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title=f"{BRAND_NAME} | Subscription Predictor",
                   page_icon="◆", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:wght@500;600&display=swap');

:root{
  --ink:#06241b; --muted:#5b6b63; --line:#d9e1db; --paper:#eef2ee; --card:#ffffff;
  --teal:#0b6b4f; --teal-soft:#dff2e8; --gold:#d4a73a; --gold-deep:#a97f16; --gold-soft:#f8edc9;
  --amber:#a97f16; --amber-soft:#f8edc9; --red:#b4443c; --red-soft:#f9e5e3;
  --blue:#2f5d50; --blue-soft:#e6efe9;
}
html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans', sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
#MainMenu, footer, [data-testid="stDecoration"] { visibility:hidden; }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1280px; }

h1,h2,h3,h4 { font-family:'IBM Plex Sans', sans-serif; letter-spacing:-0.01em; color:var(--ink); }

/* Sidebar */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#04251c 0%,#0a3d2e 100%); border-right:3px solid var(--gold); }
[data-testid="stSidebar"] * { color:#e8eef3; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }
.brand { font-family:'Newsreader', serif; font-size:34px; font-weight:600; margin-bottom:2px; color:var(--gold) !important; }
.brand-sub { font-size:13px; color:#9fb2c1 !important; margin-bottom:6px; }
.side-label { font-size:12px; color:#9fb2c1 !important; margin-bottom:2px; }
.side-value { font-size:16px; font-weight:600; margin-bottom:12px; color:#f0d58a !important; }

/* Header */
.hero { background:linear-gradient(120deg,#04251c 0%,#0b5a43 100%); border:1px solid #0b5a43;
        border-bottom:4px solid var(--gold); border-radius:12px; padding:30px 34px; margin-bottom:18px; }
.hero h1 { font-family:'Newsreader', serif; font-size:40px; font-weight:600; margin:0 0 8px 0; color:#fff; }
.hero p { color:#cfe3d8; font-size:16px; margin:0; max-width:720px; line-height:1.55; }

/* Panels */
.panel-title { font-size:15px; font-weight:600; margin:2px 0 10px 0; }
div[data-testid="stForm"] { background:var(--card); border:1px solid var(--line);
                            border-radius:10px; padding:22px 24px; }
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 16px; font-weight:500; }

/* Buttons */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
  border-radius:8px; font-weight:600; height:46px; }
div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#b98a1c,#e3b94d); border:1px solid #a97f16; color:#06241b; }
div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover { background:linear-gradient(90deg,#a97f16,#d4a73a); border-color:#8c6910; color:#06241b; }

/* Brand overrides for Streamlit red accents */
.stTabs [data-baseweb="tab"][aria-selected="true"] { color:var(--teal); }
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--gold) !important; height:3px; }
.stTabs [data-baseweb="tab"]:hover { color:var(--gold-deep); }
[data-baseweb="slider"] div[role="slider"] { background-color:var(--gold) !important; box-shadow:none !important; }
[data-baseweb="slider"] > div > div > div { background:linear-gradient(90deg,#0b6b4f,#d4a73a) !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color:var(--gold) !important; }
a { color:var(--gold-deep) !important; }
[data-testid="stSidebar"] a { color:var(--gold) !important; }

/* Result */
.result { background:var(--card); border:1px solid var(--line); border-top:4px solid var(--gold); border-radius:10px; padding:24px 28px; }
.badge { display:inline-block; padding:5px 12px; border-radius:6px; font-size:13px; font-weight:600; }
.b-high { background:var(--teal-soft); color:var(--teal); }
.b-mid  { background:var(--blue-soft); color:var(--blue); }
.b-nurt { background:var(--amber-soft); color:var(--amber); }
.b-low  { background:var(--red-soft); color:var(--red); }
.prob { font-family:'Newsreader', serif; font-size:68px; font-weight:600; color:var(--teal); line-height:1; margin:10px 0 4px 0; }
.small-muted { color:var(--muted); font-size:14px; line-height:1.5; }
.kv { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line); font-size:14px; }
.kv:last-child { border-bottom:none; }
.kv span:first-child { color:var(--muted); }
.action { border-left:4px solid var(--gold); background:var(--gold-soft); padding:12px 16px;
          border-radius:0 8px 8px 0; margin-top:14px; font-size:15px; line-height:1.5; }

/* KPI */
.kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
.kpi .l { color:var(--muted); font-size:13px; }
.kpi .v { font-size:28px; font-weight:600; margin-top:4px; color:var(--teal); }
.kpi { border-top:3px solid var(--gold); }

.legal { color:#8794a0; font-size:12.5px; text-align:center; margin-top:34px; line-height:1.6; }
@media (max-width:700px){ .hero h1{font-size:28px;} .prob{font-size:50px;} }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


try:
    model = load_model()
except Exception as e:
    st.error("The prediction model could not be loaded. Check your connection and refresh the page.")
    st.exception(e)
    st.stop()


# ============================================================
# HELPERS
# ============================================================
def tier_for(p: float, t: float):
    """Return (label, css class, recommended action)."""
    if p >= min(t + 0.25, 0.95):
        return ("Hot lead", "b-high",
                "Call within 24 hours. Offer the term deposit directly and assign an experienced agent.")
    if p >= t:
        return ("Qualified lead", "b-mid",
                "Include in the next outreach wave. Prepare a tailored term deposit offer.")
    if p >= t * 0.6:
        return ("Nurture", "b-nurt",
                "Keep warm with low-cost channels such as email or SMS, then review in 30 days.")
    return ("Low priority", "b-low",
            "Do not spend call time on this customer for now. Re-score if their profile changes.")


def gauge_svg(p: float, t: float) -> str:
    """Horizontal probability bar with a labelled threshold marker."""
    p = max(0.0, min(1.0, p))
    parts = [
        '<div style="position:relative;margin:26px 0 38px 0">',
        '<div style="height:16px;border-radius:8px;background:#e3e9e5;overflow:hidden">',
        f'<div style="width:{max(p*100, 1):.1f}%;height:100%;border-radius:8px;'
        'background:linear-gradient(90deg,#0b6b4f,#d4a73a)"></div></div>',
        f'<div style="position:absolute;left:{t*100:.1f}%;top:-6px;width:3px;height:28px;'
        'background:#06241b;border-radius:2px;transform:translateX(-50%)"></div>',
        f'<div style="position:absolute;left:{t*100:.1f}%;top:28px;transform:translateX(-50%);'
        f'font-size:12px;font-weight:600;color:#06241b;white-space:nowrap">Threshold {t:.0%}</div>',
        '</div>',
    ]
    return "".join(parts)


def apply_preset():
    preset = PRESETS.get(st.session_state.get("preset"))
    if preset:
        for k, v in preset.items():
            st.session_state[f"in_{k}"] = v


def score(df: pd.DataFrame):
    return model.predict_proba(df[FEATURES])[:, 1]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f'<div class="brand">{BRAND_NAME}</div><div class="brand-sub">{BRAND_TAGLINE}</div>',
                unsafe_allow_html=True)
    st.divider()

    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, DEFAULT_THRESHOLD, 0.01,
        help="Customers scoring at or above this value are marked as likely to subscribe. "
             "Lower it to reach more customers; raise it to focus on the surest leads.")
    st.caption("Tuned default: 44%")
    st.divider()

    st.markdown('<div class="side-label">Model</div><div class="side-value">Tuned Random Forest</div>'
                f'<div class="side-label">Test F1 score</div><div class="side-value">{TEST_F1}</div>'
                f'<div class="side-label">Test ROC-AUC</div><div class="side-value">{TEST_AUC}</div>',
                unsafe_allow_html=True)
    st.divider()
    st.caption(f"Need help? {SUPPORT_EMAIL}")


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="hero">
  <h1>Find the customers most likely to open a term deposit</h1>
  <p>Score a single customer in seconds, or upload your full contact list and get a ranked
     call list with a recommended action for every lead.</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Batch scoring", "About the model"])

# ============================================================
# TAB 1: SINGLE CUSTOMER
# ============================================================
with tab_single:
    st.selectbox("Load an example", list(PRESETS.keys()), key="preset", on_change=apply_preset,
                 help="Fill the form with a sample profile, then adjust any field.")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.form("customer_form"):
            st.markdown('<div class="panel-title">Customer profile</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 100, 35, key="in_age")
            job = c2.selectbox("Job", JOBS, key="in_job")
            marital = c1.selectbox("Marital status", ["married", "single", "divorced"], key="in_marital")
            education = c2.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"],
                                     key="in_education")

            st.markdown('<div class="panel-title" style="margin-top:14px">Finances</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            balance = c1.number_input("Account balance", value=1000, step=100, key="in_balance")
            default = c2.selectbox("Credit in default", ["no", "yes"], key="in_default")
            housing = c1.selectbox("Housing loan", ["no", "yes"], key="in_housing")
            loan = c2.selectbox("Personal loan", ["no", "yes"], key="in_loan")

            st.markdown('<div class="panel-title" style="margin-top:14px">Contact details</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            contact = c1.selectbox("Contact type", ["cellular", "telephone", "unknown"], key="in_contact")
            duration = c2.number_input("Call duration (seconds)", 0, value=180, step=10, key="in_duration")
            month = c1.selectbox("Last contact month", MONTHS, index=4, key="in_month")
            day = c2.number_input("Last contact day", 1, 31, 15, key="in_day")

            st.markdown('<div class="panel-title" style="margin-top:14px">Campaign history</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            campaign = c1.number_input("Contacts in this campaign", 1, value=1, key="in_campaign")
            previous = c2.number_input("Contacts before this campaign", 0, value=0, key="in_previous")
            pdays = c1.number_input("Days since previous contact (-1 = never)", -1, value=-1, key="in_pdays")
            poutcome = c2.selectbox("Previous campaign outcome",
                                    ["unknown", "failure", "other", "success"], key="in_poutcome")

            submitted = st.form_submit_button("Score this customer", type="primary",
                                              use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome}])
        try:
            st.session_state["result"] = float(score(row)[0])
        except Exception as e:
            st.error("The prediction failed. Please check the entered values.")
            st.exception(e)

    with right:
        p = st.session_state.get("result")
        if p is None:
            st.markdown("""
            <div class="result">
              <div class="panel-title">Result</div>
              <p class="small-muted">Fill in the customer profile and select
              <b>Score this customer</b>. The probability, lead tier and recommended next step
              will appear here.</p>
</div>""", unsafe_allow_html=True)
        else:
            label, css, action = tier_for(p, threshold)
            verdict = "Likely to subscribe" if p >= threshold else "Unlikely to subscribe"
            st.markdown(f"""
<div class="result">
<span class="badge {css}">{label}</span>
<div class="prob">{p:.1%}</div>
<div class="small-muted">Probability of subscribing to a term deposit</div>
{gauge_svg(p, threshold)}
<div class="kv"><span>Decision</span><span><b>{verdict}</b></span></div>
<div class="kv"><span>Threshold</span><span>{threshold:.0%}</span></div>
<div class="kv"><span>Distance from threshold</span><span>{(p - threshold)*100:+.1f} pts</span></div>
<div class="action"><b>Next step:</b> {action}</div>
</div>""", unsafe_allow_html=True)

# ============================================================
# TAB 2: BATCH
# ============================================================
with tab_batch:
    st.markdown('<div class="panel-title">Score a full customer list</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Upload a CSV with the 16 columns below. You will get every '
                'customer scored and ranked, ready to download.</p>', unsafe_allow_html=True)

    template = pd.DataFrame([PRESETS["Young professional"], PRESETS["Retired saver"]])[FEATURES]
    st.download_button("Download CSV template", template.to_csv(index=False).encode(),
                       "customer_template.csv", "text/csv")

    file = st.file_uploader("Customer list (CSV)", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception:
            st.error("This file could not be read as a CSV. Save it as CSV (comma separated) and try again.")
            st.stop()

        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing) + ". Download the template for the exact format.")
        else:
            try:
                probs = score(data)
            except Exception as e:
                st.error("Scoring failed. Check that categorical values match the template (for example job and month).")
                st.exception(e)
                st.stop()

            out = data.copy()
            out.insert(0, "probability", probs.round(4))
            out.insert(1, "decision", ["Likely" if x >= threshold else "Unlikely" for x in probs])
            out.insert(2, "lead_tier", [tier_for(x, threshold)[0] for x in probs])
            out = out.sort_values("probability", ascending=False).reset_index(drop=True)

            n, likely = len(out), int((probs >= threshold).sum())
            k1, k2, k3, k4 = st.columns(4)
            for col, lab, val in [(k1, "Customers scored", f"{n:,}"),
                                  (k2, "Likely to subscribe", f"{likely:,}"),
                                  (k3, "Share likely", f"{likely / n:.1%}"),
                                  (k4, "Average probability", f"{probs.mean():.1%}")]:
                col.markdown(f'<div class="kpi"><div class="l">{lab}</div><div class="v">{val}</div></div>',
                             unsafe_allow_html=True)

            st.write("")
            st.dataframe(out, use_container_width=True, height=380,
                         column_config={"probability": st.column_config.ProgressColumn(
                             "probability", min_value=0.0, max_value=1.0, format="%.2f")})
            st.download_button("Download ranked results", out.to_csv(index=False).encode(),
                               "scored_customers.csv", "text/csv", type="primary")

# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
        <div class="result">
          <div class="panel-title">How it works</div>
          <p class="small-muted">The model is a tuned Random Forest trained on historical bank
          marketing campaigns. For each customer it returns the probability of subscribing to a
          term deposit. A decision threshold turns that probability into a clear yes or no.</p>
          <p class="small-muted">The default threshold of 44% was chosen to balance catching real
          subscribers against wasting calls on customers who will decline. You can change it
          in the sidebar to match your campaign budget.</p>
        </div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""
        <div class="result">
          <div class="panel-title">Model quality (held-out test set)</div>
          <div class="kv"><span>Algorithm</span><span>Tuned Random Forest</span></div>
          <div class="kv"><span>F1 score</span><span>{TEST_F1}</span></div>
          <div class="kv"><span>ROC-AUC</span><span>{TEST_AUC}</span></div>
          <div class="kv"><span>Default threshold</span><span>{DEFAULT_THRESHOLD:.0%}</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="legal">
  {BRAND_NAME} provides statistical estimates to support marketing decisions. It does not guarantee
  individual outcomes and should not be the only basis for any decision about a customer.
</div>""", unsafe_allow_html=True)
