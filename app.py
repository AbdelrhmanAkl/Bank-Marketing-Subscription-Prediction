
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load(
        "models/bank_marketing_threshold.joblib"
    )
    return model

model = load_model()

THRESHOLD = 0.44


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Section titles */
    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        text-align: center;
        min-height: 120px;
    }

    .result-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 700;
    }

    /* Prediction result */
    .prediction-success {
        padding: 18px;
        border-radius: 12px;
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        text-align: center;
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
    }

    .prediction-negative {
        padding: 18px;
        border-radius: 12px;
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        text-align: center;
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/bank_marketing_threshold.joblib"
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

THRESHOLD = 0.44


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Bank Marketing"
    )

    st.caption(
        "Subscription Prediction System"
    )

    st.divider()

    st.markdown(
        "### Model"
    )

    st.write(
        "Tuned Random Forest"
    )

    st.markdown(
        "### Decision Threshold"
    )

    st.write(
        f"{THRESHOLD:.0%}"
    )

    st.divider()

    st.markdown(
        "### Model Performance"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "F1",
            "47.86%"
        )

    with metric_col2:

        st.metric(
            "ROC-AUC",
            "78.32%"
        )

    st.divider()

    st.markdown(
        "### About"
    )

    st.write(
        """
        This application predicts whether a bank customer
        is likely to subscribe to a term deposit based on
        customer and campaign information.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Bank Marketing Subscription Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine learning powered customer subscription prediction '
    'using a tuned Random Forest model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Customer Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("#### Personal Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        [
            "married",
            "single",
            "divorced"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )


with col2:

    st.markdown("#### Financial Information")

    default = st.selectbox(
        "Credit in Default",
        [
            "no",
            "yes"
        ]
    )

    balance = st.number_input(
        "Account Balance",
        value=1000,
        step=100
    )

    housing = st.selectbox(
        "Housing Loan",
        [
            "yes",
            "no"
        ]
    )

    loan = st.selectbox(
        "Personal Loan",
        [
            "yes",
            "no"
        ]
    )


with col3:

    st.markdown("#### Contact Information")

    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "unknown"
        ]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15,
        step=1
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=180,
        step=10
    )


# ============================================================
# CAMPAIGN INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Campaign Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    campaign = st.number_input(
        "Contacts During Current Campaign",
        min_value=1,
        value=1,
        step=1
    )


with col2:

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1,
        value=-1,
        step=1
    )


with col3:

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        value=0,
        step=1
    )


poutcome = st.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "other",
        "success"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    predict = st.button(
        "Predict Subscription",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    new_customer = pd.DataFrame(
        {
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "balance": [balance],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "day": [day],
            "month": [month],
            "duration": [duration],
            "campaign": [campaign],
            "pdays": [pdays],
            "previous": [previous],
            "poutcome": [poutcome]
        }
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        probability = model.predict_proba(
            new_customer
        )[0, 1]

        prediction = probability >= THRESHOLD

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Result Cards
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Subscription Probability
                </div>
                <div class="result-value">
                    {probability:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with result_col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Decision Threshold
                </div>
                <div class="result-value">
                    {THRESHOLD:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with result_col3:

        decision_text = (
            "Likely to Subscribe"
            if prediction
            else
            "Unlikely to Subscribe"
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Final Decision
                </div>
                <div class="result-value">
                    {decision_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Probability Visualization
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        "#### Subscription Probability"
    )

    st.progress(
        min(float(probability), 1.0)
    )

    st.caption(
        f"Model probability: {probability:.2%}"
    )


    # --------------------------------------------------------
    # Final Decision
    # --------------------------------------------------------

    if prediction:

        st.markdown(
            """
            <div class="prediction-success">
                Customer is likely to subscribe.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="prediction-negative">
                Customer is unlikely to subscribe.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Bank Marketing Subscription Predictor
        | Tuned Random Forest
        | Decision Threshold: 0.44
        | Test F1: 47.86%
        | Test ROC-AUC: 78.32%
    </div>
    """,
    unsafe_allow_html=True
)
