import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="ASSB Conductivity Predictor",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Solid-State Electrolyte Conductivity Predictor")

st.markdown("""
Predict ionic conductivity of solid-state electrolytes using a trained XGBoost model.

### Supported Elements
Li, Na, O, S, P, F, Cl

### Output
- Predicted log₁₀(Ionic Conductivity)
- Predicted Ionic Conductivity (S/cm)
""")

# =====================================================
# Load Model
# =====================================================

try:
    model = joblib.load("conductivity_model.pkl")
except Exception as e:
    st.error(f"Failed to load conductivity_model.pkl\n\n{e}")
    st.stop()

# =====================================================
# Formula Input
# =====================================================

st.header("1. Chemical Formula")

formula = st.text_input(
    "Enter Chemical Formula",
    placeholder="Example: Li6PS5Cl"
)

# 初始状态不显示任何内容
if formula.strip() == "":
    st.info("Please enter a chemical formula.")
    st.stop()

# =====================================================
# Parse Formula
# =====================================================

try:

    comp = Composition(formula)

    element_dict = comp.get_el_amt_dict()

    Li_amt = element_dict.get("Li", 0.0)
    Na_amt = element_dict.get("Na", 0.0)
    O_amt = element_dict.get("O", 0.0)
    S_amt = element_dict.get("S", 0.0)
    P_amt = element_dict.get("P", 0.0)
    F_amt = element_dict.get("F", 0.0)
    Cl_amt = element_dict.get("Cl", 0.0)

    n_elements = len(element_dict)

except Exception:

    st.error("Invalid chemical formula.")
    st.stop()

# =====================================================
# Composition Table
# =====================================================

st.header("2. Parsed Composition")

df_elements = pd.DataFrame(
    element_dict.items(),
    columns=["Element", "Amount"]
)

st.dataframe(
    df_elements,
    use_container_width=True
)

# =====================================================
# Unsupported Elements Check
# =====================================================

supported = {"Li", "Na", "O", "S", "P", "F", "Cl"}

unsupported = [
    elem for elem in element_dict.keys()
    if elem not in supported
]

if len(unsupported) > 0:
    st.warning(
        "Unsupported elements detected: "
        + ", ".join(unsupported)
        + "\n\nCurrent model only uses Li, Na, O, S, P, F and Cl."
    )

# =====================================================
# Crystal Structure Input
# =====================================================

st.header("3. Crystal Structure Parameters")

col1, col2 = st.columns(2)

with col1:

    sg = st.number_input(
        "Space Group Number",
        min_value=1,
        max_value=230,
        value=None,
        placeholder="e.g. 216"
    )

    a = st.number_input(
        "a (Å)",
        value=None,
        placeholder="e.g. 9.85"
    )

    b = st.number_input(
        "b (Å)",
        value=None,
        placeholder="e.g. 9.85"
    )

    c = st.number_input(
        "c (Å)",
        value=None,
        placeholder="e.g. 9.85"
    )

with col2:

    alpha = st.number_input(
        "α (degree)",
        value=None,
        placeholder="90"
    )

    beta = st.number_input(
        "β (degree)",
        value=None,
        placeholder="90"
    )

    gamma = st.number_input(
        "γ (degree)",
        value=None,
        placeholder="90"
    )

    z = st.number_input(
        "Z",
        value=None,
        placeholder="4"
    )

# =====================================================
# Feature Preview
# =====================================================

st.header("4. Generated Features")

feature_preview = pd.DataFrame({
    "Feature": [
        "Space group #",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "Z",
        "n_elements",
        "Li_amt",
        "Na_amt",
        "O_amt",
        "S_amt",
        "P_amt",
        "F_amt",
        "Cl_amt"
    ],
    "Value": [
        sg,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        z,
        n_elements,
        Li_amt,
        Na_amt,
        O_amt,
        S_amt,
        P_amt,
        F_amt,
        Cl_amt
    ]
})

st.dataframe(
    feature_preview,
    use_container_width=True
)

# =====================================================
# Prediction
# =====================================================

st.header("5. Predict")

if st.button("Predict Conductivity"):

    required_values = [
        sg,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        z
    ]

    if any(v is None for v in required_values):
        st.error(
            "Please complete all crystal structure parameters."
        )
        st.stop()

    features = [
        'Space group #',
        'a',
        'b',
        'c',
        'alpha',
        'beta',
        'gamma',
        'Z',
        'n_elements',
        'Li_amt',
        'Na_amt',
        'O_amt',
        'S_amt',
        'P_amt',
        'F_amt',
        'Cl_amt'
    ]

    sample = pd.DataFrame(
        [[
            sg,
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            z,
            n_elements,
            Li_amt,
            Na_amt,
            O_amt,
            S_amt,
            P_amt,
            F_amt,
            Cl_amt
        ]],
        columns=features
    )

    pred_log = model.predict(sample)[0]

    conductivity = 10 ** pred_log

    st.success(
        f"Predicted log₁₀(IC) = {pred_log:.4f}"
    )

    st.metric(
        label="Predicted Ionic Conductivity",
        value=f"{conductivity:.3e} S/cm"
    )

    st.info(
        f"Ionic Conductivity = {conductivity:.3e} S/cm"
    )

    st.caption(
        "IC = Ionic Conductivity, Unit = S/cm"
    )
