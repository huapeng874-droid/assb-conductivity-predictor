import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ASSB Predictor",
    layout="wide"
)

st.title("🔋 ASSB Conductivity Predictor")

model = joblib.load("conductivity_model.pkl")

features = [
    'Space group #',
    'a','b','c',
    'alpha','beta','gamma',
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

sg = st.number_input("Space Group", value=216)

a = st.number_input("a", value=10.0)
b = st.number_input("b", value=10.0)
c = st.number_input("c", value=10.0)

alpha = st.number_input("alpha", value=90.0)
beta = st.number_input("beta", value=90.0)
gamma = st.number_input("gamma", value=90.0)

z = st.number_input("Z", value=4)

n_elements = st.number_input("n_elements", value=4)

Li_amt = st.number_input("Li_amt", value=6)
Na_amt = st.number_input("Na_amt", value=0)

O_amt = st.number_input("O_amt", value=0)
S_amt = st.number_input("S_amt", value=5)

P_amt = st.number_input("P_amt", value=1)
F_amt = st.number_input("F_amt", value=0)

Cl_amt = st.number_input("Cl_amt", value=1)

if st.button("Predict"):

    sample = pd.DataFrame([[
        sg,a,b,c,
        alpha,beta,gamma,
        z,
        n_elements,
        Li_amt,
        Na_amt,
        O_amt,
        S_amt,
        P_amt,
        F_amt,
        Cl_amt
    ]], columns=features)

    pred = model.predict(sample)[0]

    st.success(
        f"Predicted log10(IC) = {pred:.3f}"
    )

    conductivity = 10 ** pred

    st.info(
        f"Estimated conductivity = {conductivity:.3e} S/cm"
    )