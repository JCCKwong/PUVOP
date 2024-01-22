"""
Title: Posterior Urethral Valves Outcomes Prediction (PUVOP): a machine learning tool to predict clinically relevant
outcomes in boys with posterior urethral valves

Authors: Jethro CC. Kwong, Adree Khondker, Jin Kyu Kim, Michael Chua, Daniel T. Keefe, Joana Dos Santos, Marta Skreta,
Lauren Erdman, John Weaver, Gregory Tasian, Chia Wei Teoh, Mandy Rickard, Armando J. Lorenzo

PUVOP was developed to predict three specific outcomes:
1. Any decline in renal function, based on CKD stage progression
2. Need for renal replacement therapy (dialysis or transplant)
3. Need for clean intermittent catheterization
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sksurv.ensemble import RandomSurvivalForest

st.set_page_config(page_title="PUVOP - Posterior Urethral Valves Outcome Prediction",
                   page_icon=":toilet:",
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

st.title("Posterior Urethral Valves Outcomes Prediction (PUVOP)")
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/The_Hospital_for_Sick_Children_Logo.svg/'
                 '2560px-The_Hospital_for_Sick_Children_Logo.svg.png',
                 use_column_width=True)

@st.cache_data()
def load_models():
    ckd = joblib.load(r'models/PUVOP CKD.joblib')
    rrt = joblib.load(r'models/PUVOP RRT.joblib')
    cic = joblib.load(r'models/PUVOP CIC.joblib')

    return ckd, rrt, cic

ckd, rrt, cic = load_models()

st.sidebar.header("Enter patient values")
st.subheader("Instructions")
st.markdown(
    """
1. Enter patient values on the left
    1. **High-Grade VUR on initial VCUG**: presence of Grade IV or V vesicoureteral reflux (VUR) on initial 
    voiding cystourethrogram (VCUG)
    2. **Serum nadir creatinine at first year of presentation**: for patients with neonatal diagnosis of PUV, this 
    would refer to serum nadir creatinine within the first year of life. Please ensure creatinine is inputted in 
    the correct units
    3. **Renal dysplasia at presentation**: this includes increased echogenicity, cortical cysts, or reduced 
    corticomedullary differentiation on renal ultrasound
    4. **Baseline eGFR at one year, or at time of presentation** 
1. Press submit button
1. The models will generate predictions
"""
)

col1, col2, col3 = st.columns([1, 1, 1])
col1.header("CKD progression")
col1.write("This will predict the probability of your kidney function worsening, based on progression "
           "in stage of chronic kidney disease (CKD).")
col2.header("Requiring RRT")
col2.write("This will predict the probability of needing to start renal replacement therapy (RRT), "
           "such as dialysis or transplant.")
col3.header("Requiring CIC")
col3.write("This will predict the probability of needing to start clean intermittent catheterization (CIC).")

# Define choices and labels for feature inputs
CHOICES = {0: 'No', 1: 'Yes'}

def format_func_yn(option):
    return CHOICES[option]

with st.sidebar:
    with st.form(key='my_form'):
        vur = st.selectbox('High Grade VUR on initial VCUG', options=list(CHOICES.keys()),
                           format_func=format_func_yn, index=1)
        units = st.radio('Units of measurement for creatinine',('mg/dL', 'umol/L'), index=0)
        snc = st.number_input('Serum nadir creatinine at first year of presentation', 0.00, 1000.00, value=0.50)
        renal_dysplasia = st.selectbox('Renal dysplasia at presentation', options=list(CHOICES.keys()),
                                       format_func=format_func_yn, index=1)
        egfr = st.number_input('Baseline eGFR at one year, or at time of presentation', 0.00, 1000.00, value=58.00)

        submitted = st.form_submit_button(label='Submit')

if submitted:
    if units == 'mg/dL':
        snc = snc
    else:
        snc = snc/88.42
    data = {'Max VUR (high vs low grade)': vur,
            'SNC1 (mg/dL)': snc,
            'Renal dysplasia': renal_dysplasia,
            'Baseline eGFR': egfr
            }

    pt_features = pd.DataFrame(data, index=[0])

    # Risk of CKD
    progress_bar = st.progress(0, text="Calculating risk of CKD, please wait :hourglass_flowing_sand:...")
    ckd_surv = ckd.predict_survival_function(pt_features)
    ckd_fig_individual, ax_ind = plt.subplots(1, 1, figsize=(6, 5))

    for fn in ckd_surv:
        ax_ind.step(fn.x, 1 - fn(fn.x), where="post", label=None, color='#005BA8', lw=3, ls='-')
    ax_ind.set_ylabel("Risk of developing CKD (%)")
    ax_ind.set_xlabel("Time from baseline assessment (years)")
    ax_ind.set_ylim([0, 1])
    ax_ind.set_yticks(np.arange(0, 1.1, 0.1))
    ax_ind.set_yticklabels(np.arange(0, 110, 10))
    ax_ind.set_xlim([0, 3650])
    ax_ind.set_xticks(np.arange(0, 3660, 365))
    ax_ind.set_xticklabels(np.arange(0, 11, 1))

    ax_ind.grid(which='major', axis='both', color='k', linestyle='-', linewidth=1, alpha=.1)
    ax_ind.legend().remove()

    # Print Survival probabilities at 1, 3, 5, and 10 years
    ckd_risk_1yr = round(np.interp(1 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    ckd_risk_3yr = round(np.interp(3 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    ckd_risk_5yr = round(np.interp(5 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    ckd_risk_10yr = round(np.interp(10 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)

    ckd_individual_risk = pd.DataFrame({"Time (years)": [1, 3, 5, 10],
                                        "Risk of developing CKD (%)": [ckd_risk_1yr, ckd_risk_3yr, ckd_risk_5yr,
                                                                       ckd_risk_10yr]
                                        })

    # Risk of needing RRT
    progress_bar.progress(33, text="Calculating risk of requiring RRT, please wait :hourglass_flowing_sand:...")
    rrt_surv = rrt.predict_survival_function(pt_features)
    rrt_fig_individual, ax_ind = plt.subplots(1, 1, figsize=(6, 5))

    for fn in rrt_surv:
        ax_ind.step(fn.x, 1 - fn(fn.x), where="post", label=None, color='#005BA8', lw=3, ls='-')
    ax_ind.set_ylabel("Risk of requiring RRT (%)")
    ax_ind.set_xlabel("Time from baseline assessment (years)")
    ax_ind.set_ylim([0, 1])
    ax_ind.set_yticks(np.arange(0, 1.1, 0.1))
    ax_ind.set_yticklabels(np.arange(0, 110, 10))
    ax_ind.set_xlim([0, 3650])
    ax_ind.set_xticks(np.arange(0, 3660, 365))
    ax_ind.set_xticklabels(np.arange(0, 11, 1))

    ax_ind.grid(which='major', axis='both', color='k', linestyle='-', linewidth=1, alpha=.1)
    ax_ind.legend().remove()

    # Print Survival probabilities at 1, 3, 5, and 10 years
    rrt_risk_1yr = round(np.interp(1 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    rrt_risk_3yr = round(np.interp(3 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    rrt_risk_5yr = round(np.interp(5 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)
    rrt_risk_10yr = round(np.interp(10 * 365, fn.x, 1 - fn(fn.x)) * 100, 0)

    rrt_individual_risk = pd.DataFrame({"Time (years)": [1, 3, 5, 10],
                                        "Risk of requiring RRT (%)": [rrt_risk_1yr, rrt_risk_3yr, rrt_risk_5yr,
                                                                      rrt_risk_10yr]
                                        })

    # Risk of needing CIC
    progress_bar.progress(66, text="Calculating risk of requiring CIC, please wait :hourglass_flowing_sand:...")
    cic_surv = cic.predict_survival_function(pt_features)
    cic_fig_individual, ax_ind = plt.subplots(1, 1, figsize=(6, 5))

    for fn in cic_surv:
        ax_ind.step(fn.x, 1 - fn(fn.x), where="post", label=None, color='#005BA8', lw=3, ls='-')
    ax_ind.set_ylabel("Risk of requiring CIC (%)")
    ax_ind.set_xlabel("Time from baseline assessment (years)")
    ax_ind.set_ylim([0, 1])
    ax_ind.set_yticks(np.arange(0, 1.1, 0.1))
    ax_ind.set_yticklabels(np.arange(0, 110, 10))
    ax_ind.set_xlim([0, 3650])
    ax_ind.set_xticks(np.arange(0, 3660, 365))
    ax_ind.set_xticklabels(np.arange(0, 11, 1))

    ax_ind.grid(which='major', axis='both', color='k', linestyle='-', linewidth=1, alpha=.1)
    ax_ind.legend().remove()

    # Print Survival probabilities at 1, 3, 5, and 10 years
    cic_risk_1yr = round(np.interp(1*365, fn.x, 1 - fn(fn.x)) * 100, 0)
    cic_risk_3yr = round(np.interp(3*365, fn.x, 1 - fn(fn.x)) * 100, 0)
    cic_risk_5yr = round(np.interp(5*365, fn.x, 1 - fn(fn.x)) * 100, 0)
    cic_risk_10yr = round(np.interp(10*365, fn.x, 1 - fn(fn.x)) * 100, 0)

    cic_individual_risk = pd.DataFrame({"Time (years)": [1, 3, 5, 10],
                                        "Risk of requiring CIC (%)": [cic_risk_1yr, cic_risk_3yr, cic_risk_5yr,
                                                                      cic_risk_10yr]
                                        })

    progress_bar.progress(100, text="Completing prediction, please wait :hourglass_flowing_sand:...")
    progress_bar.empty()

    # Display results
    col1.pyplot(ckd_fig_individual, use_container_width=True)
    col1.dataframe(data=ckd_individual_risk, use_container_width=True, hide_index=True)
    col2.pyplot(rrt_fig_individual, use_container_width=True)
    col2.dataframe(data=rrt_individual_risk, use_container_width=True, hide_index=True)
    col3.pyplot(cic_fig_individual, use_container_width=True)
    col3.dataframe(data=cic_individual_risk, use_container_width=True, hide_index=True)
