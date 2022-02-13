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


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import SessionState
from pysurvival.utils import load_model

def main():
    st.title("Posterior Urethral Valves Outcomes Prediction (PUVOP)")
    st.sidebar.image("SickKids logo.png", use_column_width=True)
    st.sidebar.subheader("Navigation")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "PUVOP Tool": full_app,
        "About": about
    }
    page = st.sidebar.selectbox("Select Page", options=list(PAGES.keys()))
    PAGES[page](session_state)


def full_app(session_state):
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
    col1.header("CKD progression-free survival")
    col1.write("This will predict the probability of your kidney function worsening, based on progression "
               "in stage of chronic kidney disease (CKD).")
    col1.write("""""")
    col2.header("RRT-free survival")
    col2.write("This will predict the probability of avoiding the need to start renal replacement therapy (RRT), "
               "such as dialysis or transplant.")
    col2.write("""""")
    col3.header("CIC-free survival")
    col3.write("This will predict the probability of avoiding the need to start clean intermittent catheterization (CIC).")
    col3.write("""""")

    # Load saved items from Google Drive
    CKD_location = st.secrets['CKD']
    RRT_location = st.secrets['RRT']
    CIC_location = st.secrets['CIC']

    @st.cache(allow_output_mutation=True)
    def load_items():
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)
        CKD_checkpoint = Path('model/CKD.zip')
        RRT_checkpoint = Path('model/RRT.zip')
        CIC_checkpoint = Path('model/CIC.zip')

        # download from Google Drive if model or features are not present
        if not CKD_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(CKD_location, CKD_checkpoint)
        if not RRT_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(RRT_location, RRT_checkpoint)
        if not CIC_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(CIC_location, CIC_checkpoint)

        CKD_model = load_model(CKD_checkpoint)
        RRT_model = load_model(RRT_checkpoint)
        CIC_model = load_model(CIC_checkpoint)

        return CKD_model, RRT_model, CIC_model

    CKD_model, RRT_model, CIC_model = load_items()

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
                    snc=snc
                else:
                    snc=snc/88.42
                data = {'Max VUR grade': vur,
                        'SNC1 (mg/dL)': snc,
                        'Antenatal/Postnatal renal dysplasia': renal_dysplasia,
                        'Baseline eGFR': egfr
                        }

                data_features = pd.DataFrame(data, index=[0])

    if submitted:
        st.write("""""")

        # CKD progression-free survival
        CKD_survival = CKD_model.predict_survival(data_features).flatten()
        CKD_survival_1yr = CKD_model.predict_survival(data_features, t=365)
        CKD_survival_3yr = CKD_model.predict_survival(data_features, t=1095)

        # Displaying the functions
        fig, ax = plt.subplots()
        plt.plot(CKD_model.times, CKD_survival, color='blue', lw=2, ls='-')

        # Axis labels
        plt.xlabel('Time from baseline assessment (years)')
        plt.ylabel('CKD progression-free survival (%)')

        # Tick labels
        plt.ylim(0, 1.05)
        y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        y_labels = ('0', '20', '40', '60', '80', '100')
        plt.yticks(y_positions, y_labels, rotation=0)
        plt.xlim(0, 4000)
        x_positions = (0, 365, 1095, 1825, 3650)
        x_labels = ('0', '1', '3', '5', '10')
        plt.xticks(x_positions, x_labels, rotation=0)

        # Tick vertical lines
        plt.axvline(x=365, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1095, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1825, color='black', ls='--', alpha=0.2)
        plt.axvline(x=3650, color='black', ls='--', alpha=0.2)

        CKDprob_1yr = str(np.round(CKD_survival_1yr*100, 1))[1:-1]
        CKDprob_3yr = str(np.round(CKD_survival_3yr*100, 1))[1:-1]

        col1.write(f"**Probability of avoiding CKD progression at 1 year:** {CKDprob_1yr}")
        col1.write(f"**Probability of avoiding CKD progression at 3 years:** {CKDprob_3yr}")
        if egfr < 15:
            col1.write("""""")
            col1.write("The patient has already progressed to end-stage renal disease based on the information "
                       "provided")
        else:
            col1.pyplot(fig)

        # RRT progression-free survival
        RRT_survival = RRT_model.predict_survival(data_features).flatten()
        RRT_survival_1yr = RRT_model.predict_survival(data_features, t=365)
        RRT_survival_3yr = RRT_model.predict_survival(data_features, t=1095)

        # Displaying the functions
        fig2, ax2 = plt.subplots()
        plt.plot(RRT_model.times, RRT_survival, color='red', lw=2, ls='-')

        # Axis labels
        plt.xlabel('Time from baseline assessment (years)')
        plt.ylabel('RRT-free survival (%)')

        # Tick labels
        plt.ylim(0, 1.05)
        y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        y_labels = ('0', '20', '40', '60', '80', '100')
        plt.yticks(y_positions, y_labels, rotation=0)
        plt.xlim(0, 4000)
        x_positions = (0, 365, 1095, 1825, 3650)
        x_labels = ('0', '1', '3', '5', '10')
        plt.xticks(x_positions, x_labels, rotation=0)

        # Tick vertical lines
        plt.axvline(x=365, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1095, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1825, color='black', ls='--', alpha=0.2)
        plt.axvline(x=3650, color='black', ls='--', alpha=0.2)

        RRTprob_1yr = str(np.round(RRT_survival_1yr * 100, 1))[1:-1]
        RRTprob_3yr = str(np.round(RRT_survival_3yr * 100, 1))[1:-1]

        col2.write(f"**Probability of avoiding RRT at 1 year:** {RRTprob_1yr}")
        col2.write(f"**Probability of avoiding RRT at 3 years:** {RRTprob_3yr}")
        col2.pyplot(fig2)

        # CIC progression-free survival
        CIC_survival = CIC_model.predict_survival(data_features).flatten()
        CIC_survival_1yr = CIC_model.predict_survival(data_features, t=365)
        CIC_survival_3yr = CIC_model.predict_survival(data_features, t=1095)

        # Displaying the functions
        fig3, ax3 = plt.subplots()
        plt.plot(CIC_model.times, CIC_survival, color='green', lw=2, ls='-')

        # Axis labels
        plt.xlabel('Time from baseline assessment (years)')
        plt.ylabel('CIC-free survival (%)')

        # Tick labels
        plt.ylim(0, 1.05)
        y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        y_labels = ('0', '20', '40', '60', '80', '100')
        plt.yticks(y_positions, y_labels, rotation=0)
        plt.xlim(0, 4000)
        x_positions = (0, 365, 1095, 1825, 3650)
        x_labels = ('0', '1', '3', '5', '10')
        plt.xticks(x_positions, x_labels, rotation=0)

        # Tick vertical lines
        plt.axvline(x=365, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1095, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1825, color='black', ls='--', alpha=0.2)
        plt.axvline(x=3650, color='black', ls='--', alpha=0.2)

        CICprob_1yr = str(np.round(CIC_survival_1yr * 100, 1))[1:-1]
        CICprob_3yr = str(np.round(CIC_survival_3yr * 100, 1))[1:-1]

        col3.write(f"**Probability of avoiding CIC at 1 year:** {CICprob_1yr}")
        col3.write(f"**Probability of avoiding CIC at 3 years:** {CICprob_3yr}")
        col3.pyplot(fig3)


def about(session_state):
    st.markdown(
        """
    Welcome to Posterior Urethral Valves Outcomes Prediction (PUVOP) tool. PUVOP was developed to predict three specific
    outcomes:
    * Any decline in renal function, based on CKD stage progression
    * Need for renal replacement therapy (dialysis or transplant)
    * Need for clean intermittent catheterization
    
    The CKD-progression, renal replacement therapy, and clean intermittent catheterization-free surivival models 
    achieved a c-index of 0.765, 0.952, and 0.700, respectively, and outperformed Cox proportional hazards regression. 
    Additional information can be found in the reference below or by contacting the authors.
    
    """
    )
    st.subheader("Reference")
    st.markdown(
        """
    **Posterior Urethral Valves Outcomes Prediction (PUVOP): a machine learning tool to predict clinically relevant 
    outcomes in boys with posterior urethral valves**\n
    *Jethro CC. Kwong, Adree Khondker, Jin Kyu Kim, Michael Chua, Daniel T. Keefe, Joana Dos Santos, Marta Skreta, 
    Lauren Erdman, John Weaver, Gregory Tasian, Chia Wei Teoh, Mandy Rickard, Armando J. Lorenzo*
    
    """
    )

if __name__ == "__main__":
    st.set_page_config(page_title="PUVOP - Posterior Urethral Valves Outcome Prediction",
                       page_icon=":toilet:",
                       layout="wide",
                       initial_sidebar_state="expanded"
                       )
    main()
