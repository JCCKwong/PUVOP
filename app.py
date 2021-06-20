import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import SessionState
from pysurvival.utils import load_model

def main():
    st.title("Posterior Urethral Valves Outcomes Prediction")
    st.sidebar.subheader("Navigation")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "Application": full_app,
        "About": about,
        "Model development and explanation": dev,

    }
    page = st.sidebar.selectbox("Select Page", options=list(PAGES.keys()))
    PAGES[page](session_state)


def full_app(session_state):
    st.sidebar.header("Enter patient values")
    st.markdown(
        """
    **Instructions**:

    1. Enter patient values on the left
    1. Press submit button
    1. The models will generate predictions for the three outcomes
    """
    )

    # Load saved items from Google Drive
    CKD_location = st.secrets['CKD']
    RRT_location = st.secrets['RRT']

    @st.cache(allow_output_mutation=True)
    def load_items():
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)
        CKD_checkpoint = Path('model/CKD.zip')
        RRT_checkpoint = Path('model/RRT.zip')

        # download from Google Drive if model or features are not present
        if not CKD_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(CKD_location, CKD_checkpoint)
        if not RRT_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(RRT_location, RRT_checkpoint)

        CKD_model = load_model(CKD_checkpoint)
        RRT_model = load_model(RRT_checkpoint)

        return CKD_model, RRT_model

    CKD_model, RRT_model = load_items()

    # Define choices and labels for feature inputs
    CHOICES = {0: 'No', 1: 'Yes'}

    def format_func_yn(option):
        return CHOICES[option]

    with st.sidebar:
        with st.form(key='my_form'):
            egfr = st.number_input('Baseline eGFR', 0, 1000, value=60, key=1)
            oligohydramnios = st.selectbox('Antenatal oligohydramnios', options=list(CHOICES.keys()),
                                           format_func=format_func_yn, index=0)
            renal_dysplasia = st.selectbox('Antenatal/Postnatal renal dysplasia', options=list(CHOICES.keys()),
                                           format_func=format_func_yn, index=0)
            vur = st.selectbox('High Grade VUR on initial VCUG (Grade 4-5)', options=list(CHOICES.keys()),
                               format_func=format_func_yn, index=0)
            submitted = st.form_submit_button(label='Submit')

            if submitted:
                data = {'Baseline eGFR': egfr,
                        'Antenatal oligohydramnios': oligohydramnios,
                        'Antenatal/Postnatal renal dysplasia': renal_dysplasia,
                        'Max VUR grade': vur
                        }

                data_features = pd.DataFrame(data, index=[0])

    if submitted:
        st.write("""""")
        col1, col2 = st.beta_columns([1, 1])
        # CKD progression-free survival
        survival = CKD_model.predict_survival(data_features).flatten()
        survival_6mo = CKD_model.predict_survival(data_features, t=182.5)
        survival_12mo = CKD_model.predict_survival(data_features, t=365)

        # Displaying the functions
        fig, ax = plt.subplots()
        plt.plot(CKD_model.times, survival, color='blue', lw=2, ls='-')

        # Axis labels
        plt.xlabel('Time from baseline assessment (months)')
        plt.ylabel('CKD progression-free survival (%)')

        # Tick labels
        plt.ylim(0, 1.05)
        y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        y_labels = ('0', '20', '40', '60', '80', '100')
        plt.yticks(y_positions, y_labels, rotation=0)
        plt.xlim(0, 1200)
        x_positions = (0, 91.25, 182.5, 365, 547.5, 730, 1095)
        x_labels = ('0', '3', '6', '12', '18', '24', '36')
        plt.xticks(x_positions, x_labels, rotation=0)

        # Tick vertical lines
        plt.axvline(x=91.25, color='black', ls='--', alpha=0.2)
        plt.axvline(x=182.5, color='black', ls='--', alpha=0.2)
        plt.axvline(x=365, color='black', ls='--', alpha=0.2)
        plt.axvline(x=547.5, color='black', ls='--', alpha=0.2)
        plt.axvline(x=730, color='black', ls='--', alpha=0.2)
        plt.axvline(x=1095, color='black', ls='--', alpha=0.2)

        CKDprob_6mo = str(np.round(survival_6mo*100, 1))[1:-1]
        CKDprob_12mo = str(np.round(survival_12mo*100, 1))[1:-1]
        col1.markdown("**Probability of CKD progression at 6 months:** ", (survival_6mo*100).astype(np.int))
        col1.markdown("**Probability of CKD progression at 12 months:** ", (survival_12mo*100).astype(np.int))
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

        col2.write("**Probability of initiating RRT at 1 year:** ", str(np.round(RRT_survival_1yr * 100, 1))[1:-1])
        col2.write("**Probability of initiating RRT at 3 years:** ", str(np.round(RRT_survival_3yr * 100, 1))[1:-1])
        col2.pyplot(fig2)


def about(session_state):
    st.markdown(
        """
    Welcome to Posterior Urethral Valves Outcomes Prediction (PUVOP) tool. PUVOP was developed to predict three specific
    outcomes:
    * Any decline in renal function, based on CKD stage progression
    * Need for renal replacement therapy (dialysis or transplant)
    * eGFR in 1 year's time

    Model developement and explanation page: You will find additional details regarding how the model was developed.

    Application: You can access our simple-to-use tool.

    """
    )


def dev(session_state):
    st.markdown(
        """
    Under development

    """
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Posterior Urethral Valves Outcome Prediction", page_icon=":pencil2:"
    )
    main()
