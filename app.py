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
    Instructions:

    1. Enter patient values on the left
    1. Press submit button
    1. The models will generate predictions for the three outcomes
    """
    )

    # Load saved items from Google Drive
    CKD_location = st.secrets['CKD']

    @st.cache(allow_output_mutation=True)
    def load_items():
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)
        CKD_checkpoint = Path('model/CKD.zip')

        # download from Google Drive if model or features are not present
        if not CKD_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(CKD_location, CKD_checkpoint)

        CKD_model = load_model(CKD_checkpoint)

        return CKD_model

    CKD_model = load_items()

    with st.sidebar:
        with st.form(key='my_form'):
            egfr = st.number_input('Baseline eGFR', 0, 1000, value=60, key=1)
            oligohydramnios = st.selectbox('Antenatal oligohydramnios', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
            renal_dysplasia = st.selectox('Antenatal/Postnatal renal dysplasia', options=list({0: 'No', 1: 'Yes'}.keys()),
                                       index=0)
            vur = st.selectbox('High Grade VUR on initial VCUG (Grade 4-5)', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
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

        survival = CKD_model.predict_survival(data_features).flatten()
        survival_6mo = CKD_model.predict_survival(data_features, t=182.5)
        survival_12mo = CKD_model.predict_survival(data_features, t=365)

        # Displaying the functions
        fig, ax = plt.subplots()
        plt.plot(CKD_model.times, survival, color='blue', lw=2, ls='-')
        plt.xlabel('Time from baseline assessment (months)')
        plt.ylabel('CKD progression-free survival (%)')
        plt.ylim(0, 1.05)
        y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
        y_labels = ('0', '20', '40', '60', '80', '100')
        plt.yticks(y_positions, y_labels, rotation=0)
        plt.xlim(0, 1100)
        x_positions = (0, 91.25, 182.5, 365, 547.5, 730, 1095)
        x_labels = ('0', '3', '6', '12', '18', '24', '36')
        plt.xticks(x_positions, x_labels, rotation=0)

        st.write("Probability of CKD progression at 6 months: ", str(np.round(survival_6mo*100, 1))[1:-1])
        st.write("Probability of CKD progression at 12 months: ", str(np.round(survival_12mo*100, 1))[1:-1])
        st.pyplot(fig)


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
