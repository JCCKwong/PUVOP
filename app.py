import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import SessionState

def main():
    st.title("Posterior Urethral Valves Outcomes Prediction")
    st.sidebar.subheader("Navigation")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "About": about,
        "Model development and explanation": dev,
        "Application": full_app
    }
    page = st.sidebar.selectbox("Select Page", options=list(PAGES.keys()))
    PAGES[page](session_state)


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
    RRT_location = st.secrets['RRT']
    EGFR_location = st.secrets['EGFR']
    CKD_location = st.secrets['CKD']

    @st.cache(allow_output_mutation=True)
    def load_items():
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)
        RRT_checkpoint = Path('model/PUV RRT.pkl')
        EGFR_checkpoint = Path('model/PUV eGFR.pkl')
        CKD_checkpoint = Path('model/PUV CKD.pkl')

        # download from Google Drive if model or features are not present
        if not RRT_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(RRT_location, RRT_checkpoint)
        if not EGFR_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(EGFR_location, EGFR_checkpoint)
        if not CKD_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                gdd.download_file_from_google_drive(CKD_location, CKD_checkpoint)

        RRT_model = joblib.load(RRT_checkpoint)
        EGFR_model = joblib.load(EGFR_checkpoint)
        CKD_model = joblib.load(CKD_checkpoint)

        return RRT_model, EGFR_model, CKD_model

    RRT_model, EGFR_model, CKD_model = load_items()

    with st.sidebar:
        with st.form(key='my_form'):
            egfr = st.number_input('Baseline eGFR', 0, 1000, value=60, key=1)
            oligohydramnios = st.radio('Antenatal oligohydramnios', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
            wt = st.number_input('Birth weight, in kg', 0.00, 50.00, value=2.80, key=1)
            ga = st.number_input('Gestational age, in weeks', 0, 50, value=37, key=1)
            renal_dysplasia = st.radio('Antenatal/Postnatal renal dysplasia', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
            vur = st.selectbox('Max VUR grade', options=list({0: 'None', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}.keys()), index=3)
            aki = st.radio('Perinatal AKI', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
            time_fu = st.number_input('Time to next follow-up, in months', 0, 36, value=12, key=1)
            submitted = st.form_submit_button(label='Submit')

            if submitted:
                class_data = {'Baseline eGFR': egfr,
                              'Antenatal oligohydramnios': oligohydramnios,
                              'Birth weight': wt,
                              'Gestational age': ga,
                              'Antenatal/Postnatal renal dysplasia': renal_dysplasia,
                              'Max VUR grade': vur,
                              'Perinatal AKI': aki
                              }
                reg_data = {'Baseline eGFR': egfr,
                            'Max VUR grade': vur,
                            'time from first follow-up': time_fu*30.4
                            }

                class_features = pd.DataFrame(class_data, index=[0])
                reg_features = pd.DataFrame(reg_data, index=[0])
    if submitted:
        st.write(class_features)
        st.write(reg_features)

        prob_RRT = RRT_model.predict_proba(class_features)[:,1]
        prob_CKD = CKD_model.predict_proba(class_features)[:,1]
        pred_EGFR = EGFR_model.predict(reg_features)

        st.write("Probability of any CKD progression: ", str(np.round(prob_CKD,3))[1:-1])
        st.write("Probability of need for renal replacement therapy (dialysis or transplant): ", str(np.round(prob_RRT,3))[1:-1])
        st.write(f"Predicted eGFR at {time_fu} months: {str(np.round(pred_EGFR,3))[1:-1]} ml/min/1.73 m^2")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Posterior Urethral Valves Outcome Prediction", page_icon=":pencil2:"
    )
    main()
