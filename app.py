import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as plt
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import SessionState


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

    # Load saved items
    from pysurvival.utils import load_model
    CKD_model = load_model('PUV CKD survival.zip')
    
    with st.sidebar:
        with st.form(key='my_form'):
            egfr = st.number_input('Baseline eGFR', 0, 1000, value=60, key=1)
            oligohydramnios = st.radio('Antenatal oligohydramnios', options=list({0: 'No', 1: 'Yes'}.keys()), index=0)
            renal_dysplasia = st.radio('Antenatal/Postnatal renal dysplasia', options=list({0: 'No', 1: 'Yes'}.keys()),
                                       index=0)
            vur = st.selectbox('Max VUR grade',
                               options=list({0: 'None', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}.keys()), index=3)
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
        figure = plt.plot(CKD_model.times, survival, color='blue', label='predicted', lw=2, ls='-',
                          title="Predicted CKD progression-free survival curve", ylim=(0, 1.05),
                          xlim=(0,1000))

        st.write(figure)

        st.write("Probability of CKD progression at 6 months: ", str(np.round(survival_6mo, 3))[1:-1])
        st.write("Probability of CKD progression at 6 months: ", str(np.round(survival_12mo, 3))[1:-1])

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
