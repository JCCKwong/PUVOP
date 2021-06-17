import numpy as np
import pandas as pd
import streamlit as st
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

    with st.sidebar:
        with st.form(key='my_form'):
            egfr = st.number_input('Baseline eGFR', 0, 1000, key=1)
            oligohydramnios = st.radio('Antenatal oligohydramnios', options=list({0: 'No', 1: 'Yes'}.keys()))
            wt = st.number_input('Birth weight, in kg', 0.00, 50.00, key=1)
            ga = st.number_input('Gestational age, in weeks', 0, 50, key=1)
            renal_dysplasia = st.radio('Antenatal/Postnatal renal dysplasia', options=list({0: 'No', 1: 'Yes'}.keys()))
            vur = st.selectbox('Max VUR grade', options=list({0: 'None', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}.keys()))
            aki = st.radio('Perinatal AKI', options=list({0: 'No', 1: 'Yes'}.keys()))
            time_fu = st.number_input('Time to next follow-up, in months', 0, 36, key=1)
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
                            'time from first follow-up': time_fu*30
                            }

                class_features = pd.DataFrame(class_data, index=[0])
                reg_features = pd.DataFrame(reg_data, index=[0])

    st.write(class_features)
    st.write(reg_features)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Posterior Urethral Valves Outcome Prediction", page_icon=":pencil2:"
    )
    main()
