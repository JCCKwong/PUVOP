
import streamlit as st

st.set_page_config(page_title="PUVOP - Posterior Urethral Valves Outcome Prediction",
                   page_icon=":toilet:",
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

st.title("About PUVOP")
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/The_Hospital_for_Sick_Children_Logo.svg/'
                 '2560px-The_Hospital_for_Sick_Children_Logo.svg.png',
                 use_column_width=True)

st.markdown(
    """
Welcome to Posterior Urethral Valves Outcomes Prediction (PUVOP) tool. PUVOP was developed to predict three specific
outcomes:
* Any decline in renal function, based on CKD stage progression
* Need for renal replacement therapy (dialysis or transplant)
* Need for clean intermittent catheterization
"""
)

st.header("Reference", divider='gray')
st.markdown(
    """
**Posterior Urethral Valves Outcomes Prediction (PUVOP): a machine learning tool to predict clinically relevant 
outcomes in boys with posterior urethral valves**\n
*Jethro CC. Kwong, Adree Khondker, Jin Kyu Kim, Michael Chua, Daniel T. Keefe, Joana Dos Santos, Marta Skreta, 
Lauren Erdman, John Weaver, Gregory Tasian, Chia Wei Teoh, Mandy Rickard, Armando J. Lorenzo*

For more information, the full manuscript is available [here](https://doi.org/10.1007/s00467-021-05321-3).
"""
)

st.header("Abstract", divider='gray')
st.markdown(
    """
    **Background**: Early kidney and anatomic features may be predictive of future progression and need for additional
     procedures in patients with posterior urethral valve (PUV). The objective of this study was to use machine 
     learning (ML) to predict clinically relevant outcomes in these patients.\n
     
    **Methods**: Patients diagnosed with PUV with kidney function measurements at our institution between 2000 and 2020 
    were included. Pertinent clinical measures were abstracted, including estimated glomerular filtration rate (eGFR) 
    at each visit, initial vesicoureteral reflux grade, and renal dysplasia at presentation. ML models were developed 
    to predict clinically relevant outcomes: progression in CKD stage, initiation of kidney replacement therapy (KRT), 
    and need for clean-intermittent catheterization (CIC). Model performance was assessed by concordance index 
    (c-index) and the model was externally validated.\n

    **Results**: A total of 103 patients were included with a median follow-up of 5.7 years. Of these patients, 26 
    (25%) had CKD progression, 18 (17%) required KRT, and 32 (31%) were prescribed CIC. Additionally, 
    22 patients were included for external validation. The ML model predicted CKD progression (c-index = 0.77; 
    external C-index = 0.78), KRT (c-index = 0.95; external C-index = 0.89) and indicated CIC (c-index = 0.70; external 
    C-index = 0.64), and all performed better than Cox proportional-hazards regression. The models have been packaged 
    into a simple easy-to-use tool, available at https://sickkidsurology-puvop.streamlit.app/.\n
    
    **Conclusion**: ML-based approaches for predicting clinically relevant outcomes in PUV are feasible. Further 
    validation is warranted, but this implementable model can act as a decision-making aid. A higher resolution 
    version of the Graphical abstract is available as Supplementary information.
    """
)