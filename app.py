import streamlit as st
import sklearn
import numpy as np
import pandas as pd
from joblib import load


result_dict = load('result_dict')
teams = load('team')
model = load('model.sklearn')

st.title('Bundesliga Orakel')

st.write('Diese App sagt Bundesliga-Ergebnisse voraus!')

heimteam = st.selectbox(
    'Heimteam',
     teams.classes_)

anderesteam = st.selectbox(
    'Anders Team',
     teams.classes_)


halbzeitstand = st.text_input('Halbzeitstand:', '0:0')

if st.button('Vorhersagen...'):
    heim = teams.transform([heimteam])
    anderes = teams.transform([anderesteam])
    hthg, htag = map(int, halbzeitstand.split(':'))
    htr = 1
    if hthg > htag:
        htr = 2
    elif hthg < htag:
        htr = 3

    input_data = np.array([heim, anderes, hthg, htag, htr])
    res = model.predict([input_data])[0]
    if res == 1:
        gewinner = 'Unentschieden'
    elif res == 2:
        gewinner = heimteam
    else:
        gewinner = anderesteam

    st.write('Gewinner: ' + gewinner)
