import streamlit as st
import sklearn
import numpy as np
import pandas as pd
from joblib import load


result_dict = load('result_dict')
teams = load('team_encoder')
model = load('model.sklearn')

st.title('Bundesliga Orakel')

st.write('Diese App sagt Bundesliga-Ergebnisse voraus!')
st.write('Es handelt sich hierbei um eine KI (Künstliche Intelligenz), die mit sämtlichen [Spielergebnissen](https://www.kaggle.com/thefc17/bundesliga-results-19932018) seit 1993 trainiert wurde.')
st.write('Der Sourcecode ist [hier](https://github.com/dadav/bundesliga) zu finden!')

heimteam = st.selectbox(
    'Heimteam',
     teams.classes_)

anderesteam = st.selectbox(
    'Anders Team',
     teams.classes_)


halbzeitstand = st.text_input('Halbzeitstand:', '0:0')

if st.button('Vorhersagen...'):
    heim = teams.transform([heimteam])[0]
    anderes = teams.transform([anderesteam])[0]

    if heim == anderes:
        st.write('Bitte unterschiedliche Teams wählen!')
    else:
        hthg, htag = map(int, halbzeitstand.split(':'))
        htr = 1
        if hthg > htag:
            htr = 2
        elif hthg < htag:
            htr = 3

        input_data = np.array([heim, anderes, hthg, htag])
        u_win, h_win, a_win = model.predict_proba([input_data])[0]
        res = {
            'Unentschieden': u_win,
            heimteam: h_win,
            anderesteam: a_win,
        }

        for k, v in res.items():
            st.write(f'{k}: {v * 100:.2f}%')

        gewinner = max(res, key=res.get)

        st.write('Gewinner: ' + gewinner)
