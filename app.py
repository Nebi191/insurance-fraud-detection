import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt

best_model = joblib.load('best_model_lgb.pkl')
preprocessor = joblib.load('preprocessor.pkl')
defaults = joblib.load('defaults.pkl')


st.title('Insurance Fraud Detection')
st.subheader('Manuel Guess')

incident_severity = st.selectbox('Incident Severity', ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'])
auto_year = st.number_input('Auto Year', min_value=1990, max_value=2015, value=2005)

witnesses = st.number_input('Witnesses', min_value=0, max_value=10, value=0)
capital_gains = st.number_input('Capital Gains', min_value=0, max_value=100000, value=0)

input_data = defaults.copy()
input_data['incident_severity'] = incident_severity
input_data['auto_year'] = auto_year
input_data['witnesses'] = witnesses
input_data['capital-gains'] = capital_gains

input_df = pd.DataFrame([input_data])

input_processed = preprocessor.transform(input_df)
prediction = best_model.predict_proba(input_processed)[:, 1]

st.metric('Fraud Probability', f'{prediction[0]:.2%}')

if prediction[0] > 0.5:
    st.error(f'⚠️ Yüksek Risk: {prediction[0]:.2%}')
elif prediction[0] > 0.3:
    st.warning(f'⚠️ Orta Risk: {prediction[0]:.2%}')
else:
    st.success(f'✅ Düşük Risk: {prediction[0]:.2%}')



feature_names = [name.split('__')[1] for name in preprocessor.get_feature_names_out()]
input_processed_df = pd.DataFrame(input_processed, columns=feature_names)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer(input_processed_df)

shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(plt.gcf())