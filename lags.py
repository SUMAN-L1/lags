import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic, bic, hqic

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

def find_optimal_lags(df, max_lags=5):
    model = VAR(df)
    results = {}
    
    for lags in range(1, max_lags + 1):
        try:
            var_model = model.fit(lags)
            aic_val = aic(var_model.llf, var_model.nobs, var_model.k_ar * var_model.k_vars)
            bic_val = bic(var_model.llf, var_model.nobs, var_model.k_ar * var_model.k_vars)
            hqic_val = hqic(var_model.llf, var_model.nobs, var_model.k_ar * var_model.k_vars)
            
            results[lags] = {'AIC': aic_val, 'BIC': bic_val, 'HQIC': hqic_val}
        except Exception as e:
            st.write(f"Error fitting model with {lags} lags: {e}")
    
    return results

def main():
    st.title('Optimal Lag Selection for Time Series Data')
    
    uploaded_file = st.file_uploader("Upload a CSV file with time series data", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Data Preview:")
            st.write(df.head())
            
            max_lags = st.slider("Select maximum number of lags", min_value=1, max_value=10, value=5)
            
            results = find_optimal_lags(df, max_lags)
            
            st.write("Optimal Lags Results:")
            st.write(pd.DataFrame(results).T)
        else:
            st.write("Uploaded file is empty or not readable.")

if __name__ == "__main__":
    main()
