import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

model = joblib.load('rf_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
scaler_km = joblib.load('scaler_kmeans.pkl')
kmeans = joblib.load('kmeans.pkl')
geo_scaler = joblib.load('geo_scaler.pkl')
geo_kmeans = joblib.load('geo_kmeans.pkl')

# Load CSV 
cluster_p = pd.read_csv('cluster_p.csv', index_col=0)
world = pd.read_csv('world_country_small.csv')

st.set_page_config(layout="wide")
tab1, tab2, tab3 = st.tabs(["💰 income_Prediction", "🔍 KMeans", "🌍 Geospatial"])

with tab1:
    st.title('Income predictor >50k?')
    age = st.number_input('age', 17, 90, 30, key='age1')
    workclass = st.selectbox("Workclass", ['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked'])
    occupation = st.selectbox("Occupation", ['Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces'])
    race = st.selectbox("Race", ['White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black'])
    sex = st.selectbox("Sex", ['Male','Female'])
    marital_status = st.selectbox("Marital", ['Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'])
    education_num = st.number_input('education_num', 1, 16, 10)
    capital_gain = st.number_input('capital_gain', 0, 99999, 0, 500)
    capital_loss = st.number_input('capital_loss', 0, 4356, 0)
    hours_per_week = st.number_input('hours_per_week', 1, 80, 40, key='h1')

    if st.button('predict_income_>50?'):
        capital_gain_log = np.log1p(capital_gain)
        capital_loss_log = np.log1p(capital_loss)
        cat_data = pd.DataFrame([[workclass, marital_status, occupation, race, sex]], columns=['workclass','marital_status','occupation','race','sex'])
        num_data = pd.DataFrame([[age, education_num, capital_gain_log, capital_loss_log, hours_per_week]], columns=['age','education_num','capital_gain_log','capital_loss_log','hours_per_week'])
        cat_encoded = encoder.transform(cat_data)
        num_scaled = scaler.transform(num_data)
        combined_data = np.concatenate([cat_encoded, num_scaled], axis=1)
        proba = model.predict_proba(combined_data)[0][1]
        threshold = 0.677
        prediction = 1 if proba >= threshold else 0

        st.success(f"Prediction: {'>50K' if prediction==1 else '<=50K'} | Prob: {proba:.2f} (threshold {threshold})")
        if prediction==1:
            st.info("High income predicted")
        else:
            st.warning("Low income predicted")
with tab2:
    st.header("KMeans: Three Work Patterns")
    st.write("K=3, Silhouette=0.593, WCSS elbow at 3")
    st.dataframe(cluster_p)
    st.dataframe(pd.DataFrame({'Cluster':[0,1,2],'Meaning':['Nine_to_Five - No capital activity','High Earners - Capital Gain driven','Risk_Takers- Capital loss']}))

    age_k = st.number_input("Age", 17, 90, 30, key='agek')
    hours_k = st.number_input('hours_k', 1, 80, 40, key='hk')
    gain_real = st.slider("Capital Gain ($)", 0, 100000, 0, 500)
    gain_log = np.log1p(gain_real)
    st.caption(f"log value: {gain_log:.2f}")
    loss_real = st.slider("Capital Loss ($)", 0, 4356, 0, 50)
    loss_log = np.log1p(loss_real)
    st.caption(f"log value: {loss_log:.2f}")

    if st.button("Find my cluster"):
        Xk = scaler_km.transform([[age_k, gain_log, loss_log, hours_k]])
        cl = kmeans.predict(Xk)[0]
        if cl == 1:
            st.success(f"You belong to Cluster 1 - High Earners Capital Gain (63% >50K)")
        elif cl == 2:
            st.success(f"You belong to Cluster 2 - Risk Takers Capital Loss (51% >50K)")
        else:
            st.success(f"You belong to Cluster 0 - Nine to Five No capital (20% >50K)")

with tab3:
    st.header("Geospatial: Low vs Higher Resource Countries")
    import geopandas as gpd
    import matplotlib.pyplot as plt


    gdf = gpd.read_file("world_country.geojson")

    st.dataframe(gdf.drop(columns='geometry'))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column='geo_cluster', cmap='coolwarm', ax=ax, legend=True, edgecolor='black', linewidth=0.3)
    ax.set_title('World Map of Country Clusters', fontsize=16, pad=20)
    ax.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig)