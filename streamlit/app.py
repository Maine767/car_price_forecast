import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Car Price Pro", page_icon="🚗", layout="wide")
st.title("🚗 Прогнозирование стоимости автомобиля")

@st.cache_resource
def load_pipeline():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'car_price_pipeline.pkl')

        with open(file_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline['model'], pipeline['scaler'], pipeline['model_columns']
    except FileNotFoundError:
        st.error("Файл 'car_price_pipeline.pkl' не найден.")
        return None, None, None

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    df = pd.read_csv(url)
    
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
            
    return df

model, scaler, model_columns = load_pipeline()
df_train = load_data()

def clean_torque(x):
    if pd.isna(x): return pd.Series([np.nan, np.nan])
    x = str(x).lower().replace(',', '')
    factor = 9.80665 if 'kgm' in x else 1.0
    nums = re.findall(r'\d+(?:\.\d+)?', x)
    nums = [float(n) for n in nums]
    if len(nums) == 0: return pd.Series([np.nan, np.nan])
    torque_val = nums[0] * factor
    rpm_val = max(nums[1:]) if len(nums) > 1 else np.nan
    return pd.Series([torque_val, rpm_val])

def preprocess_input(df_input, scaler_obj, expected_cols):
    df = df_input.copy()
    
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    
    if 'torque' in df.columns:
        df[['torque_nm', 'max_torque_rpm']] = df['torque'].apply(clean_torque)
        df.drop(columns=['torque'], inplace=True)
    
    numeric_cols_in_df = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols_in_df] = df[numeric_cols_in_df].fillna(0)

    df['name'] = df['name'].astype(str).apply(lambda x: x.split()[0])
    df['age'] = 2020 - df['year']
    df['age_sq'] = df['age'] ** 2
    
    engine_liters = (df['engine'] / 1000).replace(0, 1e-5)
    df['power_per_liter'] = df['max_power'] / engine_liters
    
    df['km_per_year'] = df['km_driven'] / (df['age'].replace(0, 1))
    df['is_sport'] = (df['max_power'] > 150).astype(int)
    df['low_mileage'] = (df['km_driven'] < 20000).astype(int)

    # Scaling
    if scaler_obj:
        try:
            scale_cols = scaler_obj.feature_names_in_
            for c in scale_cols:
                if c not in df.columns:
                    df[c] = 0
            df[scale_cols] = scaler_obj.transform(df[scale_cols])
        except Exception as e:
            st.warning(f"Warning during scaling: {e}")

    potential_cats = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    existing_cats = [c for c in potential_cats if c in df.columns]
    
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False)
    
    df_final = pd.DataFrame(columns=expected_cols)
    
    df_final = pd.concat([df_final, df], axis=0, join='outer')
    
    df_final = df_final[expected_cols].fillna(0)
    
    return df_final

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🎯 Предсказание", "⚙️ Модель"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего авто", df_train.shape[0])
    col2.metric("Медианная цена", f"{df_train['selling_price'].median():,.0f} RUB")
    col3.metric("Медианный пробег", f"{df_train['km_driven'].median():,.0f} км")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(df_train['selling_price'], kde=True, ax=ax)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        df_train['brand'] = df_train['name'].apply(lambda x: x.split()[0])
        top_brands = df_train['brand'].value_counts().head(10).index
        sns.barplot(y=top_brands, x=df_train['brand'].value_counts().head(10), ax=ax)
        st.pyplot(fig)

with tab2:
    mode = st.radio("Режим:", ["Ручной ввод", "Загрузка CSV"])

    if mode == "Ручной ввод":
        with st.form("entry_form"):
            c1, c2 = st.columns(2)
            
            brands = sorted(df_train['name'].astype(str).apply(lambda x: x.split()[0]).unique())
            fuel_types = df_train['fuel'].unique()
            transmissions = df_train['transmission'].unique()
            sellers = df_train['seller_type'].unique()
            owners = df_train['owner'].unique()
            
            min_year, max_year = int(df_train['year'].min()), 2025
            max_km = int(df_train['km_driven'].max())
            max_eng = int(df_train['engine'].max())
            max_bhp = int(df_train['max_power'].max())

            with c1:
                name = st.selectbox("Бренд", brands)
                year = st.number_input("Год", min_year, max_year, 2018)
                km_driven = st.number_input("Пробег", 0, max_km, 50000)
                fuel = st.selectbox("Топливо", fuel_types)
                transmission = st.selectbox("Коробка", transmissions)
                mileage_val = st.number_input("Расход (kmpl)", 0.0, 50.0, 20.0)

            with c2:
                seller_type = st.selectbox("Продавец", sellers)
                owner = st.selectbox("Владелец", owners)
                seats = st.selectbox("Места", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
                engine = st.number_input("Объем (CC)", 0, max_eng, 1200)
                max_power = st.number_input("Мощность (bhp)", 0.0, float(max_bhp), 85.0)
                torque = st.text_input("Крутящий момент", "115Nm@4000rpm")

            if st.form_submit_button("Рассчитать"):
                if model:
                    row = {
                        'name': [name], 'year': [year], 'km_driven': [km_driven],
                        'fuel': [fuel], 'seller_type': [seller_type], 
                        'transmission': [transmission], 'owner': [owner],
                        'seats': [seats], 'engine': [engine], 'max_power': [max_power],
                        'torque': [torque], 'mileage': [f"{mileage_val} kmpl"]
                    }
                    X = preprocess_input(pd.DataFrame(row), scaler, model_columns)
                    pred = np.expm1(model.predict(X))[0]
                    st.success(f"Оценка: {pred:,.0f} RUB")

    else:
        file = st.file_uploader("CSV файл", type="csv")
        if file and model:
            df_up = pd.read_csv(file)
            if st.button("Обработать"):
                X = preprocess_input(df_up, scaler, model_columns)
                df_up['Predicted_Price'] = np.expm1(model.predict(X))
                st.dataframe(df_up)
                st.download_button("Скачать", df_up.to_csv(index=False), "result.csv")

with tab3:
    if model and hasattr(model, 'coef_'):
        feat_df = pd.DataFrame({'Feature': model_columns, 'Weight': model.coef_})
        feat_df['Abs'] = feat_df['Weight'].abs()
        feat_df = feat_df.sort_values('Abs', ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=feat_df, y='Feature', x='Weight', ax=ax, palette='viridis')
        st.pyplot(fig)
    else:
        st.info("Интерпретация весов недоступна.")