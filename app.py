import streamlit as st
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("tehranhouses.csv")

address_sorted = sorted(df["Address"].astype(str).unique())

model = joblib.load('model.joblib')

st.title('Prediction of House Price in Tehran 🏠')

Area = st.number_input("Area in squared meter", 60) 
Room = st.selectbox("Number of Rooms", [0, 1, 2, 3, 4, 5])
Parking = st.selectbox("Does it have Parking?", ['True','False'])
Warehouse = st.selectbox("Does it have Warehouse?", ['True','False'])
Elevator = st.selectbox("Does it have Elevator?", ['True','False'])
Address = st.selectbox("Neighborhood", address_sorted) 

def predict_price(Area,Room,Parking,Warehouse,Elevator,Address):
    Area = float(Area)
    Room = int(Room)
    Address = str(Address)
    Parking = Parking == 'True'
    Warehouse = Warehouse == 'True'
    Elevator = Elevator == 'True'

    X = pd.DataFrame({
        'Area': [Area],
        'Room': [Room],
        'Parking': [Parking],
        'Warehouse': [Warehouse],
        'Elevator': [Elevator],
        'Address': [Address]
    })
    
    X["Address"] = pd.Categorical(X['Address'], categories=address_sorted)
    
    prediction = model.predict(X)
    st.success(f'Predicted Price: {int(prediction[0]):,} tooman')

trigger = st.button('Predict',
                    on_click=predict_price,
                    args=(Area, Room, Parking, Warehouse, Elevator, Address))