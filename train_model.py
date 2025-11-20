#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data.csv")

# Rename columns to simpler names
df.rename(columns={
    "Wind Speed (m/s)": "Wind Speed",
    "Theoretical_Power_Curve (KWh)": "Theoretical Power",
    "Wind Direction (°)": "Wind Direction",
    "LV ActivePower (kW)": "Active Power"
}, inplace=True)

# Define features and target
X = df[["Wind Speed", "Theoretical Power", "Wind Direction"]]
y = df["Active Power"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")


# In[ ]:




