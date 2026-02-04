import pandas as pd
import numpy as np
from hmmlearn import hmm

# Load the weather data
df = pd.read_csv('/home/aymuos/Documents/Github/starship/weather_info/open-meteo-shanghai.csv', skiprows=3)

# Drop rows where temperature is not numeric
df = df[pd.to_numeric(df['temperature_2m (°C)'], errors='coerce').notna()]

# Select relevant columns
df = df[['time', 'temperature_2m (°C)', 'precipitation (mm)']]

# Convert to numeric
df['temperature_2m (°C)'] = pd.to_numeric(df['temperature_2m (°C)'])
df['precipitation (mm)'] = pd.to_numeric(df['precipitation (mm)'])

# Discretize temperature
def discretize_temp(temp):
    if temp < 25:
        return 0  # cold
    elif temp <= 30:
        return 1  # mild
    else:
        return 2  # hot

# Discretize precipitation
def discretize_precip(precip):
    if precip == 0:
        return 0  # dry
    elif precip <= 1:
        return 1  # light
    else:
        return 2  # heavy

df['temp_bin'] = df['temperature_2m (°C)'].apply(discretize_temp)
df['precip_bin'] = df['precipitation (mm)'].apply(discretize_precip)

# Create observations as pairs
observations = df[['temp_bin', 'precip_bin']].values

# Assume 2 hidden states: Sunny, Rainy
n_states = 2

# Create HMM model
model = hmm.CategoricalHMM(n_components=n_states, random_state=42)

# Fit the model
model.fit(observations)

# Predict hidden states
hidden_states = model.predict(observations)

# Print results
print("Transition matrix:")
print(model.transmat_)
print("\nEmission probabilities:")
print(model.emissionprob_)
print("\nPredicted hidden states (first 10):")
print(hidden_states[:10])
print("\nObservations (first 10):")
print(observations[:10])