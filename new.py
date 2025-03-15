import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
# Load Dataset
df = pd.read_csv("crop_yield.csv")  # Replace with actual file path

# Identify Categorical Columns
categorical_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

# One-Hot Encode Categorical Features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # Converts categories into numeric

# Separate Features (X) and Target (y)
X = df.drop(columns=["Yield_tons_per_hectare"]).values  # Features
y = df["Yield_tons_per_hectare"].values   # Target

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Check if CUDA (GPU) is available
device = "/GPU:0" if tf.config.experimental.list_physical_devices('GPU') else "/CPU:0"
print(device)
# Run on GPU (if available)
with tf.device(device):  
    # Define MLP Model for Regression
    model = keras.Sequential([
        Dense(128, input_shape=(X_train.shape[1],)),  
        BatchNormalization(),
        ReLU(),
        Dense(64),  
        BatchNormalization(),
        ReLU(),
        Dense(32),  
        BatchNormalization(),
        ReLU(),
        Dense(1)  # No activation for regression
    ])

    # Compile Model (Regression uses MSE loss)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train Model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate Model
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Mean Absolute Error: {mae:.4f} tons per hectare")


# df.hist(bins=50, figsize =(20,10))
# plt.show()
#sns.scatterplot(x='Rainfall_mm',
                #y='Yield_tons_per_hectare',
                #hue = 'Crop',
                #data = df)
#plt.show()
# crops = ['Maize', 'Cotton', 'Rice', 'Wheat', 'Barley', 'Soybean']
# for crop in crops:
#     # Filter the DataFrame to only include the current crop
#     crop_df = df[df['Crop'] == crop]

#     # Create the scatter plot for the current crop
#     sns.scatterplot(x='Rainfall_mm', y='Yield_tons_per_hectare', data=crop_df)

#     # Set the title with the crop name
#     plt.title(f'Scatter Plot for {crop}')

#     # Show the plot
#     plt.show()
# Calculate the average yield for each crop
# Calculate the average yield for each crop
# Filter the DataFrame to only include Rice and Wheat
rice_wheat_df = df[df['Crop'].isin(['Rice', 'Wheat'])]

# # Calculate the average yield for Rice and Wheat
# average_yield_rice_wheat = rice_wheat_df.groupby('Crop')['Yield_tons_per_hectare'].mean()

# # Print the average yields
# print(average_yield_rice_wheat)
# average_yield = df.groupby('Crop')['Yield_tons_per_hectare'].mean().reset_index()

# # Create a bar plot for the average yield of each crop
# ax = sns.barplot(x='Crop', y='Yield_tons_per_hectare', data=average_yield)

# # Add the values on top of the bars
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.2f}',  # Format the value to 2 decimal places
#                 (p.get_x() + p.get_width() / 2., p.get_height()),  # Position on top of the bar
#                 ha='center', va='center',  # Horizontal and vertical alignment
#                 fontsize=12, color='black',  # Text properties
#                 xytext=(0, 5),  # Text offset (so it's above the bar)
#                 textcoords='offset points')

# # Set the title of the plot
# plt.title('Average Yield per Crop')

# # Show the plot
# plt.show()

