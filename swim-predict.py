# This is an AI used for predicting times based on user input
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

def main():

    data = {
        "strokes": ["free", "free", "free", "free", "back", "back", "breast", "breast", "breast", "fly", "fly", "fly", "IM", "IM"], # 1 = free, 2 = back, 3 = breast, 4 = fly, 5= IM
        "distance": [50, 100, 200, 500, 50, 100, 50, 100, 200, 50, 100, 200, 100, 200], # Distance is in yards
        "times": [21.71, 47.49, 103.30, 293.53, 29.84, 64.23, 33.22, 69.52, 205.53, 24.22, 51.07, 112.35, 64.13, 118.91] # Times are all in seconds
    }

    # Store it into a dataframe (table with rows and columns to store information)
    df = pd.DataFrame(data)

    # Encode the stroke column into binary columns (done with get_dummies)
    df_encoded = pd.get_dummies(df, columns=["strokes"])

    # Create an X dataframe for the inputs (drop the outputs)
    X = df_encoded.drop(columns=["times"])
   
    # Create a Y dataframe for the outputs
    Y = df_encoded[["times"]]

    # Normalize features so 'distance' (50-500) doesn't overpower the binary stroke columns (0-1)
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Make a model that creates layers (input, hidden, output)
    model = Sequential()
    input_dimension = X_scaled.shape[1]

    # Input layer 
        # Units is the number of nodes, ReLU creates non-linearity, 
        # and input_shape is the number of "features" put in a tuple
        # (for the input it is stroke and distance)
    model.add(Dense(units = 64, activation = "relu", input_shape = (input_dimension,)))

    # Hidden layers
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 16, activation = "relu"))

    # Output layer
    model.add(Dense(units = 1))

    # Compile the model (set rules)
        # Optimizers determines how much to shift the weights by and in
        # what direction ("adam" is the most popular, general one)
        # Loss compares the actual value to the predicted value
    model.compile(optimizer = "adam", loss = "mean_squared_error")

    # Now we train the model on our data (inputs and outputs)
        # Batch_size is how big of a batch the model analyzes
        # After going through all of the batches in the sample, 
        # the model adjusts the weights and does all of that epochs times
    model.fit(X_scaled, Y, epochs = 125, batch_size = 5)

    # User input mapping numbers to the string keys
    stroke_map = {1: "free", 2: "back", 3: "breast", 4: "fly", 5: "IM"}
    
    while True:    
        try: 
            user_stroke_num = int(input("Please enter your desired stroke (1 = free, 2 = back, 3 = breast, 4 = fly, 5 = IM): "))
            if 1 <= user_stroke_num <= 5: 
                selected_stroke = stroke_map[user_stroke_num]
                break
            print("Please enter a number between 1 and 5")
        except: print("Please enter a valid integer")
    while True:    
        try:
            user_distance = int(input("Please enter your desired distance (50, 100, 200, 500): "))
            if user_distance != 50 and user_distance != 100 and user_distance != 200 and user_distance != 500:
                print("Please enter one of the given numbers")
            else:
                break
        except:
            print("Please enter a valid integer")
# Build a dictionary matching the exact columns created by get_dummies
    sample_dict = {
        "distance": [user_distance],
        "strokes_back": [1 if selected_stroke == "back" else 0],
        "strokes_breast": [1 if selected_stroke == "breast" else 0],
        "strokes_fly": [1 if selected_stroke == "fly" else 0],
        "strokes_free": [1 if selected_stroke == "free" else 0],
        "strokes_IM": [1 if selected_stroke == "IM" else 0]
    }
    
    sample_df = pd.DataFrame(sample_dict)
    sample_df = sample_df[X.columns]  # Ensure column alignment

    # Normalize the sample input using training mean and std
    sample_scaled = (sample_df - X_mean) / X_std

    # Predict the time
    sample_time = model.predict(sample_scaled)
    print(f"Your predicted time for {user_distance} {selected_stroke} is {sample_time[0][0]:,.2f} seconds.")
    print("Note: If you chose events that you haven't swum, the predicted time will be VERY inaccurate!")

if __name__ == "__main__":
    main()