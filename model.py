import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def test_and_train(df_encoded,savemodel=True):
    # Assuming df_encoded is the DataFrame with one-hot encoded features
    X = df_encoded.drop(columns=['Disease_label'])  # Drop target column for features
    y = df_encoded['Disease_label']  # Target column

    # Normalize data (importa   nt for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into 70% training and 30% validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Build the neural network model
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Input layer with ReLU activation
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),  # Hidden layer with ReLU activation
        Dropout(0.2),  # Dropout layer
        Dense(5, activation='softmax')  # Output layer with Softmax for classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    if(savemodel == True):
        save(model,history)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

def save(model,history):
    # Save model
    model.save("model/disease_model.h5")

    # Save training history
    import pickle
    with open("model/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Save processed DataFrame if needed
    df_encoded.to_csv("dataset/processed_df.csv", index=False)

if __name__ == '__main__':
    
    df_encoded = pd.read_csv("dataset/feature_engineered_dataset.csv")
    test_and_train(df_encoded,False)