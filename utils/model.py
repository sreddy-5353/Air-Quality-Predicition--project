import numpy as np
import os
import pickle

def build_model(input_dim, hidden_layers, neurons, activation, dropout, lr, optimizer_name):
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.BatchNormalization())

    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(neurons, activation=activation,
                                      kernel_regularizer=keras.regularizers.l2(1e-4)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1))

    opts = {
        "adam": keras.optimizers.Adam(learning_rate=lr),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=lr),
        "sgd": keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    }
    model.compile(optimizer=opts[optimizer_name], loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    import tensorflow as tf
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return history


def save_model(model, scaler, features, history=None):
    os.makedirs("saved_model", exist_ok=True)
    saved = {"model": model, "scaler": scaler, "features": features}
    if history:
        saved["history"] = history
    with open("saved_model/model.pkl", "wb") as f:
        pickle.dump(saved, f)
