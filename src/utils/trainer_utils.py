
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.loader_utils import config_loader

def get_model(config_path: str):
    # Load configuration
    config = config_loader(config_path)
    train_cfg = config["train"]
    data_cfg = config["data_ingestion"]

    # Training parameters
    base_filters = train_cfg["filters"]
    kernel_size = train_cfg["kernel_size"]
    padding = train_cfg["padding"]
    activation = train_cfg["activation"]
    pool_size = train_cfg["pool_size"]   # FIXED typo
    dropout_rate = train_cfg["rate"]
    lstm_units = train_cfg["units"]

    # Data parameters
    input_shape = (
        data_cfg["sequence_length"],
        data_cfg["input_size"]["image_height"],
        data_cfg["input_size"]["image_width"],
        data_cfg["input_size"]["channels"],
    )
    num_classes = len(data_cfg["classes_list"])

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Convolutional blocks
    filter_multipliers = [1, 2, 4, 4]
    pool_sizes = [pool_size, pool_size, pool_size//2, pool_size//2]

    for mult, p_size in zip(filter_multipliers, pool_sizes):
        model.add(
            layers.TimeDistributed(
                layers.Conv2D(
                    filters=base_filters * mult,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                )
            )
        )
        model.add(
            layers.TimeDistributed(
                layers.MaxPooling2D(pool_size=p_size)
            )
        )
        model.add(
            layers.TimeDistributed(
                layers.Dropout(rate=dropout_rate)
            )
        )

    # Temporal modeling
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(units=lstm_units))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile
    model.compile(
        loss=train_cfg["loss"],
        optimizer=train_cfg["optimizer"],
        metrics=train_cfg["metrics"],
    )

    return model
