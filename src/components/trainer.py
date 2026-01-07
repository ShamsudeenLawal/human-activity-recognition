
# trainer component
def train_model(train_ds_dir, test_ds_dir, config_path):
    import os
    import tensorflow as tf
    from tensorflow import keras
    from src.utils.trainer_utils import get_model
    from src.utils.loader_utils import config_loader
    from src.logger import logging

    # Load model
    model = get_model(config_path)

    # Load training config
    train_config = config_loader(config_path)["train"]

    epochs = train_config.get("epochs", 100)
    batch_size = train_config.get("batch_size", 4)
    verbose = train_config.get("verbose", 2)
    shuffle = train_config.get("shuffle", False)
    model_dir = train_config["model_dir"]

    # Callbacks
    callbacks = []
    if train_config.get("early_stopping"):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=train_config.get("patience", 10),
                mode="max",
                restore_best_weights=True
            )
        )

    # Load datasets
    train_dataset = tf.data.Dataset.load(train_ds_dir)
    test_dataset = tf.data.Dataset.load(test_ds_dir)

    # Shuffle training dataset safely
    cardinality = train_dataset.cardinality()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
        buffer_size = 1000
    else:
        buffer_size = int(cardinality)

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_dataset)
    logging.info(f"Test accuracy: {test_accuracy:.2%}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)

    return test_accuracy, model_path


# sample pipeline
# def sample_pipeline():
#     from src.components.data_ingestion import ingest_data
#     from src.utils.loader_utils import config_loader
#     from src.exception import CustomException
#     from src.logger import logging
#     logging.info("Initializing Training Pipeline")
#     config_path = "configs/config.yaml"
#     train_ds_path, test_ds_path = ingest_data(config_path=config_path)
#     accuracy, model_path = train_model(train_ds_path, test_ds_path, config_path)
#     logging.info("Training Pipeline completed successfully.")
#     print(accuracy)

# if __name__ == "__main__":
#     sample_pipeline()


