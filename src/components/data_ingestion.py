# from src.utils.loader_utils import create_dataset
# from src.utils.loader_utils import config_loader
# from src.exception import CustomException
# from src.logger import logging


def ingest_data(config_path):
    # import packages
    import os
    from src.utils.loader_utils import create_dataset
    from src.utils.loader_utils import config_loader
    from src.exception import CustomException
    from src.logger import logging
    
    # load data ingestiion configuration
    data_config = config_loader(config_path).get("data_ingestion", {})

    dataset_dir = data_config["dataset_dir"]
    processed_data_dir = data_config["processed_data_directory"]
    classes_list = data_config["classes_list"]

    num_files = data_config.get("num_files", None)
    sequence_length = data_config.get("sequence_length", 1)
    seed = data_config.get("seed", 42)
    test_size = data_config.get("test_size", 0.2)

    input_size = data_config.get("input_size", {})
    image_height = input_size.get("image_height", 224)
    image_width = input_size.get("image_width", 224)
    channels = input_size.get("channels", 3)


    try:
        logging.info("Initializing data ingestion.")
        
        train_dataset, test_dataset = create_dataset(
            dataset_dir=dataset_dir, classes_list=classes_list,
            num_files=num_files, sequence_length=sequence_length, test_size=test_size,
            seed=seed, image_height=image_height, image_width=image_width, channels=channels,
            )

        buffer_size = train_dataset.cardinality()
        dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed)
        

        logging.info("Data ingested successfuly.")
        logging.info("Saving ingested data.")

        # save data to path
        train_ds_dir = os.makedirs(os.path.join(processed_data_dir, "train_dataset"), exist_ok=True)
        train_dataset.save(train_ds_dir)
        
        test_ds_dir = os.makedirs(os.path.join(processed_data_dir, "test_dataset"), exist_ok=True)
        test_dataset.save(test_ds_dir)
        logging.info("Data saved successfully.")

        return train_ds_dir, test_ds_dir
    
    except Exception as err:
        raise CustomException(err, sys)
    


def ingest_data(config_path):
    import os
    import sys
    import tensorflow as tf

    from src.utils.loader_utils import create_dataset, config_loader
    from src.exception import CustomException
    from src.logger import logging

    try:
        logging.info("Initializing data ingestion.")

        # Load config
        data_config = config_loader(config_path).get("data_ingestion", {})

        dataset_dir = data_config["dataset_dir"]
        processed_data_dir = data_config["processed_data_directory"]
        classes_list = data_config["classes_list"]

        num_files = data_config.get("num_files", 50)
        sequence_length = data_config.get("sequence_length", 20)
        seed = data_config.get("seed", 42)
        test_size = data_config.get("test_size", 0.2)

        input_size = data_config.get("input_size", {})
        image_height = input_size.get("image_height", 64)
        image_width = input_size.get("image_width", 64)
        channels = input_size.get("channels", 3)

        # Create datasets
        train_dataset, test_dataset = create_dataset(
            dataset_dir=dataset_dir,
            classes_list=classes_list,
            num_files=num_files,
            sequence_length=sequence_length,
            test_size=test_size,
            seed=seed,
            image_height=image_height,
            image_width=image_width,
            channels=channels,
        )

        logging.info("Data ingested successfully.")
        logging.info("Saving ingested data.")

        # Create directories
        train_ds_dir = os.path.join(processed_data_dir, "train_dataset")
        test_ds_dir = os.path.join(processed_data_dir, "test_dataset")

        os.makedirs(train_ds_dir, exist_ok=True)
        os.makedirs(test_ds_dir, exist_ok=True)

        # Save datasets if folder is empty (it should be empty if no run has been made)
        if len(os.listdir(train_ds_dir)) == 0:
            train_dataset.save(train_ds_dir)
            test_dataset.save(test_ds_dir)

        logging.info("Data saved successfully.")

        return train_ds_dir, test_ds_dir

    except Exception as err:
        raise CustomException(err, sys)
