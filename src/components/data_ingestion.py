from src.utils.loader_utils import create_dataset


from src.exception import CustomException
from src.logger import logging


def ingest_data(config):
    dataset_dir = config["dataset_dir"]
    classes_list = config["classes_list"]
    num_files = config["num_files"]
    sequence_length = config["sequence_length"]
    seed = config["seed"]
    test_size = 0.25
    try:
        logging.info("Initializing data ingestion.")
        
        dataset = create_dataset(
            dataset_dir=dataset_dir, classes_list=classes_list,
            num_files=num_files, sequence_length=sequence_length,
            seed=seed
            )
        
        buffer_size = dataset.cardinality()
        train_size = int((1-test_size) * buffer_size)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)

        logging.info("Data ingested successfuly.")
        
        return train_dataset, test_dataset
    
    except Exception as err:
        raise CustomException(err, sys)