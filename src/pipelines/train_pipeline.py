
def sample_pipeline():
    # import packages
    from src.utils.loader_utils import config_loader
    from src.components.data_ingestion import ingest_data
    from src.components.trainer import train_model
    from src.exception import CustomException
    from src.logger import logging
    
    # start train pipeline
    logging.info("Initializing Training Pipeline")
    config_path = "configs/config.yaml"
    train_ds_path, test_ds_path = ingest_data(config_path=config_path)
    accuracy, model_path = train_model(train_ds_path, test_ds_path, config_path)
    logging.info("Training Pipeline completed successfully.")
    print(accuracy)

# run pipeline
if __name__ == "__main__":
    sample_pipeline()
