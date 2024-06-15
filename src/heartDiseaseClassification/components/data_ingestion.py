import os
import urllib.request as request
import zipfile
from heartDiseaseClassification import logger
from heartDiseaseClassification.utils.common import get_size
from heartDiseaseClassification.entity.config_entity import DataIngestionConfig
from pathlib import Path
import ssl


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            # Create an SSL context that doesn't verify the server's certificate
            ssl_context = ssl._create_unverified_context()

            # Use the SSL context when calling urlopen
            with request.urlopen(self.config.source_URL, context=ssl_context) as response, open(self.config.local_data_file, 'wb') as out_file:
                data = response.read()  # a `bytes` object
                out_file.write(data)

            logger.info(f"{self.config.local_data_file} downloaded!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
