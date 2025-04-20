import json
import os

CONFIG_PATH = "config_data.json"

class Config:
    iteration_number = 1
    model_number = 1
    batch_size = 16
    epochs = 10
    dataset_rows = 10
    LEARNING_RATE = 0.0001
    current_folder=""


    @classmethod
    def load(cls):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                cls.iteration_number = data.get("iteration_number", cls.iteration_number)
                cls.model_number = data.get("model_number", cls.model_number)
                cls.batch_size = data.get("batch_size", cls.batch_size)
                cls.epochs = data.get("epochs", cls.epochs)
                cls.dataset_rows = data.get("dataset_rows", cls.dataset_rows)
                cls.LEARNING_RATE = data.get("LEARNING_RATE", cls.LEARNING_RATE)
                cls.current_folder= data.get("current_folder", cls.current_folder)


    @classmethod
    def save(cls):
        with open(CONFIG_PATH, "w") as f:
            json.dump({
                "iteration_number": cls.iteration_number,
                "model_number": cls.model_number,
                "batch_size": cls.batch_size,
                "epochs": cls.epochs,
                "dataset_rows": cls.dataset_rows,
                "LEARNING_RATE": cls.LEARNING_RATE,
                "current_folder": str(cls.current_folder)
            }, f, indent=2)