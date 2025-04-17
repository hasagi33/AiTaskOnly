import json
import os

CONFIG_PATH = "config_data.json"

class Config:
    iteration_number = 1
    batch_size = 128
    epochs = 100
    dataset_rows = 1000
    LEARNING_RATE = 0.0001


    @classmethod
    def load(cls):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                cls.iteration_number = data.get("iteration_number", cls.iteration_number)
                cls.batch_size = data.get("batch_size", cls.batch_size)
                cls.epochs = data.get("epochs", cls.epochs)
                cls.dataset_rows = data.get("dataset_rows", cls.dataset_rows)
                cls.LEARNING_RATE = data.get("LEARNING_RATE", cls.LEARNING_RATE)


    @classmethod
    def save(cls):
        with open(CONFIG_PATH, "w") as f:
            json.dump({
                "iteration_number": cls.iteration_number,
                "batch_size": cls.batch_size,
                "epochs": cls.epochs,
                "dataset_rows": cls.dataset_rows,
                "LEARNING_RATE": cls.LEARNING_RATE

            }, f, indent=2)