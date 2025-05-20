import json
import os

CONFIG_PATH = \
    "config_data.json"


class Config:
    iteration_number = 1
    model_number = 1
    batch_size = 16
    epochs = 2
    dataset_rows = 5000
    LEARNING_RATE = 0.0001
    current_folder = ""
    current_loss=1
    current_layers=1
    loss_description= ""
    layers_description=""

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
                cls.current_folder = data.get("current_folder", cls.current_folder)
                cls.current_loss = data.get("current_loss", cls.current_loss)
                cls.current_layers = data.get("current_layers", cls.current_layers)
                cls.loss_description = data.get("loss_description", cls.loss_description)
                cls.layers_description=data.get("layers_description", cls.layers_description)

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
                "current_folder": str(cls.current_folder),
                "current_loss": cls.current_loss,
                "current_layers": cls.current_layers,
                "loss_description": cls.loss_description,
                "layers_description": cls.layers_description,
            }, f, indent=2)
