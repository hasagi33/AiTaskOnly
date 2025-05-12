# AiTaskOnly

**AiTaskOnly** is a Python-based framework for training and evaluating multiple shallow learning models across different dataset sizes and batch configurations. It is designed for large-scale experimentation to measure model performance under varied training conditions.

This repository is a subrepository of [HananB27/AI](https://github.com/HananB27/AI) and is specifically used for research and experimentation with the task duration prediction model presented there.

---

## Project Structure

```bash
AiTaskOnly/
├── models/
│   ├── size_1/           # 100 rows
│   ├── size_2/           # 1,000 rows
│   ├── size_3/           # 10,000 rows
│   ├── size_4/           # 100,000 rows
│   └── size_5/           # 1,000,000 rows
│       ├── dataset.csv           # Original dataset
│       ├── model_0/              # Model trained with batch size 16
│       ├── model_1/              # Model trained with batch size 256
│       ├── model_2/              # Model trained with batch size 4096
│       ├── processed/            # Scaled inputs and labels
│       ├── pickle_loss/          # Pickle files for loss graphs
│       └── batches/              # .bat files to visualize graphs
├── activations.py                # Activation functions
├── config.py                     # Configuration logic
├── config_data.json              # Stores run settings
├── dataset_generator.py          # Generates synthetic task datasets
├── dense_layer.py                # Implementation of dense layers
├── CustomHuberLoss.py                     # Loss functions
├── predict.py                    # Make predictions from trained models
├── preprocess_dataset.py         # Preprocessing utilities
├── research.py                   # Main driver script - trains all 15 models
├── showfig.py                    # Graphing utility (reads pickle_loss)
├── task_input.py                 # Input encoder and formatter
├── task_model.py                 # Model architecture
├── team_optimizer.py             # Experimental optimization logic
├── train_model.py                # Trains a single model and saves graph
├── *.bat                         # Batch files to launch graphs from .fig.pickle
```

---

## Getting Started

### Requirements

- Python 3.11+
- NumPy, Matplotlib, TQDM, Pandas
- No external ML frameworks used

### Installation

```bash
git clone https://github.com/hasagi33/AiTaskOnly.git
cd AiTaskOnly
pip install -r requirements.txt
```

---

## Training Models

To train 15 models with combinations of 5 dataset sizes and 3 batch sizes:

```bash
python research.py research
```

To clear all previously trained models and start fresh, run:

```bash
python research.py clean
```

This will:

- Load datasets of sizes: 100, 1k, 10k, 100k, 1M
- Train 3 models per dataset with batch sizes:
  - `model_0` = batch size 16
  - `model_1` = batch size 256
  - `model_2` = batch size 4096
- Save:
  - Trained weights
  - Scaled input/output
  - Loss graphs as `.fig.pickle` files

---

## Visualizing Loss

After training, loss graphs are saved in `pickle_loss/` as `.fig.pickle`.

You can visualize them using the pre-created `.bat` batch files in `batches/`:

```bash
models/size_3/batches/view_loss_plot_1.bat
```

These run the graph viewer using `pythonw.exe` so the user only sees the graph window, simulating an executable.

---

## Model Evaluation and Predictions

To evaluate all 15 trained models and see their prediction precision on a sample task, run:

```bash
python predict.py
```

This will:
- Load each of the 15 trained models (5 dataset sizes × 3 batch sizes)
- Predict on a sample of 100 tasks from each dataset
- Calculate the **Mean Absolute Percentage Error (MAPE)** for each model
- Print the overall model precision percentage
- Output the predicted duration (in days and hours) for a fixed sample task with defined properties

---

## Configuration

The `config_data.json` file holds:
- Current dataset size
- Batch size
- Number of epochs
- Learning rate
- Model number
- Folder structure for logging

The `config.py` and `configPrint.py` handle dynamic loading and printing of these values.

---

## Architecture

The project implements a minimal neural network using:
- Custom dense layers (`dense_layer.py`)
- Activation functions like ReLU and Sigmoid (`activations.py`)
- Mean Squared Error and Huber Loss functions (`losses.py`)
- Fully vectorized forward and backward passes

---

## Notes on Models Folder

Each `size_n/` folder contains:
- `dataset.csv` – raw dataset
- `processed/` – NumPy arrays for training
- `model_0/`, `model_1/`, `model_2/` – models trained with 3 different batch sizes
- `pickle_loss/` – saved loss curves
- `batches/` – auto-generated batch files to open plots with `pythonw.exe`

---

## License

This project is licensed under the MIT License.

---

## Contributions

Feel free to fork the repo and submit pull requests to:
- Add model evaluation
- Include GUI tools
- Optimize architecture and memory usage

---

## Author

Maintained by [@hasagi33](https://github.com/hasagi33) and [@HananB27](https://github.com/HananB27)
