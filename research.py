#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import shutil
import time
import sys
import platform

from config import Config

Config.load()

def get_python_command():
    """Determine the best Python command to use based on the platform"""
    # Try different Python commands and use the first one that works
    commands = ["python", "py", "python3"]
    
    for cmd in commands:
        try:
            # Test if the command works
            result = subprocess.run(f"{cmd} --version", 
                                   shell=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"Using Python command: {cmd}")
                return cmd
        except:
            pass
    
    # If none worked, return default based on platform
    if platform.system() == "Windows":
        return "python"
    else:
        return "python3"

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {elapsed:.2f} seconds")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
        exit(result.returncode)

def clean_directories():
    """Remove contents of models folder and other generated data"""
    print("\n🗑️  Cleaning up previous files...")

    paths = ["./processed", "./realistic_tasks_large.csv", "./models"]

    # Add all contents inside ./models to be deleted
    models_path = Path("./models")
    if models_path.exists() and models_path.is_dir():
        for item in models_path.iterdir():
            paths.append(str(item))

    Config.batch_size=16
    Config.dataset_rows=1000
    Config.iteration_number=1
    Config.model_number=1
    Config.current_folder=""
    Config.save()
    # Delete all paths
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
            else:
                print(f"Removing file: {path}")
                os.remove(path)

    # Optionally recreate the empty ./models folder
    if not models_path.exists():
        os.makedirs(models_path)
        print("🆕 Recreated empty './models' directory.")

def main():
    """Main function to orchestrate the retraining process"""
    print("\n🚀 Starting complete model retraining process")
    
    # Get the appropriate Python command for this system
    python_cmd = get_python_command()
    original_folder = Config.current_folder
    num1 = sys.argv[1]

    if num1=="research":
        clean_directories()
        base_folder = Path("models")
        os.makedirs(base_folder, exist_ok=True)

        for i in range(10):

            Config.current_folder = base_folder / f"size_{Config.iteration_number}"
            os.makedirs(Config.current_folder, exist_ok=True)

            print(Config.current_folder)
            print(Config.current_folder)

            run_command(f"{python_cmd} dataset_generator.py {Config.current_folder}", "Dataset Generation")
            run_command(f"{python_cmd} preprocess_dataset.py {Config.current_folder}", "Data Preprocessing")
            for j in range(3):
                print(f"Starting training {Config.model_number}:")
                run_command(f"{python_cmd} train_model.py {Config.current_folder}", "Model Training")
                Config.model_number+=1
                Config.batch_size*=4
                Config.save()
            Config.batch_size=16
            Config.dataset_rows*=10
            Config.iteration_number+=1
            Config.model_number=0
            Config.save()
    # elif num1!="old":            #optional skip the dataset generation
    #     # Step 1: Clean up
    #     clean_directories()
    #     # Step 2: Generate dataset
    #     run_command(f"{python_cmd} dataset_generator.py", "Dataset Generation")
    #     # Step 3: Preprocess dataset
    #     run_command(f"{python_cmd} preprocess_dataset.py", "Data Preprocessing")
    #     # Step 4: Train model
    #     run_command(f"{python_cmd} train_model.py", "Model Training")
    elif num1=="clean":
        clean_directories()
        return

    print("\n✨ Retraining process completed successfully!")
    print("You can now use the new model for predictions.")

if __name__ == "__main__":
    main()