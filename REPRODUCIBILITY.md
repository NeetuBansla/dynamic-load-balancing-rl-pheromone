# Reproducibility Information

## 1. Overview

This repository provides the implementation and evaluation code for the proposed dynamic load-balancing framework using reinforcement learning and pheromone-based optimization.

The repository includes the main implementation, preprocessing script, evaluation scripts, statistical validation code, raw experimental results, figure data, generated figures, dependency information, and execution instructions.

## 2. Dataset

The experiments use the Google 2019 Cluster Sample dataset.

Dataset source:

https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

Download the dataset from the above source and place the required `borg_traces_data.csv` file in the repository directory.

The original dataset is not redistributed in this repository because it is available from the external dataset provider.

## 3. Preprocessing

The preprocessing procedure is provided separately in:

`preprocessing/preprocessing.py`

The preprocessing procedure includes:

* Removal of unnecessary columns
* Conversion of relevant attributes to numeric data types
* Extraction of CPU and memory resource requests
* Processing of CPU usage information
* Handling of missing values
* Selection of features used by the scheduler
* Generation of edge/cloud labels

The preprocessing script generates the processed workload used by the main experiment.

The processed workload output is stored in:

`preprocessing/processed_workload.csv`

The preprocessing function is called from `main.py`, allowing the analyzed workload to be regenerated from the original `borg_traces_data.csv` source file.

Feature standardization and stratified train-test splitting are subsequently performed in `main.py`.

## 4. Experimental Configuration

The main experimental configuration is defined in `main.py`.

Important settings include:

* Train/test split: 80% / 20%
* Hidden size: 32
* Optimizer: Adam
* Learning rate: 0.00003
* Number of epochs: 35
* Batch size: 128
* Classification threshold: 0.35

## 5. Random Seeds and Statistical Validation

Five independent experimental runs are performed using the following fixed seeds:

`42, 43, 44, 45, 46`

The statistical validation script is located at:

`validation/statistical_validation.py`

The script executes `main.py` independently for each of the five fixed seeds and records the experimental performance.

The random seed is applied to:

* Python random module
* NumPy
* PyTorch
* Train-test splitting

The resulting validation files are stored in the `validation/` directory:

* `validation/validation_results.csv` – individual results from the five independent runs
* `validation/validation_summary.csv` – statistical summary across the five runs

## 6. Raw Result Files

Raw numerical results generated from the experiments are provided in:

`raw_results/`

These CSV files contain the numerical results used for reporting and evaluating the proposed framework.

The five-run statistical validation results are separately available in:

`validation/`

This organization allows the reported experimental results and statistical analysis to be directly inspected and reproduced.

## 7. Figure Data and Generated Figures

The numerical data used to generate the experimental figures are provided as CSV files in:

`figure_data/`

These files contain the underlying numerical values used for the corresponding comparison plots.

The generated figure images are provided in:

`figures/`

This separation allows the reported figures to be regenerated and independently checked using the corresponding raw numerical data.

## 8. Running the Experiment

Install the required dependencies using:

`pip install -r requirements.txt`

Download `borg_traces_data.csv` from the dataset source specified above and place it in the repository directory.

Run the main experiment using:

`python main.py`

The preprocessing function is automatically called by `main.py`.

The preprocessing procedure can also be executed separately using:

`python preprocessing/preprocessing.py`

Run the five-seed statistical validation from the repository root using:

`python validation/statistical_validation.py`

The statistical validation generates:

* `validation/validation_results.csv`
* `validation/validation_summary.csv`

## 9. Reproducibility Scope

The released code supports reproduction of the preprocessing procedure, analyzed workload, model training, scheduling procedure, evaluation metrics, figures, and five-run statistical validation.

The original dataset itself is not redistributed through this repository. It must be downloaded separately from the dataset source specified above.

Runtime-dependent measurements, particularly execution and training time, may vary depending on hardware, operating system, background processes, and software environment.

## 10. Repository Contents

* `main.py` – main implementation and model execution
* `evaluation.py` – evaluation procedures
* `preprocessing/` – preprocessing script and processed workload
* `validation/` – five-run statistical validation script and validation CSV files
* `raw_results/` – raw experimental result files
* `figure_data/` – CSV data used to generate the experimental figures
* `figures/` – generated experimental figures
* `requirements.txt` – required Python packages
* `README.md` – project overview and dataset information
* `How to run the code.docx` – execution instructions
* `REPRODUCIBILITY.md` – detailed reproducibility information

## 11. Versioned Reproducibility Release

A versioned GitHub release is provided to preserve the exact implementation, configuration, preprocessing workflow, raw results, figure data, and statistical validation files used for the reported experiments.

Release tag:

`v1.0.3`
