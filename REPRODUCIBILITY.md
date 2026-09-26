\# Reproducibility Information



\## 1. Overview



This repository provides the implementation and evaluation code for the

proposed dynamic load-balancing framework using reinforcement learning and

pheromone-based optimization.



The repository includes the main implementation, evaluation scripts,

statistical validation code, experimental results, dependency information,

and execution instructions.



\## 2. Dataset



The experiments use the Google 2019 Cluster Sample dataset.



Dataset source:

https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample



Download the dataset from the above source and place the required

`borg\_traces\_data.csv` file in the same directory as `main.py`.



The original dataset is not redistributed in this repository because it is

available from the external dataset provider.



\## 3. Preprocessing



The preprocessing procedure is integrated directly into `main.py`.



It includes:



\- Removal of unnecessary columns

\- Conversion of relevant attributes to numeric data types

\- Extraction of CPU and memory resource requests

\- Processing of CPU usage information

\- Handling of missing values

\- Selection of features used by the scheduler

\- Generation of edge/cloud labels

\- Feature standardization

\- Stratified train-test splitting



Therefore, a separate preprocessing script is not required.



\## 4. Experimental Configuration



The main experimental configuration is defined directly in `main.py`.



Important settings include:



\- Train/test split: 80% / 20%

\- Hidden size: 32

\- Optimizer: Adam

\- Learning rate: 0.00003

\- Number of epochs: 35

\- Batch size: 128

\- Classification threshold: 0.35



\## 5. Random Seeds and Statistical Validation



Five independent experimental runs are performed using the following seeds:



42, 43, 44, 45, 46



The file `statistical\_validation.py` executes `main.py` independently for

each seed and records the experimental performance.



The random seed is applied to:



\- Python random module

\- NumPy

\- PyTorch

\- Train-test splitting



\## 6. Result Files



The repository provides the following experimental result files:



\- `validation\_results.csv` - results obtained from each independent run

\- `validation\_summary.csv` - mean, standard deviation, and 95% confidence

&#x20; interval calculated across the five runs

\- `results.docx` - additional reported experimental results



The statistical validation script automatically regenerates the two CSV

files.



\## 7. Running the Experiment



Install the required dependencies using:



pip install -r requirements.txt



Place `borg\_traces\_data.csv` in the repository directory.



Run the main experiment using:



python main.py



Run the five-seed statistical validation using:



python statistical\_validation.py



The statistical validation generates:



\- validation\_results.csv

\- validation\_summary.csv



\## 8. Reproducibility Scope



The released code supports reproduction of the preprocessing, model

training, scheduling procedure, evaluation metrics, and five-run

statistical validation reported by the implementation.



The original dataset itself is not redistributed through this repository.

It must be downloaded separately from the dataset source specified above.



Runtime-dependent measurements, particularly execution and training time,

may vary depending on hardware, operating system, background processes,

and software environment.



\## 9. Repository Contents



\- `main.py` - main implementation and integrated preprocessing

\- `evaluation.py` - evaluation procedures

\- `statistical\_validation.py` - five-run statistical validation

\- `validation\_results.csv` - individual run results

\- `validation\_summary.csv` - statistical summary

\- `results.docx` - additional experimental results

\- `requirements.txt` - required Python packages

\- `README.md` - project overview and dataset information

\- `How to run the code.docx` - execution instructions

\- `REPRODUCIBILITY.md` - reproducibility information

