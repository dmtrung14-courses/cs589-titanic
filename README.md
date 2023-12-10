# CS 589 Final Project Repository Guideline


Student: Trung Dang\
SPIRE ID: 33858723

## Table of Contents

- [CS 589 Final Project Repository Guideline](#cs-589-final-project-repository-guideline)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [References](#references)


## Code Structure


The code structure for this repository is as follows:

- `README.md`: This file contains the guideline for the CS 589 Final Project.
- `data/`: This directory contains any necessary data files for the project.
- `modules/`: The backbone of the project. All datapreprocessing files are contained here.
- `models/`: Contains the final model and the training plots 
    - `ckpt`: Contains `.pkl` files of training parameters
    - `plt`: Contains relevant plot for the report
    - `res`: Contains the `.csv` files used for submission to Kaggle

Please refer to the specific files and directories for more details.

## Usage

Install the required packages using: 

```shell
pip install -r requirements.txt
```

To train the model from scratch, run
```
python titanic.py
```

To generate the predictions for the test set, run: 
```
python inference.py
```

Generally inference.py will conclude instantaneously as a model has been provided in the submission. However, should the grader decide to remove the ckpt files and train the model from scratch, the file will perform all trainings like `titanic.py`, then save the inference to `models\res`

## References

- [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [CS589 Final Project Guideline](https://docs.google.com/document/d/1nvYTFs8qlL3GRzsLMCTCEUKXskOpnqjvZ_fRvjsgyIQ/edit?usp=sharing)