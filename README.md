# ivchenkov.yap

## AmesHousing Prediction project
The goal of this project is to create a service
for predicting real estate prices and is based on AmesHousing dataset.
The work was done as part of the course 'Software Engineering in Machine Learning' in MIPT.

GitLab Pages Website: https://se_ml_course.gitlab.io/2021/ivchenkov.yap
## Data
AmesHousing Dataset: http://jse.amstat.org/v19n3/decock/AmesHousing.txt

Paper: http://jse.amstat.org/v19n3/decock.pdf


## Installation
Sadly enough, right now there is no way to install this project :(

(But it should be fixed very soon)

### Dependencies

AmesHousing Prediction project requires the following dependencies:

    * Python (>= 3.9)
    * NumPy (>= 1.18.1)
    * Pandas (>= 0.25.3)
    * Scikit-learn (>= 0.23.2)

To install all dependencies, run:

``` 
pip install -r requirements.txt 
``` 

## Commands 

#### train.py 

Script for training a model based on sklearn 'train' command. 

#### test.py 

Script for model evaluation. The test dataset should have the same format as the train dataset. 

## Results 

R<sup>2</sup> on test dataset (0.2 of all dataset): 0.903 

MSE on test dataset (while predicting log target) 0.0152
