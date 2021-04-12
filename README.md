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

# ames_housing Docker container

This project also provides Docker container with basic functions and their Flask interface.

## Getting Started

Following instructions will cover usage information for the ames_housing container.

### Prerequisites

In order to run this container you'll need Docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)

### Usage

#### Container Parameters

Run Flask app

```shell
docker run -d -p 5000:5000 registry.gitlab.com/se_ml_course/2021/ivchenkov.yap:latest
```

#### Useful File Locations

* `/scripts/train.py` - model training script

* `/scripts/test.py` - model testing script
 
* `/scripts/split.py` - creation of train/test samples script
  
* `/src/housinglib` - library directory

## Built With

* Python 3.9.2
* Python libraries, complete list of which is in requirements.txt

## Find Me

* [GitHub](https://github.com/vinnibuh/AmesHousing)

For the versions available, see the 
[tags on this repository](https://gitlab.com/se_ml_course/2021/ivchenkov.yap).

## Authors

* **Ivchenkov Yaroslav**
