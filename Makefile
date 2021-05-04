.PHONY: add-remote clean requirements data train predict test sync-data-to-gdrive sync-data-from-gdrive

PYTHON_INTERPRETER = python3
BUCKET=1OSz5yQSK49qBmYO-CntIM_iKM8lW9MgW

include .env

## Add GDrive file storage as DVC remote
add-remote: create_environment
	dvc remote add -f -d mipt_drive gdrive://$(BUCKET)

## Setup environment for python (all processes run in venv)
create_environment:
	@echo $(PYTHON_INTERPRETER)
	$(PYTHON_INTERPRETER) -m venv .venv
	. .venv/bin/activate
	$(PYTHON_INTERPRETER) setup.py install
	$(PYTHON_INTERPRETER) -m pip install -U pip 

## Remove all cached and compile Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Install project requirements
requirements: create_environment
	$(PYTHON_INTERPRETER) -m pip install -r requirements-test.txt

## Create train and test samples, and store them in GDrive with DVC
data: requirements add-remote
	$(PYTHON_INTERPRETER) ./scripts/split.py \
		--run-name ${RUN_NAME} \
		--data-path ${RAW_DATA} \
		--log-path ${LOG_PATH}
	dvc add -R $(DATA_PATH)
	dvc add -R $(LOG_PATH)
	dvc commit
	dvc push

## Pull latest file versions from GDrive storage
sync-data-from-gdrive:
	dvc pull -R data/

## Push files to GDrive storage
sync-data-to-gdrive:
	dvc push -R data/ -r mipt_drive

## Train model and store all data and info in GDrive with DVC
train: data
	$(PYTHON_INTERPRETER) ./scripts/train.py \
		--run-name ${RUN_NAME} \
		--data-path ${PROCESSED_TRAIN_DATA} \
		--models-path ${MODELS_PATH} \
		--log-path ${LOG_PATH}
	dvc add -R ${MODELS_PATH}
	dvc add -R ${LOG_PATH}$
	dvc commit
	dvc push -r mipt_drive

## Make predictions using trained model and store all info in GDrive with DVC
predict: train
	$(PYTHON_INTERPRETER) ./scripts/test.py \
        	--run-name ${RUN_NAME} \
        	--data-path ${PROCESSED_TEST_DATA} \
        	--results-path ${RESULTS_PATH} \
        	--models-path ${MODELS_PATH} \
        	--log-path ${LOG_PATH}
	dvc add -R ${RESULTS_PATH}
	dvc add -R ${LOG_PATH}
	dvc commit
	dvc push -r mipt_drive

## Run tests (with creation of different reports)
test: data
	pytest --cov=housinglib \
		--cov-branch \
		--cov-report term-missing \
		--cov-report xml:/content/coverage.xml \
		--junitxml=/content/report.xml \
		test

.DEFAULT: help
## Show help board
help:
	@echo "usage: make {add-remote,clean,requirements,data,train,predict,test,sync-data-to-gdrive,sync-data-from-gdrive}"
	@echo "make add-remote"
	@echo "			add GDrive as DVC remote"
	@echo "make clean"
	@echo "			clean Python cache"
	@echo "make requirements"
	@echo "			install Python dependencies"
	@echo "make data"
	@echo "			split and preprocess data"
	@echo "make train"
	@echo "			train model"
	@echo "make predict"
	@echo "			make predictions with trained model"
	@echo "make test"
	@echo "			run tests"
	@echo "make sync-data-to-gdrive"
	@echo "			update GDrive files"
	@echo "make sync-data-from-gdrive"
	@echo "			update local files"






