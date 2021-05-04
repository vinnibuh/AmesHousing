.PHONY: clean requirements data train predict test sync-data-to-gdrive sync-data-from-gdrive

PYTHON_INTERPRETER = python3

include .env

create_environment:
	@echo $(PYTHON_INTERPRETER)
	$(PYTHON_INTERPRETER) -m venv .venv
	. .venv/bin/activate
	$(PYTHON_INTERPRETER) setup.py install
	$(PYTHON_INTERPRETER) -m pip install -U pip 

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

requirements: create_environment
	$(PYTHON_INTERPRETER) -m pip install -r requirements-test.txt

data: requirements
	$(PYTHON_INTERPRETER) ./scripts/split.py \
		--run-name ${RUN_NAME} \
		--data-path ${RAW_DATA} \
		--log-path ${LOG_PATH}
	dvc add -R $(DATA_PATH)
	dvc add -R $(LOG_PATH)
	dvc commit
	dvc push

sync-data-from-gdrive:
	dvc pull -R data/

sync-data-to-gdrive:
	dvc push -R data/ -r mipt_drive

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

test: data
	pytest --cov=housinglib \
		--cov-branch \
		--cov-report term-missing \
		--cov-report xml:/content/coverage.xml \
		--junitxml=/content/report.xml \
		test

.DEFAULT: help
help:
	@echo "usage: make {clean,requirements,data,train,predict,test,sync-data-to-gdrive,sync-data-from-gdrive}"
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






