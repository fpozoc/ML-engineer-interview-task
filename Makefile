.PHONY: help create_environment requirements train predict
.DEFAULT_GOAL := help

NO_OF_TEST_FILES := $(words $(wildcard tests/test_*.py))
NO_OF_REPORT_FILES := $(words $(wildcard reports/))
NO_OF_REPORT_FILES := $(words $(filter-out reports/.gitkeep, $(SRC_FILES)))
DATASET := data/raw/immo_data.csv

install: ## install dependencies
	pip install -e ".[test, extra]"

clean: ## clean artifacts
	@echo ">>> cleaning files"
	rm data/processed/* models/*.pkl || true

make-dataset:
	@echo ">>> generating dataset"
	python src/data/make_dataset.py --dataset data/raw/immo_data.csv --output data/processed/training_set.v1.tsv.gz

make-dataset-text:
	@echo ">>> translating and creating nlp tasks for dataset 2"
	python -m src.features.text --dataset data/raw/immo_data.csv --output data/processed/training_set.v2.tsv.gz

train: ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	python src/model/train.py  --dataset data/processed/training_set.v1.tsv.gz --model_selection --evaluation R2

train-text: ## train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
	@echo ">>> training model"
	python src/model/train.py  --dataset data/processed/training_set.v2.tsv.gz --model_selection --evaluation R2

# serve: ## serve trained model with a REST API using dploy-kickstart to-be-done
# 	@echo ">>> serving the trained model"
# 		 serve -e src/model/predict.py -l .

explain: ## create an ExplainerDashboard for the model interpretability task
	@echo ">>> training model"
	python src/model/explainer.py --dataset data/processed/training_set.v1.tsv.gz --model models/model.v.1.0.0.pkl

run-pipeline: install clean make-dataset train serve  ## install dependencies -> clean artifacts -> generate dataset -> train -> serve

lint: ## flake8 linting and black code style
	@echo ">>> black files"
	black src tests
	@echo ">>> linting files"
	flake8 src tests

coverage: ## create coverage report
	@echo ">>> running coverage pytest"
	pytest --cov=./ --cov-report=xml

test: ## run unit tests in the current virtual environment
	@echo ">>> running unit tests with the existing environment"
	pytest

test-docker: ## run unit tests in docker environment
	@echo ">>> running unit tests in an isolated docker environment"
	docker-compose up test

help: ## show help on available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
