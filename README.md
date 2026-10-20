# NLP Sentiment Analysis Pipeline

A configurable NLP project for binary sentiment classification using **Amazon product reviews and Twitter sentiment data**. The project explores text preprocessing, alternative text representations, classical machine-learning models, experiment tracking, and API-based inference.

## Project Objectives

- Build reusable text preprocessing pipelines for different text domains.
- Experiment with different text vectorization and representation techniques.
- Train and evaluate sentiment classification models.
- Track experiments and metrics with MLflow.
- Define reproducible training workflows with DVC.
- Expose sentiment predictions through a FastAPI REST API.

## Pipeline Overview

1. **Data Loading:** Load the selected dataset using configurable paths and settings.
2. **Text Preprocessing:** Apply text-cleaning and normalization steps appropriate to the dataset.
3. **Feature Representation:** Transform text into numerical features.
4. **Model Training:** Train and compare classical machine-learning classifiers.
5. **Evaluation:** Assess model performance using classification metrics.
6. **Experiment Tracking:** Log experiment parameters and metrics with MLflow.
7. **Artifact Management:** Save model artifacts and evaluation metrics.
8. **Inference:** Serve predictions through FastAPI.

## Models and Text Representations

The project includes support for:

**Text representations**
- Bag of Words (BoW)
- TF-IDF
- BM25
- Word embeddings

**Machine-learning models**
- Logistic Regression
- Random Forest
- Gaussian Naive Bayes

The active configuration is controlled through `src/params.yaml`.

## Tech Stack

- **Programming:** Python
- **Data Processing:** pandas, NumPy
- **NLP:** NLTK
- **Machine Learning:** scikit-learn, Gensim
- **Experiment Tracking:** MLflow
- **Pipeline and Data Versioning:** DVC
- **API Development:** FastAPI, Uvicorn
- **Containerization:** Docker

## Repository Structure

```text
NLP_project/
├── Notebooks/
├── src/
│   ├── api/
│   ├── data_loader/
│   ├── preprocessing/
│   ├── train/
│   ├── vectorization/
│   └── params.yaml
├── dvc.yaml
├── metrics.json
├── dockerfile
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13 or a compatible version
- uv
- Git
- Access to the required datasets

### Installation

```bash
git clone https://github.com/NouranG/NLP_project.git
cd NLP_project
uv sync
```

### Configure the Data

Place the required datasets in the locations specified in `src/params.yaml`. Update the paths and dataset settings to match your local environment.

If you use the word-embedding configuration, ensure the required embedding file is available at its configured path.

### Train the Model

```bash
uv run python -m src.train.train
```

The training configuration determines the dataset, preprocessing, text representation, and model settings. The training script generates evaluation metrics and logs experiment information to MLflow.

### Run the DVC Pipeline

```bash
uv run dvc repro
```

Ensure the required data and DVC configuration are available before running the pipeline.

### Run the API

After training and generating the model artifact expected by the API, start the service:

```bash
uv run uvicorn src.api.main:app --reload
```

The API provides endpoints for service status, health checks, and sentiment prediction. Refer to `src/api/main.py` for the exact request and response schemas.

## Evaluation

The project uses classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score

Actual performance depends on the dataset, preprocessing strategy, feature representation, and selected model. Refer to the committed metrics and experiment results for the evaluated configuration.

## Reproducibility and Notes

- Use configuration files to keep experiments consistent and reproducible.
- Keep datasets and large embedding files out of Git unless their distribution is permitted.
- Do not commit credentials or private configuration files.
- Record the dataset version and experiment configuration when comparing results.

This project demonstrates an end-to-end workflow for NLP, classical machine learning, experiment tracking, and model serving.
