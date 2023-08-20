# Kaggle NLP with Disaster Tweets Competition

This repository contains the code and resources for my submissions to Kaggle's [NLP with Disaster Tweets Competition](https://www.kaggle.com/c/nlp-getting-started).

## Notebooks

### 1. Neural Network
- [Notebook_NN](https://github.com/zsoltgeier/NLP-with-disaster-tweets/blob/main/Notebooks/naive_bayes_solution.ipynb): This Jupyter notebook contains the code for my second highest scoring scoring submission using a custom **Neural Network** model.

### 2. TF-IDF and Multinomial Naive Bayes
- [Notebook_NB](https://github.com/zsoltgeier/NLP-with-disaster-tweets/blob/main/Notebooks/naive_bayes_solution.ipynb): This Jupyter notebook contains the code for my top scoring submission using **TF-IDF** vectorization and a **Multinomial Naive Bayes** model.

## Streamlit App

I have also deployed a Streamlit app to showcase my **Neural Network** submission. You can check it out here: https://nlp-with-disaster-tweets.streamlit.app

## Cleaned Python Scripts

You can find the cleaned python scripts of the notebooks in the [scripts](https://github.com/zsoltgeier/NLP-with-disaster-tweets/tree/main/Scripts) directory.

## Model and Vectorization Combinations

Here's a summary of the scores achieved with different model and vectorization combinations:

| Model                        | Vectorization              | Accuracy   |
| ---------------------------- | -------------------------- | ---------- |
| Multinomial Naive Bayes      | TF-IDF                     | **79.83%** |
| Neural Network               | Tokenization and Padding   | **79.62%** |
| Multinomial Naive Bayes      | CountVectorizer            | **78.33%** |
| Logistic Regression          | CountVectorizer            | **73.49%** |
| Logistic Regression          | TF-IDF                     | **71.19%** |

## Requirements

To run the notebooks and scripts, make sure to install the required packages listed in [requirements.txt](https://github.com/zsoltgeier/NLP-with-disaster-tweets/blob/main/requirements.txt) using the following command:

```bash
pip install -r requirements.txt
