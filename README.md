# Identifying chilhood cancer patients who have receiver care in mental health related unit

This folder contains all scipts used in the LERACA project. TurkuNLP neural parsers can be found [here](https://github.com/TurkuNLP/Turku-neural-parser-pipeline)

## Pipeline
1. Create base datasets with module `create_dataset.py` function `create_dataset`from `dataset_helpers.py` module
    - Create dataset either only with cancer or both diabetes and cancer patients
    - Late effect used is 356 days &rarr; patients that have mental health related contact after 356 days from cancer/diabetes diagnose is labeled as 1, others to 0
    - If patient belongs to late effect group, notes one week before the contact are removed to reduce bias in model learning
    - After labeling all patient texts are lemmatized with TurkuNLP neural parser (found from `lemmatizer.py`)
    - Finally count matrix is created from lemmatized data with sklearn CountVectorizer
    - Two base dataset were created patients with cancer and patients with cancer or diabetes 
2. Create repeated nested cross validation folds with module `create_nested_cv_folds.py` function `create_folds` from `nested_cv_helpers.py` module
    - Each repeat is created with different random seed
    - Inside each repeat same seed is used for the splits (inner,outer and validation)
    - Test sets contain only cancer patients
3. Run repeated nested cross validation for three different ML algorithms: Random forest, Neural network and Logistic regression with `cancer_nested_cv.py`
    - NN model is implemented in `nn_model.py`
    - Repeats are computed with `scripts/`
4. Extracts results, create plots and calculate statistical significance of results using `statistical_analysis.py`
    - Significance Bayesian Correlated T-test
    - Calculate HDI with R package `bayestestR` with `ci` function

## Tests
- test_data.py - test that uids not outer folds are not found in inner in cross validation data
