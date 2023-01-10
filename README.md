# Identifying childhood cancer patients who have receiver care in a mental health-related unit

This folder contains all scripts used in the LERACA project. TurkuNLP neural parser can be found [here](https://github.com/TurkuNLP/Turku-neural-parser-pipeline)

## Pipeline
1. Create base datasets with module `create_dataset.py` function `create_dataset`from `dataset_helpers.py` module
    - Create a dataset either only with cancer or both diabetes and cancer patients
    - Late effect used is 356 days &rarr; patients that have mental health-related contact after 356 days from cancer/diabetes diagnosis are labelled as 1, others to 0
    - If the patient belongs to the late effect group, notes one week before the contact are removed to reduce bias in model learning
    - After labelling, all patient texts are lemmatized with TurkuNLP neural parser (found from `lemmatizer.py`)
    - Finally, the count matrix is created from lemmatized data with sklearn CountVectorizer
    - Two base datasets were created for patients with cancer and patients with cancer or diabetes 
2. Create repeated nested cross-validation folds with module `create_nested_cv_folds.py` function `create_folds` from the `nested_cv_helpers.py` module
    - Each repeat is created with a different random seed
    - Inside each repeat, the same seed is used for the splits (inner, outer and validation)
    - Test sets contain only cancer patients
3. Run repeated nested cross-validation for three different ML algorithms: Random forest, Neural network and Logistic regression with `cancer_nested_cv.py`
    - NN model is implemented in `nn_model.py`
    - Repeats are computed with `scripts/`
4. Extracts results, create plots and calculate the statistical significance of results using `statistical_analysis.py`
    - Significance Bayesian Correlated T-test and sampling probabilities for Credibility Interval (CI) calculation
5. Use `hdi_cancer_markdown.Rmd` to calculate HDIs and CIs with R package `bayestestR` with the `ci` function
