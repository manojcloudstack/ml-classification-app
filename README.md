# Breast Cancer Classification

## a. Problem Statement
Build and compare five classification models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) to predict whether a breast tumour is **malignant or benign** from measurements of cell nuclei taken from digitised images of a fine-needle aspirate (FNA). The models are evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC, and demonstrated through an interactive Streamlit web app.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic) – UCI Machine Learning Repository
- **Type:** Binary classification (0 = Malignant, 1 = Benign)
- **Instances:** 569 (requirement ≥ 500) | **Features:** 30 numeric (requirement ≥ 12)
- **Features:** mean, standard error and "worst" value of 10 nucleus properties: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
- **Class balance:** 357 benign / 212 malignant. No missing values.
- **Split:** 80/20 stratified (random_state=42) → 455 train / 114 test. The 20% test set is saved as `test_data.csv`.
- **Preprocessing:** StandardScaler inside sklearn Pipelines for Logistic Regression, KNN and SVM (so the scaler is saved with the model). Decision Tree and Naive Bayes use raw features.

## c. GitHub Repository Link
`https://github.com/manojcloudstack/ml-classification-app`

**Live Streamlit app:** `https://ml-classification-app-54irzdukwipczzul3knqgc.streamlit.app/`

```
project-folder/
│-- app.py
│-- train_models.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
│   ├-- logistic_regression.joblib
│   ├-- decision_tree.joblib
│   ├-- knn.joblib
│   ├-- naive_bayes.joblib
│   ├-- support_vector_machine.joblib
│   └-- metrics_summary.csv
```

## d. Models Used

### Comparison Table (test set, 114 rows)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|:-----------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression    | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree          | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| KNN                    | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes            | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Best overall (tied with SVM on accuracy/F1/MCC, highest AUC 0.9954). The classes are close to linearly separable after scaling, so a simple linear model works very well and is also the most interpretable. |
| Decision Tree | Weakest model (accuracy 0.921, AUC 0.916, MCC 0.834). A single depth-5 tree has high variance and overfits the small training set; its hard splits also give poorly ranked probabilities, hence the low AUC. |
| KNN | Good (accuracy 0.956, recall 0.972). Performs well because features are scaled, but it is slightly below LR/SVM as distance-based voting is sensitive to the 30-dimensional space and the choice of k. |
| Naive Bayes | Decent accuracy (0.939) but a very high AUC (0.988), meaning it ranks cases well but its probability threshold is poorly calibrated. The independence assumption is violated because many features (radius, perimeter, area) are strongly correlated. |
| Support Vector Machine | Ties with Logistic Regression (accuracy 0.9825, MCC 0.9623, AUC 0.9950). The RBF kernel on scaled data separates the classes with a clean margin; costlier to train and less interpretable than LR. |
| **Overall winner** | **Logistic Regression** – joint-highest accuracy, F1 and MCC with the best AUC, while being the simplest and fastest model. SVM is a very close runner-up. |

## Streamlit App Features
1. CSV upload of test data (features + `target` column)
2. Dropdown to select any of the 5 saved models
3. Display of Accuracy, AUC, Precision, Recall, F1, MCC
4. Confusion matrix and classification report
5. Comparison table of all 5 models on the uploaded data

Models are loaded from `model/` – **no retraining** happens in the app.

## Run Locally
```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates models + test_data.csv
streamlit run app.py
```
