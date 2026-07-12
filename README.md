# Customer Support Ticket Classification NLP

An end-to-end Natural Language Processing project that classifies customer support tickets into the correct support topic group using traditional machine learning.

The project covers data exploration, text feature engineering, baseline modeling, preprocessing experiments, error analysis, model comparison, final model saving, and deployment through a Streamlit web app.

## Project Objective

Customer support teams receive many text-based tickets every day. Manually routing each ticket to the correct support group can be slow and inconsistent.

This project builds a machine learning classifier that reads a ticket description and predicts one of the available `Topic_group` categories.

## Supported Topic Groups

The model is trained to predict 8 ticket categories:

| Topic Group | Meaning |
| --- | --- |
| Access | Login, account access, password, access card, and permission-related tickets |
| Administrative rights | Requests for elevated/admin privileges |
| Hardware | Laptop, workstation, phone, printer, scanner, and device issues |
| HR Support | Employee onboarding, starters, and HR-related support |
| Internal Project | Internal task, project code, and project support requests |
| Miscellaneous | General tickets that do not clearly fit one specific group |
| Purchase | Procurement, purchase order, equipment, and license requests |
| Storage | Mailbox, shared folder, drive, and storage-related tickets |

## Dataset

The processed dataset is stored in:

```text
Dataset/all_tickets_processed_improved_v3.csv
```

Dataset summary:

| Detail | Value |
| --- | --- |
| Total records | 47,837 |
| Columns | `Document`, `Topic_group` |
| Number of classes | 8 |

Class distribution:

| Topic Group | Records |
| --- | ---: |
| Hardware | 13,617 |
| HR Support | 10,915 |
| Access | 7,125 |
| Miscellaneous | 7,060 |
| Storage | 2,777 |
| Purchase | 2,464 |
| Internal Project | 2,119 |
| Administrative rights | 1,760 |

The dataset is imbalanced, so weighted evaluation metrics were used during model evaluation.

## Project Structure

```text
Customer Support Ticket Classification NLP/
|
|-- app.py
|-- Observations.txt
|-- Smart_IT_Support_Ticket_Classifier_Project_Plan_Updated.pdf
|
|-- Dataset/
|   |-- all_tickets_processed_improved_v3.csv
|
|-- Models/
|   |-- logistic_model.pkl
|   |-- tfidf_vectorizer.pkl
|
|-- Notebooks/
|   |-- data.csv
|   |-- baselineResults.csv
|   |-- 01.ipynb
|   |-- 02.ipynb
|   |-- 03_BaselineModel.ipynb
|   |-- 04_ErrorAnalysis.ipynb
|   |-- 05_TextPreprocessing.ipynb
|   |-- 06_ModelComparison.ipynb
|   |-- 07_FinalModel.ipynb
|
|-- Comparison/
|   |-- Comparison.docx
```

## Workflow

### 1. Data Understanding

The ticket text is stored in the `Document` column and the target class is stored in the `Topic_group` column.

During exploration, many tickets were found to contain common email-style words such as:

```text
dear, thanks, hello, hi, regards, please, good morning
```

These words appear across many categories and do not directly describe the actual technical issue.

### 2. Baseline Model

The first strong baseline used:

```text
Raw ticket text -> TF-IDF Vectorizer -> Logistic Regression
```

This baseline performed well because TF-IDF automatically reduces the importance of very common words and gives more weight to informative terms such as `vpn`, `printer`, `mailbox`, `password`, `oracle`, and `wireless`.

### 3. Text Preprocessing Experiments

A cleaning pipeline was tested by:

- converting text to lowercase
- removing punctuation
- removing numbers
- removing English stopwords
- removing custom conversational stopwords

The preprocessing experiments did not significantly improve overall performance. In some classes, performance slightly decreased.

Final decision: keep the original `Document` text representation with TF-IDF because it already handled common words effectively.

### 4. Error Analysis

Prediction results were saved in:

```text
Notebooks/baselineResults.csv
```

This file contains:

```text
Document, Actual, Predicted
```

It was used to inspect correct and incorrect predictions. The weakest class was `Administrative rights`, mainly because it has fewer samples and overlaps with categories such as `Access` and `HR Support`.

### 5. Model Comparison

Multiple models were compared using the same TF-IDF features:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 85.82% | 86.26% | 85.82% | 85.83% |
| Linear SVM | 85.61% | 85.73% | 85.61% | 85.62% |
| Random Forest | 83.09% | 84.21% | 83.09% | 83.05% |
| Multinomial Naive Bayes | 73.76% | 78.12% | 73.76% | 72.60% |

Logistic Regression achieved the best overall score and was selected as the final model.

## Final Model

The final model artifacts are stored in the `Models/` folder:

| File | Purpose |
| --- | --- |
| `logistic_model.pkl` | Trained Logistic Regression classifier |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer used to transform ticket text |

Final selected pipeline:

```text
Ticket Description -> TF-IDF Vectorizer -> Logistic Regression -> Predicted Topic Group
```

## Performance Summary

The final model achieved approximately:

| Metric | Score |
| --- | ---: |
| Accuracy | 85.82% |
| Weighted F1 Score | 85.83% |

Class-level observations:

- `Purchase`, `Storage`, and `Access` performed strongly.
- `Administrative rights` had the lowest recall.
- `Hardware` and `Miscellaneous` sometimes overlap because some support requests contain broad or mixed issue descriptions.

## Streamlit App

The project includes a Streamlit interface in:

```text
app.py
```

The app loads the saved TF-IDF vectorizer and Logistic Regression model, accepts a ticket description, and returns the predicted topic group.

Run the app with:

```bash
streamlit run app.py
```

## Required Libraries

The project uses common Python data science and app libraries:

```bash
pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
```

Main libraries used:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `streamlit`
- `matplotlib`
- `seaborn`

## Important Files

| File | Description |
| --- | --- |
| `Observations.txt` | Notes and conclusions from experiments |
| `Notebooks/03_BaselineModel.ipynb` | Baseline TF-IDF + Logistic Regression model |
| `Notebooks/04_ErrorAnalysis.ipynb` | Analysis of actual vs predicted results |
| `Notebooks/05_TextPreprocessing.ipynb` | Text cleaning and preprocessing experiments |
| `Notebooks/06_ModelComparison.ipynb` | Comparison of multiple machine learning models |
| `Notebooks/07_FinalModel.ipynb` | Final model training and artifact saving |
| `Notebooks/baselineResults.csv` | Saved prediction results for error analysis |
| `Models/logistic_model.pkl` | Final trained classifier |
| `Models/tfidf_vectorizer.pkl` | Final fitted vectorizer |
| `app.py` | Streamlit deployment app |

## Limitations

This project was created for personal learning and portfolio purposes.

The model can make incorrect predictions when:

- the ticket is too short
- the ticket contains multiple issues
- the wording is very different from the training data
- the ticket belongs to a minority class
- the issue overlaps between categories

The predictions should be treated as model suggestions, not final business decisions.

## Future Improvements

Possible next steps:

- collect more data for minority classes, especially `Administrative rights`
- tune Logistic Regression hyperparameters
- try class balancing techniques
- compare with transformer-based models such as BERT
- add prediction confidence scores to the Streamlit app
- improve the UI with better sample tickets and category explanations

## Conclusion

This project shows that a simple and interpretable NLP pipeline using TF-IDF and Logistic Regression can perform well for multi-class customer support ticket classification. The final model reaches about 86% accuracy and provides a practical baseline for automatic ticket routing.
