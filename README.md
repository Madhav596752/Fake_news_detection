# Fake News Detection Using BERT and Machine Learning

A beginner-friendly deep learning project that classifies news articles as **FAKE** or **REAL**. It compares a traditional machine learning baseline (**TF-IDF + Logistic Regression**) with a fine-tuned **BERT** (`bert-base-uncased`) model.

Everything runs **locally inside one Google Colab notebook**. No OpenAI, Gemini, Groq, GPT or any other API is used, and no API keys are needed.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Files](#3-project-files)
4. [Dataset](#4-dataset)
5. [How to Run (Step by Step)](#5-how-to-run-step-by-step)
6. [Configuration Settings](#6-configuration-settings)
7. [Notebook Walkthrough](#7-notebook-walkthrough)
8. [Preprocessing and Data Leakage](#8-preprocessing-and-data-leakage)
9. [Models Explained](#9-models-explained)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Testing With Your Own News](#11-testing-with-your-own-news)
12. [Saving and Reloading the Model](#12-saving-and-reloading-the-model)
13. [Expected Runtime](#13-expected-runtime)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [Glossary](#17-glossary)

---

## 1. Project Overview

**Problem:** Fake news spreads quickly and can mislead people. Manual fact-checking cannot keep up with the amount of content published daily.

**Goal:** Build a binary text classifier that reads a news article (title + text) and predicts:

| Label | Class |
|---|---|
| 0 | FAKE |
| 1 | REAL |

**Approach:**
1. Baseline: TF-IDF converts text to numbers, and Logistic Regression classifies it.
2. Deep learning: BERT is fine-tuned on the same data with PyTorch and Hugging Face Transformers.
3. Both models are evaluated on the same unseen test set and compared on Accuracy, Precision, Recall, F1-score and training time.

**Important:** This project does not report fixed accuracy numbers. All results are produced by running the notebook, and they depend on the sample size, GPU/CPU and random seed.

---

## 2. Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| Platform | Google Colab (GPU recommended) |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Classical ML | Scikit-learn (TfidfVectorizer, LogisticRegression, metrics) |
| Deep learning | PyTorch |
| NLP model | Hugging Face Transformers, `bert-base-uncased` |

No external APIs and no extra frameworks.

---

## 3. Project Files

```
Fake_News_Detection_BERT.ipynb   <- the complete project (single notebook)
README.md                        <- this file
Fake.csv                         <- dataset file 1 (you download from Kaggle)
True.csv                         <- dataset file 2 (you download from Kaggle)
saved_bert_fake_news/            <- created by the notebook after training
```

There are no separate Python files. All code lives in the notebook.

---

## 4. Dataset

**Name:** Fake and Real News Dataset (Kaggle)
**Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

| File | Contents | Label used |
|---|---|---|
| `Fake.csv` | Fake news articles | 0 (FAKE) |
| `True.csv` | Real news articles | 1 (REAL) |

**Columns in each file:** `title`, `text`, `subject`, `date`

The notebook combines both files into one DataFrame and adds a `label` column.

**Download steps:**
1. Sign in to Kaggle (free account).
2. Open the dataset link and click **Download**.
3. Unzip the file to get `Fake.csv` and `True.csv`.

Keep the file names exactly `Fake.csv` and `True.csv` (capital F and T). Colab file names are case-sensitive.

---

## 5. How to Run (Step by Step)

1. Open [Google Colab](https://colab.research.google.com) and choose **File > Upload notebook**. Select `Fake_News_Detection_BERT.ipynb`.
2. Enable the GPU: **Runtime > Change runtime type > T4 GPU > Save**.
3. Run the cells from top to bottom (**Runtime > Run all** also works).
4. When the upload cell in Section 3 asks for files, click **Choose Files** and select both `Fake.csv` and `True.csv`.
   Alternative: open the **Files** panel (folder icon on the left) and upload the two files there before running.
5. Wait for training to finish. Results, graphs and the comparison table appear in the output cells.
6. Test your own news in Section 8 by editing the `my_news` variable.

Uploaded files are deleted when the Colab runtime resets, so you may need to upload them again in a new session.

---

## 6. Configuration Settings

Found in the settings cell of Section 3:

| Setting | Default | Meaning |
|---|---|---|
| `SMALL_DATASET` | `True` | `True` = balanced sample for fast runs. `False` = full dataset |
| `SAMPLES_PER_CLASS` | `2500` | Articles per class when the small dataset is on (5,000 in total) |
| `MAX_LEN` | `128` | Maximum BERT tokens per article |
| `EPOCHS` | `2` | BERT training epochs (2 to 3 recommended) |
| `BATCH_SIZE` | `16` | Articles per training batch |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |

**CPU fallback:** If no GPU is detected, the notebook automatically sets `SMALL_DATASET = True` and `SAMPLES_PER_CLASS = 400` (800 articles in total) so it finishes in a reasonable time.

**Tip:** If you run out of GPU memory, reduce `BATCH_SIZE` to 8.

---

## 7. Notebook Walkthrough

| Section | What it does |
|---|---|
| 1. Introduction | Defines fake news detection, its importance, the objectives and the workflow |
| 2. Import Libraries | Imports all libraries, sets random seeds, and detects the GPU |
| 3. Dataset Loading | Uploads and loads the CSVs, assigns labels (FAKE = 0, REAL = 1), combines them, shows the first 5 rows |
| 4. EDA | Shape, missing values, duplicates, class distribution, article length plots, and the `subject` leakage check |
| 5. Preprocessing | Removes missing values and duplicates, merges title and text, cleans text, encodes labels, optionally samples, and splits 80:20 |
| 6. Baseline Model | TF-IDF + Logistic Regression with metrics, a classification report, a confusion matrix and top words |
| 7. BERT Model | Tokenizer, custom PyTorch Dataset, DataLoaders, `BertForSequenceClassification`, AdamW training loop, loss plot, test evaluation and confusion matrix |
| 8. Custom Prediction | `predict_news()` returns FAKE or REAL with a confidence score, and runs on 4 sample inputs |
| 9. Model Comparison | Comparison table (Accuracy, Precision, Recall, F1, Training time) and a bar chart |
| 10. Conclusion | Achievements, why BERT differs from traditional ML, limitations and future scope |
| Save Model | Saves the BERT model and tokenizer, optionally zips and downloads them, and reloads them to verify |
| Presentation Prep | Architecture, working explanation, BERT in simple words, comparison table, why BERT, 5 viva Q&A and a 2-minute script |

---

## 8. Preprocessing and Data Leakage

**Data leakage** means the model gets access to information that makes the task unrealistically easy. This dataset has known shortcuts, so the notebook handles them:

| Problem | Why it is a problem | What the notebook does |
|---|---|---|
| `subject` column | Fake and real articles have completely different subject values | Column is dropped |
| `date` column | Not a news-content feature | Column is dropped |
| "CITY (Reuters) -" tag | Real articles almost always start with it | Removed with a regular expression |
| Duplicate articles | The same article could land in both train and test | Duplicates removed before splitting |
| TF-IDF fitted on all data | Test data would influence the vocabulary | `fit_transform` on train only, `transform` on test |
| Overlap check | Safety net | An `assert` verifies no article is in both train and test |

**Cleaning steps in `clean_text()`:**
1. Convert to lowercase (BERT uncased).
2. Remove the "(Reuters)" source tag.
3. Remove URLs and Twitter picture links.
4. Remove HTML tags.
5. Keep only letters, digits and basic punctuation.
6. Remove extra spaces.

The same function is used for training data and for custom predictions.

Even after these steps, some style differences between sources may remain, so very high scores on this dataset should be interpreted carefully.

---

## 9. Models Explained

### 9.1 Baseline: TF-IDF + Logistic Regression
- **TF-IDF** gives each word a weight. A word that is frequent in one article but rare across all articles gets a high score.
- **Logistic Regression** learns a weight per word and outputs the probability that an article is REAL.
- Treats text as a "bag of words", so word order and context are ignored.
- Very fast, runs on the CPU, and is easy to interpret (the notebook prints the top words for each class).

### 9.2 Deep Learning: BERT
**BERT** (Bidirectional Encoder Representations from Transformers) is a language model pre-trained on a very large amount of text.

**Pipeline in the notebook:**
1. `BertTokenizer` converts text to token IDs and adds `[CLS]` (start) and `[SEP]` (end).
2. Texts are truncated or padded to **128 tokens**.
3. A custom `NewsDataset` class wraps the token IDs and labels as PyTorch tensors.
4. `DataLoader` feeds the data in batches.
5. `BertForSequenceClassification` (2 labels) adds a classification layer on top of BERT.
6. Training loop with the **AdamW** optimizer: forward pass, loss, backward pass, gradient clipping, weight update.
7. Average training loss is printed and plotted per epoch.
8. The model is evaluated on the test set.

**Fine-tuning** means we start from the pre-trained model and train it a bit more on our fake-news task, so it learns the task quickly.

When loading the model, Hugging Face prints a warning about "newly initialized weights". This is normal: it refers to the new classification layer that is trained in this project.

### 9.3 Baseline vs BERT

| Point | TF-IDF + Logistic Regression | BERT |
|---|---|---|
| Type | Traditional ML | Deep learning (Transformer) |
| Text representation | Word weights | Contextual embeddings |
| Word order and context | Ignored | Understood |
| Pre-trained knowledge | None | Yes |
| Parameters | Thousands | About 110 million |
| Training cost | Seconds, CPU | Longer, GPU recommended |
| Interpretability | Easy | Hard |

---

## 10. Evaluation Metrics

REAL (1) is treated as the positive class.

| Metric | Question it answers |
|---|---|
| **Accuracy** | What percentage of all predictions are correct? |
| **Precision** | Of the articles predicted REAL, how many are really REAL? |
| **Recall** | Of all real articles, how many did the model find? |
| **F1-score** | Balance between precision and recall |
| **Confusion matrix** | How many articles of each class were classified correctly or wrongly? |

Reading the confusion matrix:

```
                 Predicted FAKE   Predicted REAL
True FAKE        correct          real news mistakes: fake marked REAL
True REAL        real marked FAKE correct
```

**Training time note:** Logistic Regression runs on the CPU and BERT on the GPU (if available), so the times are not a perfectly fair comparison. They still show the practical cost of each approach.

---

## 11. Testing With Your Own News

Edit and run this in Section 8:

```python
my_news = "Paste any headline or article here."
predict_news(my_news)
```

Output: the predicted class (FAKE or REAL), confidence, and the probability for each class.

**Test with known labels:**
- Use unseen test articles:
  ```python
  for i in range(5):
      print("TRUE LABEL:", "REAL" if y_test[i] == 1 else "FAKE")
      predict_news(X_test[i])
  ```
- Real news: recent articles from trusted agencies (Reuters, AP, BBC).
- Fake news: known debunked stories from fact-checkers (Snopes, PolitiFact, AltNews) or clearly sensational claims you write yourself.

**Understand what the output means:**
- The model learns writing patterns from US political news (about 2016 to 2018). It does not verify facts.
- Confidence shows how sure the model is, not whether it is correct.
- Short headlines, other topics, other countries and recent events may give poor results.

---

## 12. Saving and Reloading the Model

The notebook saves the model and tokenizer to `saved_bert_fake_news/` and creates `saved_bert_fake_news.zip` for download.

To reuse the model later:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("saved_bert_fake_news")
model = BertForSequenceClassification.from_pretrained("saved_bert_fake_news")
model.eval()
```

Remember to use the same `clean_text()` function and `max_length=128` before predicting.

---

## 13. Expected Runtime

Times depend on the Colab hardware and availability, so these are rough guides, not guarantees.

| Setup | Data used | Approximate time |
|---|---|---|
| T4 GPU, small dataset (default) | 5,000 articles | A few minutes |
| T4 GPU, full dataset | About 38,000 articles | Much longer (tens of minutes) |
| CPU only | 800 articles (auto) | Several minutes |

The notebook prints your actual training times, and they appear in the comparison table.

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Fake.csv` | Upload both CSV files; check the capitalization of the names |
| Training is very slow | Enable the GPU: Runtime > Change runtime type > T4 GPU |
| `CUDA out of memory` | Lower `BATCH_SIZE` to 8, then restart the runtime and run again |
| Colab disconnects mid-training | Use the small dataset, keep the tab active, and re-run from the top |
| Warning about "newly initialized weights" | Normal, no action needed |
| `NameError` for a variable | Cells were run out of order. Use Runtime > Restart and run all |
| Files disappeared | Colab runtime was reset. Upload the CSVs again |
| Tokenization takes 1 to 2 minutes | Expected: the slower `BertTokenizer` is used on thousands of articles |

---

## 15. Limitations

- Mostly US political news from about 2016 to 2018, so the model may not generalize to other topics, languages or recent events.
- It detects writing style and patterns, not factual truth.
- BERT reads only the first 128 tokens of each article.
- The small-dataset option and only 2 to 3 epochs trade accuracy and stability for speed.
- A single train/test split is used (no cross-validation), and no hyperparameter tuning is done.
- Some source-style differences may remain, which can make this dataset easier than real-world fake news detection.

---

## 16. Future Scope

- Train on the full dataset for more epochs, with a learning-rate scheduler and a validation set.
- Try RoBERTa, DistilBERT or DeBERTa.
- Use longer inputs (512 tokens or chunking).
- Evaluate on other datasets (LIAR, FakeNewsNet) to test generalization.
- Add explainability (LIME, SHAP or attention visualization).
- Combine text with source credibility, author and social-media signals.
- Build a small web app (Streamlit or Gradio) around the saved model.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Token** | A small piece of text (word or sub-word) that BERT reads |
| **Tokenizer** | Converts text into token IDs |
| **TF-IDF** | Word-weighting method based on how special a word is in a document |
| **Transformer** | Neural network architecture built on self-attention |
| **Self-attention** | Mechanism that lets each word look at all other words in the sentence |
| **Pre-training** | Initial training of BERT on a huge text collection |
| **Fine-tuning** | Further training of a pre-trained model on your own task |
| **Epoch** | One full pass over the training data |
| **Batch** | A small group of samples processed together |
| **Loss** | Number showing how wrong the model is; training tries to reduce it |
| **AdamW** | Optimizer that updates model weights |
| **Logits / Softmax** | Raw model scores / conversion of scores into probabilities |
| **Data leakage** | When information that should be hidden influences the model, making scores unrealistic |
| **Baseline** | A simple model used as a reference point |
