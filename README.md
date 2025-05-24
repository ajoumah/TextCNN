# 📝 TextCNN: Multi-Channel CNN for Text Classification

---

## 🚀 Overview

This project implements a **TextCNN** model for binary sentiment classification of text documents. It processes raw text data, cleans and tokenizes it, and trains a multi-channel CNN model to classify documents as positive or negative.

---

## 🛠️ Features

* 📂 Load and preprocess text documents (cleaning, tokenization, stopword removal)
* 📊 Multi-channel 1D CNN architecture with different kernel sizes (4, 6, 8)
* 🔄 Train/test dataset handling with data serialization using pickle
* ⚙️ Custom tokenizer creation and text encoding with padding
* 🧠 Model training, saving, loading, and evaluation
* 🔮 Sample prediction on new text input

---

## 📦 Requirements

* Python 3.x
* [nltk](https://www.nltk.org/) (for stopwords)
* [Keras](https://keras.io/) with TensorFlow backend
* numpy
* matplotlib (optional, for model visualization)
* pickle (Python standard library)

---

## ⚙️ Usage

### 1. Prepare Data

* Place your positive and negative text files in separate directories.
* Update the directory paths in the script accordingly.

### 2. Preprocessing & Dataset Creation

```python
# Load and clean documents, then save processed datasets
negative_docs = process_docs('path/to/neg', is_train=True)
positive_docs = process_docs('path/to/pos', is_train=True)
trainX = negative_docs + positive_docs
trainy = [0]*len(negative_docs) + [1]*len(positive_docs)
save_dataset([trainX, trainy], 'train.pkl')
```
### RequirementsModel Architecture

Three parallel 1D CNN channels with different kernel sizes (4, 6, 8)

Each channel has embedding, convolution, dropout, max-pooling, and flatten layers

Outputs merged and passed through dense layers for binary classification

📝 ### Notes
Make sure NLTK stopwords are downloaded:

python
Copy
Edit
import nltk
nltk.download('stopwords')
Adjust file paths based on your environment

The plot_model function saves a visualization of the model architecture as multichannel.png



