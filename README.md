📄 TextCNN: Multi-Channel CNN for Text Classification
A deep learning implementation of a multi-channel Convolutional Neural Network (CNN) for text classification using Keras and TensorFlow. This project covers data preprocessing, tokenization, model definition, training, saving, and evaluation.

🚀 Features
📥 Data Loading & Preprocessing

Load text files from directories

Clean text by removing punctuation, stopwords, and short tokens

Tokenize and pad sequences for model input

🧠 Multi-Channel CNN Architecture

Three parallel convolutional channels with different kernel sizes (4, 6, 8)

Embedding layers with 100-dimensional vectors

Dropout and max pooling for regularization and downsampling

Merged layers with dense layers for binary classification

📊 Training & Evaluation

Fit the model on training data with label arrays

Evaluate on training and test datasets

Save and load trained models for inference

🔍 Prediction on Sample Texts

Tokenizer and padding applied on sample sentences

Model outputs predicted sentiment scores

⚙️ Requirements
Python 3.x

TensorFlow / Keras

NLTK (with stopwords downloaded)

NumPy

Matplotlib (for plot_model visualization)

Pickle

🧩 Usage Overview
1. Data Preparation
Load and clean raw text documents from folders

Remove punctuation and stopwords

Create training and test datasets

Save processed datasets as pickle files

2. Tokenization & Encoding
Fit a tokenizer on the training data

Encode and pad text sequences to fixed length

3. Model Definition
Define a multi-channel CNN with Keras Functional API

Compile with binary cross-entropy loss and Adam optimizer

Visualize the model architecture to multichannel.png

4. Model Training
Train for 10 epochs with batch size 16

Save the trained model to disk

5. Evaluation & Prediction
Load the trained model

Evaluate on training and test sets

Predict sentiment for custom sample sentences

📁 Project Structure
graphql
Copy
Edit
.
├── data/
│   ├── neg/              # Negative sentiment text files
│   ├── pos/              # Positive sentiment text files
│   └── train.pkl         # Preprocessed training dataset
│   └── test.pkl          # Preprocessed testing dataset
├── model/
│   └── model.h5          # Trained CNN model file
├── multichannel.png      # CNN architecture visualization
└── TextCNN.ipynb         # Jupyter notebook with code
⚠️ Notes
Modify directory paths in the script according to your local setup.

Ensure NLTK stopwords are downloaded with:

python
Copy
Edit
import nltk
nltk.download('stopwords')
Input sequences are padded post to max document length.

Binary labels are used (0 = negative, 1 = positive).

🛠️ How to Run
Prepare your text dataset folders (positive/negative).

Run preprocessing functions to generate cleaned pickle datasets.

Define and train the model using the prepared datasets.

Evaluate and test on new samples.

📈 Model Summary Example
markdown
Copy
Edit
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 1380)]       0
 embedding (Embedding)          (None, 1380, 100)    1405700
 conv1d (Conv1D)                (None, 1377, 32)     12832
 dropout (Dropout)             (None, 1377, 32)     0
 max_pooling1d (MaxPooling1D)  (None, 688, 32)      0
 flatten (Flatten)              (None, 22016)        0
 ...
==================================================================================================
Total params: 162,202
Trainable params: 162,202
Non-trainable params: 0
__________________________________________________________________________________________________
🙌 Contributions
Feel free to fork, modify, and improve this project!

📫 Contact
For questions or suggestions, please reach out.
