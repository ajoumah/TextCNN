TextCNN for Sentiment Analysis on IMDB Movie Reviews
This project implements a multi-channel Convolutional Neural Network (CNN), also known as TextCNN, for binary sentiment classification of movie reviews from the IMDB dataset. The model is built using Keras and TensorFlow backend, and is capable of distinguishing between positive and negative reviews.

📁 Project Structure
bash
Copy
Edit
TextCNN/
├── dataset_preprocessing.py     # Load and clean raw text data
├── model_training.py            # Build and train multi-channel CNN
├── model_evaluation.py          # Evaluate and test model
├── TextCNN.ipynb                # All-in-one Jupyter Notebook version
├── model.h5                     # Saved trained model
├── train.pkl / test.pkl         # Serialized training and testing datasets
└── README.md                    # Project documentation
🧠 Model Architecture
This TextCNN implementation includes three parallel convolutional layers with kernel sizes 4, 6, and 8 to capture semantic features at multiple n-gram levels. Each convolutional stream is followed by a dropout and max-pooling layer, then merged and passed through fully connected layers.

rust
Copy
Edit
Input -> Embedding -> Conv1D (ks=4) -> Dropout -> MaxPooling -> Flatten
                   -> Conv1D (ks=6) -> Dropout -> MaxPooling -> Flatten
                   -> Conv1D (ks=8) -> Dropout -> MaxPooling -> Flatten
         -> Concatenate -> Dense -> Output (Sigmoid)
🚀 Getting Started
1. Requirements
Install the required Python packages:

bash
Copy
Edit
pip install nltk keras tensorflow numpy
Also download NLTK stopwords:

python
Copy
Edit
import nltk
nltk.download('stopwords')
2. Dataset
The model uses the IMDB Sentiment Dataset (txt_sentoken) which contains two folders:

pos/ — positive reviews

neg/ — negative reviews

You can find the dataset here.

Place the dataset in your Google Drive:

swift
Copy
Edit
/content/drive/MyDrive/MovieDataset/txt_sentoken/
3. Data Preprocessing
Run the preprocessing pipeline to:

Clean text: lowercasing, removing punctuation, stopwords, etc.

Serialize data to .pkl files for training/testing.

bash
Copy
Edit
python dataset_preprocessing.py
4. Training the Model
To train the TextCNN model:

bash
Copy
Edit
python model_training.py
Batch Size: 16

Epochs: 10

Loss: Binary Crossentropy

Optimizer: Adam

Model will be saved to:

swift
Copy
Edit
/content/drive/MyDrive/MovieDataset/txt_sentoken/model.h5
5. Evaluate the Model
To evaluate and test the model:

bash
Copy
Edit
python model_evaluation.py
Outputs:

Accuracy on training and test sets

Sentiment predictions on new reviews

📊 Results
Dataset	Accuracy
Training	~97%
Testing	~85-90%

📌 Sample Prediction
python
Copy
Edit
sample = "The movie had great performances and a compelling story."
predict = model.predict([X, X, X])  # where X is the encoded and padded sample
print(predict)  # Returns sentiment probability
🧪 Future Enhancements
Add BiLSTM or Transformer-based comparisons

Incorporate GloVe or Word2Vec pre-trained embeddings

Hyperparameter tuning (batch size, kernel size, embedding dims)

Use of attention mechanism for interpretability

📄 License
This project is open-sourced for educational and non-commercial use. Please cite or acknowledge appropriately if used in research or presentations.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📬 Contact
Developed by Ahmad Joumah
For questions or collaboration: ahmadjoumah.dev@gmail.com
