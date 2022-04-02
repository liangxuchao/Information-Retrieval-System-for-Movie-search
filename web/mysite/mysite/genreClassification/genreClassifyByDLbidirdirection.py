"""
preprocess_data.py.

This script creates a data preprocessing class.
"""

# Imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score


from pathlib import Path

class PreprocessData:
    """Class for data preprocessing."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.tokenizer = Tokenizer()
        self.stopwords = stopwords.words("english")

    def preprocess_train(self, train_df):
        """
        Filter the synopses of the training set.

        This function removes stopwords from the synopses and
        turns the movie genres into multi-label vectors
        """
        # Remove rows with missing values in either 'genres' or 'synopsis'
        train_df.dropna(axis=0, subset=["Description", "Genre"], inplace=True)
        
        # Load data and split it per columns
        train_df = train_df.sample(frac=1, random_state=37)  # Shuffle data
        synopses = train_df["Description"].apply(lambda x: x.lower())
        genres = train_df["Genre"].apply(lambda x: x.replace(" ","")).apply(lambda x: x.split(","))


        # Remove stopwords from the synopses
        processed_synopses = synopses.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in self.stopwords]
            )
        )

        # Turn the genres into multi-label vectors
        self.mlb.fit(genres.tolist())
        y_train = self.mlb.transform(genres.tolist())

        # Tokenize synopses
        self.tokenizer.fit_on_texts(processed_synopses)
        self.max_length = 200 # Max length of sequences

        x_train = self.tokenizer.texts_to_sequences(processed_synopses)
        pad_x_train = pad_sequences(x_train, maxlen=200, padding="post", truncating="post")

        return pad_x_train, y_train

    def preprocess_test_data(self, test_df):
        """
        Filter the synopses of the test set.

        This function removes stopwords from the synopses.
        """
        # Remove rows with missing values in 'synopsis'
        test_df.dropna(axis=0, subset=["Description"], inplace=True)
        
        # Load data and split it per columns
        self.movie_name = test_df["Name"]
        self.genre = test_df["Genre"]
        synopses = test_df["Description"].apply(lambda x: x.lower())

        # Remove stopwords from the synopses
        proc_synopses = synopses.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in self.stopwords]
            )
        )

        # Tokenize synopses
        x_test = self.tokenizer.texts_to_sequences(proc_synopses)
        padded_x_test = pad_sequences(x_test, maxlen=200, padding="post", truncating="post")

        return padded_x_test

class genreClassifyByDLbidirdirection:
    """Class for training model and make predictions."""

    def __init__(self):
        self.data = PreprocessData()

    def train_model(self, train_df):
        """
        Trains the RNN model using the training set.

        The GloVe word embeddings used were downloaded from
        https://nlp.stanford.edu/projects/glove/.
        """
        # Get preprocessed training set
        padded_x_train, y_train = self.data.preprocess_train(train_df)
        path = str(Path(__file__).resolve().parent.parent) + r"\genreClassification\glove.6B.100d.txt"
        # Using pre-trained GloVe embeddings for embedding layer
        f = open(path, encoding="utf8")

        # Creating dictionary with the words as keys
        # and vector representations as values
        embeddings_index = {}
        for row in f:
            arr = row.split()
            word = arr[0]
            coefs = np.asarray(arr[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()

        # Creating embedding matrix
        word_index = self.data.tokenizer.word_index
        vocab_size = len(word_index) + 1
        emb_dim = 100

        embedding_matrix = np.zeros((vocab_size, emb_dim))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Set random seed
        tf.random.set_seed(37)
        x_train, x_test, y_train, y_test = train_test_split(padded_x_train, y_train, test_size=0.2, random_state=42)

        # Set parameters
        max_length = self.data.max_length
        gru_num_cells = 128
        dense_num_cells = 128
        num_outputs = y_train.shape[1]

        # Creating model
        model = Sequential(
            [
                Embedding(
                    vocab_size,
                    emb_dim,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False
                ),
                Bidirectional(GRU(gru_num_cells, return_sequences=True)),
                Dropout(0.1),
                GlobalMaxPooling1D(),
                Dropout(0.1),
                Dense(dense_num_cells, activation="relu"),
                Dense(num_outputs, activation="sigmoid")
            ]
        )

        # Compiling model
        optimizer = Adam(learning_rate=1e-2)

        
        print(model.summary())
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=[Precision(0.3, 5)]
        )

        # Fit model to the data
        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=512,
            validation_split=0.2
        )
        
        modelpath = str(Path(__file__).resolve().parent.parent) + r"\genreClassification\movieGenre.h5"
        model.save(modelpath)

        
        out = model.predict(x_test)
        out = np.array(out)
        y_pred = np.zeros(out.shape)

        y_pred[out>0.5]=1
        y_pred = np.array(y_pred)

        hl = hamming_loss(y_test,y_pred)
        score = accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test,y_pred, average = 'macro')
        recall = metrics.recall_score(y_test,y_pred, average = 'macro')
        f1 = metrics.f1_score(y_test,y_pred, average = 'macro')

        return hl, score, precision,recall,f1
        # print("Hamming loss:", hl)
        # print("score:", score)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1_score:", f1)

    def predict(self, test_df):
        """Make predictions on the test set."""
        # Get preprocessed test set
        padded_x_test = self.data.preprocess_test_data(test_df)
        
        modelpath = str(Path(__file__).resolve().parent.parent) + r"\genreClassification\movieGenre.h5"
        model = load_model(modelpath)
        # Make predictions
        y_preds = model.predict(padded_x_test)
        print(y_preds)
        # Get top 5 predicted genres for each movie
        outputs = []
        classes = np.asarray(self.data.mlb.classes_)
        
        for pred in y_preds:
            sorted_indices = np.argsort(pred)[::-1]
            output = classes[sorted_indices][:3]
            output_text = " ".join(output)
            outputs.append(output_text)

        outputs = np.asarray(outputs)

        # Create DataFrame with movie ids and top 5 predicted genres
        predictions_df = pd.DataFrame(
            {"Movie": self.data.movie_name, "Actual_genre":self.data.genre ,"Predicted_genre": outputs}
        )

        predictions_df.to_csv(str(Path(__file__).resolve().parent.parent) + r"\genreClassification\bdresult.csv", index=False)

        return predictions_df.to_numpy()