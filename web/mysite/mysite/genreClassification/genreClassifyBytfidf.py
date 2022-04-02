from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import base64
import io
class genreClassifyBytfidf:

    def __init__(self,data_path):
        nltk.download('stopwords')
        self.lr = LogisticRegression()
        self.clf = OneVsRestClassifier(self.lr)
        self.dataset = pd.read_csv(data_path)
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)
        self.newdataset = {}

    # function for text cleaning
    def clean_text(self,text):
        # remove backslash-apostrophe
        text = re.sub("\'", "", text)
        # remove everything alphabets
        text = re.sub("[^a-zA-Z]"," ",text)
        # remove whitespaces
        text = ' '.join(text.split())
        # convert text to lowercase
        text = text.lower()
        
        return text


    # function to remove stopwords
    def remove_stopwords(self,text):
        
        stop_words = set(stopwords.words('english'))
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def pre_processing(self):
        clean_dataset = self.dataset.filter(['Name', 'Genre', 'Description'], axis = 1)
        #dataset.drop(labels=[.'id', 'MovieId', 'Year', 'Time', 'Rate', 'Votes', 'Poster'], axis=1,inplace=True)
        self.newdataset = clean_dataset.head(10000)
        test_genres = []
        for i in self.newdataset['Genre']:
            if i not in test_genres:
                i=str(i).replace(" ", "")
                test_genres.append(list(i.split(",")))
        self.newdataset['genre_new'] = test_genres


        # get all genre tags in a list
        all_genres = sum(test_genres,[])

        all_genres = nltk.FreqDist(all_genres)

        self.newdataset['clean_Description'] = self.newdataset['Description'].apply(lambda x: self.clean_text(x))
        self.newdataset['clean_Description'] = self.newdataset['clean_Description'].apply(lambda x: self.remove_stopwords(x))


    def model(self):
        self.multilabel_binarizer.fit(self.newdataset['genre_new'])

        # transform target variable
        y = self.multilabel_binarizer.transform(self.newdataset['genre_new'])

        xtrain, xval, ytrain, yval = train_test_split(self.newdataset['clean_Description'], y, test_size=0.1, random_state=9)
        
        # create TF-IDF features
        xtrain_tfidf = self.tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = self.tfidf_vectorizer.transform(xval)
        
        # fit model on train data
        self.clf.fit(xtrain_tfidf, ytrain)


        # make predictions for validation set
        y_pred = self.clf.predict(xval_tfidf)
        # predict probabilities
        y_pred_prob = self.clf.predict_proba(xval_tfidf)
        t = 0.35 # threshold value
        y_pred_new = (y_pred_prob >= t).astype(int)
        # evaluate performance
        f1 = f1_score(yval, y_pred_new, average="micro")

        return xval,f1


    def infer_tags(self,q):
        # q = self.clean_text(q)
        # q = self.remove_stopwords(q)
        # q_vec = self.tfidf_vectorizer.transform([q])
        # q_pred = self.clf.predict(q_vec)
        # return self.multilabel_binarizer.inverse_transform(q_pred)

        q = self.clean_text(q) 
        q = q.lower() 
        q = self.remove_stopwords(q) 
        q_vec = self.tfidf_vectorizer.transform([q]) 
        q_pred = self.clf.predict(q_vec) 
        q_pred =  self.multilabel_binarizer.inverse_transform(q_pred)[0]

        return str(q_pred)[1:-1]
      

    def print_genre_summary(self):
        all_genres = sum(self.newdataset['genre_new'],[])
        all_genres = nltk.FreqDist(all_genres)
        all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 'Count': list(all_genres.values())})

        g = all_genres_df.nlargest(columns="Count", n = 100) 
        plt.figure(figsize=(12,12))
        ax = sns.barplot(data=g, x= "Count", y = "Genre")
        ax.set(ylabel = 'Genre')
        imgdata =  io.BytesIO()
        plt.savefig(imgdata)
        # imgdata.seek(0)

        #data = imgdata.getvalue()
        b64 = base64.b64encode(imgdata.getvalue()).decode()
        return b64
      
    def print_freq_words(self, terms = 30):
        all_words = ' '.join([text for text in self.newdataset['clean_Description']])
        all_words = all_words.split()
    
        fdist = nltk.FreqDist(all_words)
        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    
        # selecting top 20 most frequent words
        d = words_df.nlargest(columns="count", n = terms) 
        plt.figure(figsize=(12,15))
        ax = sns.barplot(data=d, x= "count", y = "word")
        ax.set(ylabel = 'Word')
        # plt.show()
        imgdata =  io.BytesIO()
        plt.savefig(imgdata)
        b64 = base64.b64encode(imgdata.getvalue()).decode()
        return b64

  
    def predict(self,rangeVal):
        xval,f1 = self.model()
        resultArr = []
        for i in range(rangeVal):
            subitem = {}
           
            k = xval.sample(1).index[0]
            subitem["Movie"] = self.newdataset['Name'][k]
            subitem["Predicted_genre"] = self.infer_tags(xval[k])
            subitem["Actual_genre"] = self.newdataset['genre_new'][k]
            resultArr.append(subitem)
        return resultArr,f1