import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv(r"C:\Users\rohit\Desktop\machine learning\Datasets\NLM.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
x = df["text"]
y = df["label"]
wd =  WordNetLemmatizer()

def lower_case(x):
    for i in range(len(x)):
        x[i] = str(x[i]).lower()
def number_remove(x):
    for i in range(len(x)):
        x[i] = re.sub(r"\d+",'',x[i])
def punctuation(x):
    punct = str.maketrans('','',string.punctuation)
    for i in range(len(x)):
        x[i] = x[i].replace(","," ")
        x[i] = x[i].translate(punct)
def white_space(x):
    for i in range(len(x)):
        x[i] = " ".join(x[i].split())
def token(x):
    for i in range(len(x)):
        x[i] = word_tokenize(x[i])
def stopword(x):
    stop_words = set(stopwords.words('english'))
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            if x[i][j] not in stop_words:
                list1.append(x[i][j])
        x[i] = list1
def pos_create(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            list2 = []
            list2.append(x[i][j])
            list1.append(nltk.pos_tag(list2))
        x[i] = list1
            
def pos_place(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j][1].startswith('J'):
                x[i][j][1] = wordnet.ADJ
            elif x[i][j][1].startswith('V'):
                x[i][j][1] = wordnet.VERB
            elif x[i][j][1].startswith('N'):
                x[i][j][1] = wordnet.NOUN
            elif x[i][j][1].startswith('R'):
                x[i][j][1] = wordnet.ADV
            else:         
                x[i][j][1] =  None
def convert(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            l = str(x[i][j]).split()
            punctuation(l)
            list1.append(list(l))
        x[i] = list1
def lem(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            if (x[i][j][1] == None):
                list1.append(x[i][j][0])
            else:
                list1.append(wd.lemmatize(x[i][j][0],x[i][j][1]))
        x[i] = " ".join(list1)
lower_case(x)
number_remove(x)
punctuation(x)
white_space(x)
token(x)
stopword(x)
pos_create(x)
convert(x)
pos_place(x)
lem(x)
"""LOGISTIC REGRESSION"""
cv = CountVectorizer()
X = cv.fit_transform(x)
lr = LogisticRegression(C=24,max_iter=150)
lr.fit(X,y)
"""NAIVE BAYES"""
mnb = MultinomialNB()
mnb.fit(X,y)


            
    


