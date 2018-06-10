
# coding: utf-8

# # Predicting the Dow Jones with News


import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, Merge,BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers


dj = pd.read_csv("/home/salah/school/Capston/NEWS_base_models/dataset/news/DJIA_table.csv")


news = pd.read_csv("/home/salah/school/Capston/NEWS_base_models/dataset/news/RedditNews.csv")


# Inspect the data

dj.head()

dj.isnull().sum()

news.isnull().sum()

news.head()


print(dj.shape)
print(news.shape)

# Compare the number of unique dates. We want matching values.
print(len(set(dj.Date)))
print(len(set(news.Date)))

# Remove the extra dates that are in news
news = news[news.Date.isin(dj.Date)]

print(len(set(dj.Date)))
print(len(set(news.Date)))


# Calculate the difference in opening prices between the following and current day.
# The model will try to predict how much the Open value will change beased on the news.

dj = dj.set_index('Date').diff(periods=1)
dj['Date'] = dj.index
dj = dj.reset_index(drop=True)

# Remove unneeded features
dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1)

dj.head()

# Remove top row since it has a null value.
dj = dj[dj.Open.notnull()]

# Check if there are any more null values.
dj.isnull().sum()

# Create a list of the opening prices and their corresponding daily headlines from the news
price = []
headlines = []

for row in dj.iterrows():
    daily_headlines = []
    date = row[1]['Date']
    price.append(row[1]['Open'])
    for row_ in news[news.Date==date].iterrows():
        daily_headlines.append(row_[1]['News'])
    # Track progress
    headlines.append(daily_headlines)
    if len(price) % 500 == 0:
        print(len(price))

price = []
headlines = []

count = 0
for row in dj.iterrows():
    if count == 0:
        count = 100
        date = row[1]['Date']
        price.append(row[1]['Open'])
    else:
        daily_headlines = []
        date = row[1]['Date']
        price.append(row[1]['Open'])
        for row_ in news[news.Date==date].iterrows():
            daily_headlines.append(row_[1]['News'])
        # Track progress
        headlines.append(daily_headlines)
        if len(price) % 500 == 0:
            print(len(price))


# Find Fall, Raise Status
count = 0
price_status = []
for i,item in enumerate(price):
    if count == 0:
        count = 100
    else:
        if price[i -1] > price[i]:
            price_status.append('FALL')
        elif price[ i-1] < price[i]:
            price_status.append('RAISE')
        else:
            price_status.append('SAME')


# Compare lengths to ensure they are the same
del price[0] # Remove the first row since we only use that to find wither the stock raise or fall.... As it is the first date we have in our dataset
print(len(price))
print(len(price_status))
print(len(headlines))


# Compare the number of headlines for each day
print(max(len(i) for i in headlines))
print(min(len(i) for i in headlines))
#print(np.mean(len(i) for i in headlines))


# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    # Convert words to lower case
    text = text.lower()
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text


# Clean the headlines
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)


# Take a look at some headlines to ensure everything was cleaned well
# clean_headlines[0]

all_data = []
for i,item in enumerate( clean_headlines):
    news_tmp = ' ## '.join(clean_headlines[i])
    price_tmp = price[i]
    status_tmp = price_status[i]
    all_data.append({ 'headline':news_tmp, 'price':price_tmp, 'status':status_tmp} )



final_pd = pd.DataFrame(all_data)


final_pd.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer1_dataset/Model2/Layer1_base_dataset.csv', index=False)


################################################# End of preparation