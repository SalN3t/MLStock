import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

data = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method2/Method2_base_dataset.csv')
# Keeping only the neccessary columns
data = data[['reduced_text','stock_status']]

# Make sure we don't have ## separator in text 
data['reduced_text'].replace(' ## ','',inplace=True)

data['reduced_text'] = data['reduced_text'].apply(lambda x: str(x).lower())
data['reduced_text'] = data['reduced_text'].apply((lambda x: re.sub(r'[^a-zA-z0-9\s]','',str(x))))

print(data[ data['stock_status'] == 'RAISE'].size)
print(data[ data['stock_status'] == 'FALL'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['reduced_text'].values)

X = tokenizer.texts_to_sequences(data['reduced_text'].values)

X = pad_sequences(X)


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['stock_status']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

# Predicting the whole test set
ypred = model.predict(X_test)
print("F1-Measure",f1_score(Y_test, ypred.round(), average='weighted')  )

from sklearn.metrics import roc_auc_score

print("ROC AUC score: {:.3f} ".format( roc_auc_score(Y_test, ypred)) )

from sklearn.metrics import accuracy_score

print("accuracy score: {:.2f}% ".format(acc *100.0)  )
#accuracy_score(Y_test, ypred.round())

from sklearn.metrics import cohen_kappa_score
print("Kappa Value: {:.4f}".format( cohen_kappa_score(Y_test.argmax(axis=1), ypred.argmax(axis=1), labels=None, weights=None) ) )

print("pos_acc", float(pos_correct)/float(pos_cnt)*100.0, "%")
print("neg_acc", float(neg_correct)/float(neg_cnt)*100.0, "%")

# Save the model to file
model.save('LSTM_Model_built.h5')

# # To load the model later on 
# from keras.models import load_model
# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')