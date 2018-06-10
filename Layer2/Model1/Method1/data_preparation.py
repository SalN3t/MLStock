try:
    from utility import *
except:
    import os,sys
    sys.path.insert(0,os.path.dirname( os.path.dirname( os.path.dirname(os.getcwd() ) ) ) )
    from utility import *

import re
import string
import pickle
import os 

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import csv
import glob
from textblob import TextBlob
import nltk
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.taggers import NLTKTagger

from nltk.corpus import sentiwordnet as swn

# Load config
import json
config = json.load(open('../../../config.json'))

rx = re.compile('(["#\'\\%`])')
tb = Blobber(pos_tagger=NLTKTagger(), analyzer=NaiveBayesAnalyzer())

## Functions
def sent_from_text(text):
    test_b = tb( text )
    pos_count = 0.0
    neg_count = 0.0
    pos_sum = 0.0
    neg_sum = 0.0
    web_mention = 0.0
    for word in test_b.words:
        tmp_pos_score = 0.0
        tmp_neg_score = 0.0
        for item in swn.senti_synsets(word):
            tmp_pos_score = tmp_pos_score + item.pos_score()
            tmp_neg_score = tmp_neg_score +  item.neg_score()
            pos_sum = pos_sum + item.pos_score()
            neg_sum = neg_sum + item.neg_score()
        #print 'pos: '+str(tmp_pos_score)+' neg: '+str(   tmp_neg_score )
        if 'http' in word:
            web_mention = web_mention + 1
        if str(tmp_pos_score) > str(tmp_neg_score):
            pos_count = pos_count + 1.0
        elif str(tmp_pos_score) < str(tmp_neg_score):
            neg_count = neg_count + 1.0
    return {'pos_count':int(round(pos_count)), 'neg_count': int(round(neg_count)) , 'pos_sum': int(round(pos_sum)) , 'neg_sum': int(round(neg_sum)), 'web_mention': int(round(web_mention)) }

def stocks_name_dict(stocks_fullname):
    output_dict = dict()
    for item in stocks_fullname:
        item = item.split(',')
        key = item[0].strip()
        del item[0]
        output_dict[key] = [ v.strip() for v in item]
    return output_dict

def find_stocks_v2(words_list, stocks_fullname_dict):
    stocks_list = list()
    for word in words_list:
        if not stocks_fullname_dict.has_key( str( word.replace('$','') ).lower() ):
            stocks_list.append( word.lower() )
    text = ' '.join( stocks_list )
    for name in full_name_stocks_list:
        #if name in text:
        text = text.lower().replace(name.lower(), ' ')
    # for key in stocks_fullname_dict.keys():
    #     for name in stocks_fullname_dict[key]:
    #         text = text.lower().replace(name.lower(), ' ')
    stocks_list = filter(None, text.split(' '))
    return stocks_list

def remove_stocks_name_from_words_list_v2(words_list, stocks_fullname_dict):
    return find_stocks_v2(words_list, stocks_fullname_dict)
    #return filter(lambda a: not a.lower() in find_stocks_v2(words_list, stocks_fullname_dict),words_list)

## Process
stocks_symbols_file = config['stocks_symbols_file']
stocks_fullname_file = config['stocks_fullname_file']
twitter_dataset_directory = config['twitter_dataset_directory']
stocks_dataset_directory = config['stocks_dataset_directory']
stocks_dataset_suffex = config['stocks_dataset_suffex']
start_range_date = config['prepare']['range_filter_fot_stocks']['start_range_date']
end_range_date = config['prepare']['range_filter_fot_stocks']['end_range_date']
cleanup_output_dataset = config['prepare']['cleanup_output_dataset']
output_dataset_filename = config['prepare']['output_dataset_filename']
output_dataset_filename_arff = config['prepare']['output_dataset_filename_arff']
period = config['prepare']['period']

#np.where(new_pd['stock_close'] > new_pd['stock_previous_close'],'RAISE', np.where(new_pd['stock_close'] < new_pd['stock_previous_close'], 'FALL','SAME') )

new_pd = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer1_dataset/Model1/layer_0_dataset_delimiter_semicolon.csv', delimiter=';')

new_pd['stock_status'] = np.where(new_pd['stock_close'] > new_pd['stock_previous_close'],'RAISE', np.where(new_pd['stock_close'] < new_pd['stock_previous_close'], 'FALL','SAME') )

stocks_d = handle_file(stocks_symbols_file,'r')
stocks_d = [symbol.replace("\n",'').lower() for symbol in stocks_d ]

stocks_fullname = handle_file(stocks_fullname_file,'r')
stocks_fullname = [symbol.replace("\n",'').lower() for symbol in stocks_fullname ]
stocks_fullname_dict = stocks_name_dict(stocks_fullname)

full_name_stocks_list = list()

for key in stocks_fullname_dict.keys():
    for name in stocks_fullname_dict[key]:
        full_name_stocks_list.append(name)


# stocks_dir = get_all_filenames_from_dir(stocks_dataset_directory, stocks_dataset_suffex, stocks_d)

# # 3- create a dictionary for each stock indexing by the date of the record(Each row as a whole day stock data)
# # start_range_date = '2008'
# # end_range_date = '2015'
# stock_date_dict = read_stocks_date_from_files_list(stocks_dir, start_range_date, end_range_date)

# V2
# final_dict = dict()
# final_list = list()
# for stock in stocks_d:
#     st_data = text_with_stocks_data_for_a_stock(stock, stock_date_dict, o_data)
#     print 'Stock: '+str(stock) +', len: '+str( len(st_data) )
#     final_list.append( reduce_the_dataset_v2(st_data, stock) )
#     print datetime.datetime.strftime(datetime.datetime.now(), '%H:%M')
#     #break

pos_sum_list = list()
neg_sum_list = list()

pos_count_list = list()
neg_count_list = list()

web_mention = list()

reduced_text_list = list()
for item in tqdm(new_pd['text']):
    # reduced_text = reduce_text(item)
    reduced_text = reduce_w_remove_stock(item, stocks_fullname_dict)
    with_websites_kept_reduced_text = reduce_text_with_websites_data_kept(item)
    sent_dict = sent_from_text(with_websites_kept_reduced_text) # Get the sentiments
    reduced_text_list.append(reduced_text)
    pos_sum_list.append(sent_dict['pos_sum'])
    neg_sum_list.append(sent_dict['neg_sum'])
    pos_count_list.append(sent_dict['pos_count'])
    neg_count_list.append(sent_dict['neg_count'])
    web_mention.append(sent_dict['web_mention'])

# Build the new dataFrame columns
new_pd['reduced_text'] = reduced_text_list
new_pd['pos_sum'] = pos_sum_list
new_pd['neg_sum'] = neg_sum_list
new_pd['pos_count'] = pos_count_list
new_pd['neg_count'] = neg_count_list
new_pd['web_mention'] = web_mention

#new_pd_2 = new_pd[ new_pd['stock_close'] != 'nan' ] # Remove the rows that we are not usefull and has nan for its features
new_pd = new_pd[ new_pd['stock_open'].notnull() ] # Remove the rows that we are not usefull and has nan for its features

# See NaN values by column, if the column is not important then we don't need to drop the full row
new_pd.isnull().sum()

# Drop NaNs rows for reduced text since this is an important feature for this method
#new_pd_3 = new_pd_2[ new_pd_2['reduced_text'].notnull() ] 

# sort data by stock name to group them together
new_pd_3 = new_pd.sort_values(by=['stock'])

# Combine all tweets that are tweeted in the same day and add it's values
# We surly can assume that the stock price status is the same since it's the same day
# We will be dropping some features which are:
#   text(because we only need reduced_text),geo, hashtags, mentions, permalink, sentiment(as we only want the newly ones), text_filtered
combined_dict = dict()
for i,item in tqdm( new_pd_3['stock_date'].iteritems() ):
    key = (new_pd_3['stock_date'][i], new_pd_3['stock'][i])
    if combined_dict.has_key( key ):
        combined_dict[key] = {
            'date': new_pd_3['date'][i],    # This should be the same one
            'stock': new_pd_3['stock'][i],  # This should be the same one
            'stock_close': new_pd_3['stock_close'][i],  # This should be the same one
            'stock_date': new_pd_3['stock_date'][i],  # This should be the same one
            'stock_high': new_pd_3['stock_high'][i],  # This should be the same one
            'stock_low': new_pd_3['stock_low'][i],  # This should be the same one
            'stock_name': new_pd_3['stock_name'][i],  # This should be the same one
            'stock_open': new_pd_3['stock_open'][i],  # This should be the same one
            'stock_openint': new_pd_3['stock_openint'][i],  # This should be the same one
            'stock_previous_close': new_pd_3['stock_previous_close'][i],  # This should be the same one
            'stock_previous_date': new_pd_3['stock_previous_date'][i],  # This should be the same one
            'stock_previous_high': new_pd_3['stock_previous_high'][i],  # This should be the same one
            'stock_previous_low': new_pd_3['stock_previous_low'][i],  # This should be the same one
            'stock_previous_open': new_pd_3['stock_previous_open'][i],  # This should be the same one
            'stock_previous_openint': new_pd_3['stock_previous_openint'][i],  # This should be the same one
            'stock_previous_volume': new_pd_3['stock_previous_volume'][i],  # This should be the same one
            'stock_volume': new_pd_3['stock_volume'][i],  # This should be the same one
            'stock_status': new_pd_3['stock_status'][i],  # This should be the same one
            'favorites': combined_dict[key]['favorites'] + int( new_pd_3['favorites'][i] ),
            'id': combined_dict[key]['id'] +' ## '+ str( new_pd_3['id'][i] ),   # to keep track of how many users tweeted
            'username': combined_dict[key]['username'] +' ## '+ str( new_pd_3['username'][i] ),   # to keep track of how many users tweeted
            'retweets': combined_dict[key]['retweets'] + int( new_pd_3['retweets'][i] ),
            'reduced_text': combined_dict[key]['reduced_text'] +' ## '+ str( new_pd_3['reduced_text'][i] ),
            'pos_sum': combined_dict[key]['pos_sum'] + int( new_pd_3['pos_sum'][i] ),
            'neg_sum': combined_dict[key]['neg_sum'] + int( new_pd_3['neg_sum'][i] ),
            'pos_count': combined_dict[key]['pos_count'] + int( new_pd_3['pos_count'][i] ),
            'neg_count': combined_dict[key]['neg_count'] + int( new_pd_3['neg_count'][i] ),
            'web_mention': combined_dict[key]['web_mention'] + int( new_pd_3['web_mention'][i] )
        }
    else:
        combined_dict[key] = {
            'date': new_pd_3['date'][i],    # This should be the same one
            'stock': new_pd_3['stock'][i],  # This should be the same one
            'stock_close': new_pd_3['stock_close'][i],  # This should be the same one
            'stock_date': new_pd_3['stock_date'][i],  # This should be the same one
            'stock_high': new_pd_3['stock_high'][i],  # This should be the same one
            'stock_low': new_pd_3['stock_low'][i],  # This should be the same one
            'stock_name': new_pd_3['stock_name'][i],  # This should be the same one
            'stock_open': new_pd_3['stock_open'][i],  # This should be the same one
            'stock_openint': new_pd_3['stock_openint'][i],  # This should be the same one
            'stock_previous_close': new_pd_3['stock_previous_close'][i],  # This should be the same one
            'stock_previous_date': new_pd_3['stock_previous_date'][i],  # This should be the same one
            'stock_previous_high': new_pd_3['stock_previous_high'][i],  # This should be the same one
            'stock_previous_low': new_pd_3['stock_previous_low'][i],  # This should be the same one
            'stock_previous_open': new_pd_3['stock_previous_open'][i],  # This should be the same one
            'stock_previous_openint': new_pd_3['stock_previous_openint'][i],  # This should be the same one
            'stock_previous_volume': new_pd_3['stock_previous_volume'][i],  # This should be the same one
            'stock_volume': new_pd_3['stock_volume'][i],  # This should be the same one
            'stock_status': new_pd_3['stock_status'][i],  # This should be the same one
            'favorites': int( new_pd_3['favorites'][i] ),
            'id': str( new_pd_3['id'][i] ),   # to keep track of how many users tweeted
            'username': str( new_pd_3['username'][i] ),   # to keep track of how many users tweeted
            'retweets': int( new_pd_3['retweets'][i] ),
            'reduced_text': str( new_pd_3['reduced_text'][i] ),
            'pos_sum': int( new_pd_3['pos_sum'][i] ),
            'neg_sum': int( new_pd_3['neg_sum'][i] ),
            'pos_count': int( new_pd_3['pos_count'][i] ),
            'neg_count': int( new_pd_3['neg_count'][i] ),
            'web_mention': int( new_pd_3['web_mention'][i] )
        }


final_list = list()
for key in combined_dict.keys():
    final_list.append(combined_dict[key])


# Convert it to dataframe
final_pd = pd.DataFrame(final_list)

# Remove the stocks that had SAME status
final_pd = final_pd[final_pd['stock_status'] != 'SAME']

# print data size
print('Data Size: ',len(final_pd))

# Remove unwated characters from reduced text (optional)
final_pd['reduced_text'].replace(' ## ','',inplace=True)


# Create sentiment status for the whole reduced text combined
final_pd['sentiment_count_status'] = np.where( final_pd['pos_count'] > final_pd['neg_count'],'pos','neg'  )

final_pd['sentiment_sum_status'] = np.where( final_pd['pos_sum'] > final_pd['neg_sum'],'pos','neg'  )

# reduced_text = list()
# for item in final_pd['reduced_text']:
#     reduced_text.append( reduce_w_remove_stock(item, stocks_fullname_dict) )

# Chose the features we want to use
final_pd_2 = final_pd[['reduced_text','pos_sum','neg_sum','pos_count','neg_count','web_mention','stock_name','stock_close','stock_previous_close', 'sentiment_count_status', 'sentiment_sum_status', 'stock_status']]

# final_pd_2['reduced_text'] = reduced_text

# sort data by stock name to group them together
final_pd_2 = final_pd_2.sort_values(by=['stock_name'])

# Chose the first 10500 this is just to reduce the size  of the dataset for smaller memory machines to handle
# final_pd_3 = final_pd_2.iloc[:10500]
# final_pd_3.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method1/Method1_dataset.csv')

final_pd_2.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method1/Method1_base_dataset.csv',index=False)
# =================
# pp_f['stock_date'][2]
# '2009-08-10'
# >>> pp_f['stock_name'][2]
# '$MSFT'
# >>> pp_f['stock'][2]
# 'msft'
# 'AMZN Ex Div Date Also See DIA EPI PBR UTX ~'


# import matplotlib.pyplot as plt
# import plotly.plotly as py

# dictionary = plt.figure()

# D = {u'Label0':26, u'Label1': 17, u'Label2':30}

# plt.bar(range(len(stocks_dict)), stocks_dict.values(), align='center')
# plt.yticks(range(len(stocks_dict)), stocks_dict.keys())

# plot_url = py.plot_mpl(dictionary, filename='mpl-dictionary')

####################################
# ARFF - Attributes reduced        #
####################################
# import arff
# data = arff.load(open('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model2/Method1/Layer2_method1_base_dataset.arff','rb'))
# data.keys()
# data['attributes'][0] = (u'text','STRING')
# with open('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model2/Method1/Layer2_method1_base_dataset.arff','w') as f :
#     f.write( arff.dumps(data) )
####################################
