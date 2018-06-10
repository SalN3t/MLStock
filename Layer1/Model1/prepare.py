try:
    from utility import *
    #print '1'
except:
    import os,sys
    sys.path.insert(0,os.path.dirname( os.path.dirname(os.getcwd() ) ) )
    from utility import *
    #print '2'

import re
import string
import pickle
import os 

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

# Setup
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')
# nltk.download('movie_reviews')

# Load config
import json
config = json.load(open('../config.json'))

rx = re.compile('(["#\'\\%`])')
tb = Blobber(pos_tagger=NLTKTagger(), analyzer=NaiveBayesAnalyzer())



def get_text_from_tweets_dataset(data):
    text_list = list()
    for item in data:
        text_list.append(item.split(';')[4])
    return text_list

# rx = re.compile('(["#\'\\%`])')

def extract_stocks(text, stocks_d):
    symbols = list()
    found_dollar_symbol = False
    text = rx.sub(r' ', text)
    for item in text.split(" "):
        if found_dollar_symbol == True: # only stocks that are targeted.. this will help to take out $other_text since they are not stock symbols
            if item.upper() in stocks_d:
                symbols.append("$"+item.upper())
            elif item.replace(".","-").upper() in stocks_d:
                symbols.append("$"+item.replace(".","-").upper())
            found_dollar_symbol = False
        elif '$' in item:   #bug fix with '"' character that sometime be "$ instead of $
            if len(item) > 1:
                if item.replace("$","").upper() in stocks_d:
                    symbols.append(item.upper())
                    found_dollar_symbol = False
                elif item.replace(".","-").upper() in stocks_d:
                    symbols.append("$"+item.replace(".","-").upper())
                    found_dollar_symbol = False
            else:
                found_dollar_symbol = True
    return symbols


def read_tweets_date_from_files_list(files_list):
    all_data = list()
    header = ''
    for item in files_list:
        data = handle_file(item,'r')
        for row in data:
            if header == '' and 'username' in row:
                header = row.replace('\n','')
            elif not 'username' in row:
            # if not 'username' in row: # Remove all the headers
                all_data.append(row.replace('\n',''))
    all_data.insert(0,header)
    return all_data



#################################################

# Text Cleaning
# This to reduce the text to normal pure text
# Texts has three formats
#   1 - pure text: only words and letters
#   2 - text & links: could be picture or normal link
#   3 - text & emojes or characters: Including hash tags
# We will reduce 2 & 3 to 1
# pic.twitter.com/fcZSnh3T7a <- Twitter picture

remove_char_list = [",", ":", "\"", "=", "&", ";", "%", "$","@", "%", "^", "*", "(", ")", "{", "}","[", "]", "|", "/", "\\", ">", "<", "-","!", "?", ".", "'","--", "---", "#"]

def remove_special_chars(tweets):  # it unrolls the hashtags to normal words
	global remove_char_list
        for remove in remove_char_list:
	    tweets = tweets.replace(remove," ")
        return tweets

def remove_by_regex(tweets, regexp):
        return re.sub(regexp, "", tweets, flags=re.MULTILINE)


def remove_urls(tweets):
        text_to_remove = list()
        url_found = False
        second_search = False
        for text in tweets.split(" "):
            if 'http' in text or 'https' in text:
                text_to_remove.append(text)
                url_found = True
            elif url_found:
                text_to_remove.append(text)
                url_found = False
                second_search = True
            elif second_search and ('/' in text or '-' in text):
                text_to_remove.append(text)
                second_search = False
            elif 'pic.twitter.com' in text:
                text_to_remove.append(text)
            else:
                second_search = False
        #print text_to_remove
        for remove in text_to_remove:
            tweets = tweets.replace(remove, ' ')
        return tweets
        # return remove_by_regex(tweets,r"http.?://[^\s]+[\s]?")


def remove_usernames(tweets):
    return remove_by_regex(tweets,r"@[^\s]+[\s]?")


def remove_numbers(tweets):
    return remove_by_regex(tweets,r"\s?[0-9]+\.?[0-9]*")

#remove_special_chars(remove_urls(text_list[18]))

# Remove non ascii characters
# import string
# printable = set(string.printable)
# filter(lambda x: x in printable, s)
def remove_non_ascii_char(line):
    return filter(lambda x: x in set(string.printable), line)


def reduce_text(line):
    return remove_numbers(remove_non_ascii_char(remove_special_chars(remove_usernames(remove_urls(line)))))


def correct_tweets_dataset(tweets_data,stocks_d, header = 'username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink;stocks_mentioned;text_filtered;sentiment'):
    new_dataset = list()
    for item in tweets_data:
        try:
            text = item.split(";")[4]
        except:
            item = item.replace(",",";") # Correcting
            # text = item.split(";")[4]
            # if text.count('"') < 2:
            # index = 5
            # search_range = item.split(";")
            # for i in xrange(len(search_range)):
            #     if search_range[index].count('"') > 0:
            #         break
            #     index = index +1
            # replaced_text = ','.join(search_range[4:index+1]) # +1 to inlcude the last element
            #search_range = item.split(";")
        search_range = item.split(";")
        replaced_text = ', '.join(search_range[4:-5])
        del search_range[4:-5]
        search_range.insert(4,replaced_text)
        item = ";".join(search_range)
        text = item.split(";")[4]
        stocks = "~".join(extract_stocks(text,stocks_d))
        reduced_text = " ".join(filter(None,reduce_text(text).split(" ")))
        sent = tb(reduced_text).sentiment.classification
        new_dataset.append(str(item)+';'+str(stocks)+';"'+reduced_text+'";'+str(sent))
    # Update header
    del new_dataset[0]
    new_dataset.insert(0,header)
    return new_dataset

 # --- Text Preparation functions Start



def read_stocks_date_from_files_list(files_list):
    stock_date_dict = dict()
    for item in files_list:
        key = '$'+str(item.split("/")[-1].split('.')[0]).upper()     # get the stock name and add $ in front of it
        stock_date_dict[key] = pd.read_csv(item)
    return stock_date_dict


# Will convert dataframe to specific dict list format 
# That helps later to add the stocks prices to the target stock and also to build the final dataframe later on
def from_dataframe_to_dict_list(other, stocks_d):
    new_list = list()
    for i,item in tqdm( other['text'].iteritems() ):
        try:
            stocks = [ word.lower() for word in item.split(' ') if word in stocks_d ]        
            for stock in stocks:
                new_list.append({
                    'stock': stock,
                    'username': other['username'][i],
                    'date': other['date'][i],
                    'retweets': other['retweets'][i],
                    'favorites': other['favorites'][i],
                    'text': other['text'][i],
                    'geo': other['geo'][i],
                    'mentions': other['mentions'][i],
                    'hashtags': other['hashtags'][i],
                    'id': other['id'][i],
                    'permalink': other['permalink'][i],
                    'stocks_mentioned': '~'.join(stocks),
                    'text_filtered': other['text_filtered'][i],
                    'sentiment': other['sentiment'][i]
                })
        except Exception as e:
            print("got it : ", e)
            print('error at index: ',str(i) )
            pass
    return new_list


# --- Helpers to get stock price from date index to targeted stock
## -- Start
def helper_find_after_or_before_date_in_stock_date(stock, stock_date_dict, search_index_date, day_period, previous = False):
    found = False # To exit when we find the target date
    begin_date = datetime.datetime.strptime( stock_date_dict[stock].iloc[0]['Date'] ,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(  stock_date_dict[stock].iloc[-1]['Date'] ,'%Y-%m-%d')
    try:
        if previous:
            stock_key_date = search_index_date + datetime.timedelta(days = day_period) # day
        else:
            stock_key_date = search_index_date
        #stock_key_date =  datetime.datetime.strptime(search_index_date.strftime('%Y-%m-%d') , '%Y-%m-%d') 
        # Get dates boundries to make sure no to loop forever
        while found == False and stock_key_date > begin_date and stock_key_date < end_date:
            if not stock_date_dict[stock][ stock_date_dict[stock]['Date'] == stock_key_date.strftime('%Y-%m-%d')].empty:
                found = True
            else:
                stock_key_date = stock_key_date + datetime.timedelta(days = day_period) # day
        if stock_key_date < begin_date or stock_key_date > end_date :
            #stock_key_date = None
            return None, None
        else:
            return stock_key_date.strftime('%Y-%m-%d'), stock_date_dict[stock][ stock_date_dict[stock]['Date'] == stock_key_date.strftime('%Y-%m-%d')]
    except Exception as e:
        #stock_key_date = None
        print 'Issue in stock '+str(stock)
        return None, None

def helper_get_stock_data(stock_date_dict, stock, search_index_date):
    data_dic = dict()
    search_index_date = datetime.datetime.strptime(search_index_date.strftime('%Y-%m-%d') , '%Y-%m-%d') 
    day_period = 1  # one day ahead
    stock_date, stock_date_dataFrame = helper_find_after_or_before_date_in_stock_date(stock, stock_date_dict, search_index_date, day_period)
    # Search Previes
    day_period = -1 # one day back
    stock_previous_date, stock_previous_dataFrame = helper_find_after_or_before_date_in_stock_date(stock, stock_date_dict, search_index_date, day_period, previous = True)
    # Get the other data
    if stock_date == None or stock_previous_date == None:
        print stock
        print search_index_date
        # Current
        data_dic['stock_name'] = stock
        data_dic['stock_date'] = search_index_date.strftime('%Y-%m-%d')
        data_dic['stock_open'] = 'nan'
        data_dic['stock_high'] = 'nan'
        data_dic['stock_low'] = 'nan'
        data_dic['stock_close'] = 'nan'
        data_dic['stock_volume'] = 'nan'
        data_dic['stock_openint'] = 'nan'
        # Previous
        data_dic['stock_previous_date'] = 'nan'
        data_dic['stock_previous_open'] = 'nan'
        data_dic['stock_previous_high'] = 'nan'
        data_dic['stock_previous_low'] = 'nan'
        data_dic['stock_previous_close'] = 'nan'
        data_dic['stock_previous_volume'] = 'nan'
        data_dic['stock_previous_openint'] = 'nan'
    else:
        # Current
        data_dic['stock_name']      = stock
        data_dic['stock_date']      = str( stock_date_dataFrame['Date'].to_string().split(' ')[-1] )   #stock_date_dataFrame['Date'].to_string() will return u'1    1984-09-10'. So we split by space and get the last index
        data_dic['stock_open']      = float( stock_date_dataFrame['Open'] )
        data_dic['stock_high']      = float( stock_date_dataFrame['High'] )
        data_dic['stock_low']       = float( stock_date_dataFrame['Low'] ) 
        data_dic['stock_close']     = float( stock_date_dataFrame['Close'] ) 
        data_dic['stock_volume']    = float( stock_date_dataFrame['Volume'] ) 
        data_dic['stock_openint']   = float( stock_date_dataFrame['OpenInt'] ) 
        # Previous
        data_dic['stock_previous_date']     = str( stock_previous_dataFrame['Date'].to_string().split(' ')[-1] )
        data_dic['stock_previous_open']     = float( stock_previous_dataFrame['Open'] )
        data_dic['stock_previous_high']     = float( stock_previous_dataFrame['High'] )
        data_dic['stock_previous_low']      = float( stock_previous_dataFrame['Low'] )
        data_dic['stock_previous_close']    = float( stock_previous_dataFrame['Close'] )
        data_dic['stock_previous_volume']   = float( stock_previous_dataFrame['Volume'] )
        data_dic['stock_previous_openint']  = float( stock_previous_dataFrame['OpenInt'] ) 
    return data_dic

    # --- Text Preparation functions end
# def write_dataset(new_dataset, filename = "simple_sentence_twitter_dataset_2008_to_2014"):
#     # write the data
#     with open(filename,"w") as f:
#         for item in new_dataset:
#             if not item.split(";")[10] == '':   # Take of rows that stock mentioned is not detected .. those could be cuase of irregular formatting or string encoding
#                 f.write("%s\n"%item.replace("\n",""))


# # data = list()
# # for item in rows:
# # 	oo = ''
# # 	for key in item.keys():
# # 			oo = str(item[key].encode('ascii', 'replace') ).replace(',',';') + ',' + oo
# # 	data.append(oo[:-1])

# def write_file(data, filename):
#     with open(filename, "w") as f:
#             for item in data:
#                 f.write(item.replace('#','').replace("'","")+"\n")

# if __name__ == '__main__':
# Load the configuration data
# Note that we can pass them righ away.. but this is to make the code more readable
stocks_symbols_file = config['stocks_symbols_file']
twitter_dataset_directory = config['twitter_dataset_directory']
output_dataset_filename = config['cleanup']['output_dataset_filename']

# ---------------------- 
#   Tweet Cleaning
# ----------------------

# 1- Read stocks symbols
stocks_d = handle_file(stocks_symbols_file,"r")
stocks_d = [symbol.replace("\n",'') for symbol in stocks_d ]     # Replace \n character

# 2- Read tweets sentiments
tweets_datasets_dir_list = get_all_filenames_from_dir(twitter_dataset_directory,'csv')

# 3- Read all tweets dataset and combin them into one list
all_tweets_data = read_tweets_date_from_files_list(tweets_datasets_dir_list)

# 4- Filter, correct the tweets, extract key words (stock symbols) and find sentiment value to each tweet
all_tweets_data = correct_tweets_dataset(all_tweets_data,stocks_d)

# 5- Write the data to a file
write_dataset(all_tweets_data, filename = output_dataset_filename)

# ---------------------- 
#   Text Preparation
# ----------------------



stocks_symbols_file = config['stocks_symbols_file']
twitter_dataset_directory = config['twitter_dataset_directory']
stocks_dataset_directory = config['stocks_dataset_directory']
stocks_dataset_suffex = config['stocks_dataset_suffex']
start_range_date = config['prepare']['range_filter_fot_stocks']['start_range_date']
end_range_date = config['prepare']['range_filter_fot_stocks']['end_range_date']
cleanup_output_dataset = config['prepare']['cleanup_output_dataset']
output_dataset_filename = config['prepare']['output_dataset_filename']
output_dataset_filename_arff = config['prepare']['output_dataset_filename_arff']
period = config['prepare']['period']


data = pd.read_csv('/home/salah/school/Capston/gitlab_data/MLsTock/datasets/unfinished_prepared_data/simple_sentence_twitter_dataset_2008_to_2014',delimiter=';')

# Remove all repeated tweets by the highest retweets count. i.e some tweets are a retweet of other, where some are just repeated by copy pasting and such. so we need to take those out
reduced_data = remove_repeted_by_highest_value_pandas(data, 'text', 'retweets')

stocks_dir = get_all_filenames_from_dir(stocks_dataset_directory, stocks_dataset_suffex, stocks_d)

# 3- create a dictionary for each stock indexing by the date of the record(Each row as a whole day stock data)
# start_range_date = '2008'
# end_range_date = '2015'

stock_date_dict = read_stocks_date_from_files_list(stocks_dir)

new_list = from_dataframe_to_dict_list(reduced_data, stocks_d)

# ==== 
#appl_s[ appl_s['Date'] == '1984-09-100'].empty
# # stock_date_dict[stock]['Close']

# Update the rows with their stocks price for that targetd stock
for i,item in  enumerate( tqdm(new_list) ):
    new_list[i].update(helper_get_stock_data(stock_date_dict, '$'+str(item['stock']).upper(), datetime.datetime.strptime( item['date'].split(' ')[0], '%Y-%m-%d') ) ) # Update with stock data

# Now we need to take out all the repeated instance
#a,b = unique_instance( [str(item['text'])+str(item['stock']) for item in new_list])
a,b = unique_instance( [','.join([ str(item[key]) for key in item.keys() ] ) for item in new_list])
new_list_2 = remove_repeted(new_list,b)

# Convert it to dataframe
new_pd = pd.DataFrame(new_list_2)

# Write CSV out with ; as delimiter
new_pd.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer1_dataset/Model1/layer_0_dataset_delimiter_semicolon.csv', sep=';')
