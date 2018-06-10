import re
import string
import pickle
import os 

import csv
import glob


# try:
#     import utility
#     print '1'
# except:
#     import os,sys
#     sys.path.insert(0,os.path.dirname(os.getcwd() ) )
#     import utility
#     print '2'

def handle_file(filename,operation):
    with open(filename,operation) as f:
        data = f.readlines()
    return data

def get_all_filenames_from_dir(directory,suffex, filename_allowed_list = None):
    files_list = list()
    if filename_allowed_list == None:
        for item in glob.glob(directory+'*'+suffex): # Example /datasets/Stock_dataset/Stocks/*.txt
            files_list.append(item) 
    else:
        filename_allowed_list = [v.lower() for v in filename_allowed_list] # To avoid case sensitve
        for item in glob.glob(directory+'*'+suffex):
            if item.split("/")[-1].split('.')[0].lower() in filename_allowed_list: # Since linux is case sensitive, then so is this function, make sure the names match correctly
                files_list.append(item)
        if not len(files_list) == len(filename_allowed_list):
            print 'Some Stocks files are missing'
    return files_list


def write_dataset(new_dataset, filename = "simple_sentence_twitter_dataset_2008_to_2014"):
    # write the data
    with open(filename,"w") as f:
        for item in new_dataset:
            if not item.split(";")[10] == '':   # Take of rows that stock mentioned is not detected .. those could be cuase of irregular formatting or string encoding
                f.write("%s\n"%item.replace("\n",""))

def write_file(data, filename):
    with open(filename, "w") as f:
            for item in data:
                f.write(str(item).replace('#','').replace("'","")+"\n")

# --------------------------------
def unique_instance(un_data):
    test_dict = dict()
    indexed = list()
    count = 0
    for i,item in enumerate(un_data):
        if not test_dict.has_key( hash(item) ):
            test_dict[ hash(item) ] = 0
        else:
            count = count + 1
            indexed.append(i)
    return count, indexed

def remove_repeted(data,num_list):
    tmp_data = list(data)
    for row_index in sorted( num_list,reverse=True):
        del tmp_data[row_index]
    return tmp_data
# --------------------------------
def remove_repeted_by_highest_value(data, hieghst_value_index,target_value_index, remove_header = True):
    data_clone = list(data)
    if remove_header:
        data_clone = data_clone[1:]
    output_data = dict()
    for item in data_clone:
        item = item.replace('\n','').split(';')
        hash_value = hash( item[target_value_index] )
        if output_data.has_key(hash_value): 
            if int( output_data[hash_value]['max']) < int( item[hieghst_value_index] ):
                output_data[hash_value]['max'] = int( item[hieghst_value_index] )
                output_data[hash_value]['item'] = item
        else:
            output_data[hash_value] = dict()
            output_data[hash_value]['max'] = int( item[hieghst_value_index] )
            output_data[hash_value]['item'] = item
    return output_data
    



def remove_repeted_by_highest_value_pandas(df_data, target_index, compare_col):
    hash_tmp = {}
    for i,item in df_data[target_index].iteritems():
        if hash_tmp.has_key(str(item)):
                if hash_tmp[str(item)]['count'] > int(df_data[compare_col][i]):
                        hash_tmp[str(item)]['count'] = int(df_data[compare_col][i])
                        hash_tmp[str(item)]['index'] = i
        else:
                hash_tmp[str(item)] = {}
                hash_tmp[str(item)]['count'] = int(df_data[compare_col][i])
                hash_tmp[str(item)]['index'] = i
    return df_data.iloc[ [ hash_tmp[key]['index'] for key in hash_tmp.keys()] ]



# ------------------------------------
# Text cleaning functions


remove_char_list = ["~",",", ":", "\"", "=", "&", ";", "%", "$","@", "%", "^", "*", "(", ")", "{", "}","[", "]", "|", "/", "\\", ">", "<", "-","!", "?", ".", "'","--", "---", "#"]

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


def find_stocks(words_list, stocks_d):
    stocks_list = list()
    for word in words_list:
        if word.lower() in stocks_d:
            if not word.lower() in stocks_list: # so we won't add it again if the user mention the same stock again
                stocks_list.append(word.lower())
    return stocks_list

def remove_stocks_name_from_words_list(words_list, stocks_d):
    return filter(lambda a: not a.lower() in find_stocks(words_list, stocks_d),words_list)

def find_stocks_v2(words_list, stocks_fullname_dict, full_name_stocks_list):
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
    full_name_stocks_list = list()
    for key in stocks_fullname_dict.keys():
        for name in stocks_fullname_dict[key]:
            full_name_stocks_list.append(name)
    return find_stocks_v2(words_list, stocks_fullname_dict, full_name_stocks_list)

def reduce_text(line):
    return remove_numbers(remove_non_ascii_char(remove_special_chars(remove_usernames(remove_urls(line)))))

def reduce_text_with_websites_data_kept(line):
    return remove_numbers(remove_non_ascii_char(remove_special_chars(remove_usernames(line))))

def reduce_w_remove_stock(line, stocks_fullname_dict):
    full_name_stocks_list = list()
    for key in stocks_fullname_dict.keys():
        for name in stocks_fullname_dict[key]:
            full_name_stocks_list.append(name)
    line = reduce_text(line)
    return ' '.join( find_stocks_v2( filter(None, line.split(' ') ), stocks_fullname_dict, full_name_stocks_list) )

