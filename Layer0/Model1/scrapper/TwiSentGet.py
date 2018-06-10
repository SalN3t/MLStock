import random,time
import os.path
import datetime

# load scrapper
from twitter_scrapper import Exporter
# Load config
import json
config = json.load(open('../../config.json'))



def handle_file(filename,operation):
    with open(filename,operation) as f:
        data = f.readlines()
    return data

def chunck_stocks_list_into_allowed_twitter_search_query(stocks_symbols_file):
	# create query chuncks with or where there length is less than 256 char
	# 256 -2 = 254 taking out '(' and ')'
	str_list = dict()
	count = 254
	index = 0
	for item in stocks_symbols_file:
		item = item.replace("\n","")
		if count == 254:
			str_list[index] = '%24'+str(item) + ' OR '
			count = count - len(item) - 5 # 4 for ' OR ' and 1 for $
		elif count < 10:
			str_list[index] = str_list[index] + '%24'+str(item) 
			count = 254
			index = index +1
		else:
			str_list[index] = str_list[index] + '%24'+str(item) + ' OR '
			count = count - len(item) - 5 # 4 for ' OR '
	# Now that we have a correct string length we can start creating our queries
	# We have ' OR ' between stocks which is a twitter keyword to match A OR B
	queries = list()
	for key in str_list.keys():
		queries.append( ' OR '.join(filter(None,str_list.get(key).split(' OR ')))  )
	return queries

def get_next_datetime_period(current_datetime):
	"""
	This is a helper to find the next datetime period
	The order to check is:
		- Day
		- Month 
		- Year
	i.e if day is true then it will choose it even if year is true as well
	
	Arguments:
		current_datetime {datetime} --  The current datetime that we want to find the next period to
	
	Returns:
		datetime -- The next datetime period
	"""
	period_range = int(config['data_collect']['period']['period_range'])
	if config['data_collect']['period']['period_by_day']:
		return current_datetime + datetime.timedelta(days= period_range) # increment the date
	elif config['data_collect']['period']['period_by_month']:
		weeks_to_months = 4 * period_range	# Doing that since timedelta does not support month increments by itself.
		return current_datetime + datetime.timedelta(weeks= weeks_to_months) # increment the date
	else: # Increment by year
		return datetime.datetime(current_datetime.year + 1, current_datetime.month, current_datetime.day) # preserve the month and the day and increment only the year


def do_search(query_list):
	# Setting up the variables
	start_date 				= datetime.datetime.strptime(config['data_collect']['range']['start_date'], "%Y-%m-%d")
	end_date 				= datetime.datetime.strptime(config['data_collect']['range']['end_date'], "%Y-%m-%d")
	output_filename 		= str(config['data_collect']['output_datasets']['output_directory']) + str(config['data_collect']['output_datasets']['filename_slug'])
	output_file_time_format = str(config['data_collect']['output_datasets']['filename_datetime_format'])
	max_tweets 				= int(config['data_collect']['maxtweets_per_query'])
	# Throttles are the random number range between each request.. This help prevent any load to the twitter servers and avoid blacklisting the IP
	# The bigger the range the slower it runs but the more chance that your IP won't get blacklisted and viceversa
	day_throttle_from 		= int(config['data_collect']['scrapper_millisecond_throttle']['day']['from'])
	day_throttle_to 		= int(config['data_collect']['scrapper_millisecond_throttle']['day']['to'])
	year_throttle_from 		= int(config['data_collect']['scrapper_millisecond_throttle']['year']['from'])
	year_throttle_to 		= int(config['data_collect']['scrapper_millisecond_throttle']['year']['to'])
	allow_overrwrite 		= config['data_collect']['allow_dataset_overwrite']
	# Starting the search
	count = 0
	for item in query_list:
		count = count + 1 # This to sotre with the filename to know which patch been searched for each file
		start_date = datetime.datetime.strptime(config['data_collect']['range']['start_date'], "%Y-%m-%d")

		while start_date <= end_date:
			args = ["--querysearch", item, "--since", 0, "--until", 0, "--maxtweets",max_tweets , "--output", " "]

			args[3] = start_date.strftime('%Y-%m-%d') 	# --since value

			since_datetime = start_date.strftime(output_file_time_format) # Helper for output filename

			start_date = get_next_datetime_period(start_date)

			args[5] = start_date.strftime('%Y-%m-%d')	# --until value

			until_datetime = start_date.strftime(output_file_time_format) # Helper for output filename

			args[9] = output_filename+"_"+since_datetime+"_"+until_datetime+"_"+str(count)+".csv"
			# print args
			if not allow_overrwrite: # This is recommended since if the connection got cut of then we don't start from the begining and can skip what have we already got
				if not os.path.exists(args[9]):
					Exporter.main(args)
					time_to_sleep = random.randint(day_throttle_from, day_throttle_to)
					print("Day = Sleeping for "+str(time_to_sleep))
					time.sleep(time_to_sleep)
			else:
				Exporter.main(args)
				time_to_sleep = random.randint(day_throttle_from, day_throttle_to)
				print("Day (with allow_overrwrite set) = Sleeping for "+str(time_to_sleep))
				time.sleep(time_to_sleep)

		time_to_sleep = random.randint(year_throttle_from, year_throttle_to)
		print("Year = Sleeping for "+str(time_to_sleep))
		time.sleep(time_to_sleep)		

	return True
# def increment_search_by_month():

# def increment_search_by_year():
if __name__ == '__main__':
	# Read the data
	stocks_symbols_file = handle_file(config['data_collect']['stocks_symbols_file'],'r')

	query_list = chunck_stocks_list_into_allowed_twitter_search_query(stocks_symbols_file)
	do_search(query_list)
	# month_dict = {'0':1,'1':2,'2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'8':9,'9':10,'10':11,'11':12}
	# day_dict = {'0':'01','1':'05','2':'10','3':'15','4':'20','5':'25'}
	# year = 2008
	# month = lambda x: month_dict[str(x%12)]
	# day = lambda x: day_dict[str(x%6)]

	# start_date = datetime.datetime.strptime(config['data_collect']['range']['start_date'], "%Y-%m-%d")
	# end_date = datetime.datetime.strptime(config['data_collect']['range']['end_date'], "%Y-%m-%d")

	# count = 0
	# for item in query_list:
	# 	count = count + 1 # This to sotre with the filename to know which patch been searched for each file
	# 	year = 2008
	# 	for i in xrange(5):
	# 		year = year +1
	# 		args = ["--querysearch", item, "--since", 0, "--until", 0, "--maxtweets", config['data_collect']['maxtweets_per_query'], "--output", " "]
	# 		for j in xrange(12):
	# 			for k in xrange(6):
	# 				args[3] = str(year) +"-"+str(month(j))+'-'+day(k)
	# 				if k == 5:
	# 					if j == 11:
	# 						args[5] = str(year+1) +"-"+str(month(j+1))+'-'+day(k+1)
	# 					else:
	# 						args[5] = str(year) +"-"+str(month(j+1))+'-'+day(k+1)
	# 				else:
	# 					args[5] = str(year) +"-"+str(month(j))+'-'+day(k+1)
	# 				args[9] = "TwiSent_"+str(args[3])+"_"+str(args[5])+"_"+str(count)+".csv"
	# 				if not os.path.exists(args[9]):
	# 					Exporter.main(args)
	# 					time_to_sleep = random.randint(20,100)
	# 					print("Day = Sleeping for "+str(time_to_sleep))
	# 					time.sleep(time_to_sleep)
	# 		time_to_sleep = random.randint(60,100)
	# 		print("Year = Sleeping for "+str(time_to_sleep))
	# 		time.sleep(time_to_sleep)
# Month later
# zzz = datetime.date(2017,3,29)
# >>> zzz + datetime.timedelta(weeks=4)
# datetime.date(2017, 4, 26)

# =================================================================

	# # Read the data
	# stocks_symbols_file = handle_file(config['data_collect']['stocks_symbols_file'],'r')

	# # with open('/home/salah/school/Capston/datasets/Stock_dataset/stocks_symbols.txt','r') as f:
	# # 	st_data = f.readlines()

	# query_list = chunck_stocks_list_into_allowed_twitter_search_query(stocks_symbols_file)
	
	# month_dict = {'0':1,'1':2,'2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'8':9,'9':10,'10':11,'11':12}
	# day_dict = {'0':'01','1':'05','2':'10','3':'15','4':'20','5':'25'}
	# year = 2008
	# month = lambda x: month_dict[str(x%12)]
	# day = lambda x: day_dict[str(x%6)]
	# count = 0
	# for item in query_list:
	# 	count = count + 1 # This to sotre with the filename to know which patch been searched for each file
	# 	year = 2008
	# 	for i in xrange(5):
	# 		year = year +1
	# 		args = ["--querysearch", item, "--since", 0, "--until", 0, "--maxtweets", 100, "--output", "twiSent-"]
	# 		args = list()
	# 		args.append("--querysearch")
	# 		args.append(item)
	# 		args.append("--since")
	# 		args.append(0)
	# 		args.append("--until")
	# 		args.append(0)
	# 		args.append("--maxtweets")
	# 		args.append(100)
	# 		args.append("--output")
	# 		args.append("twiSent-")
	# 		for j in xrange(12):
	# 			for k in xrange(6):
	# 				args[3] = str(year) +"-"+str(month(j))+'-'+day(k)
	# 				if k == 5:
	# 					if j == 11:
	# 						args[5] = str(year+1) +"-"+str(month(j+1))+'-'+day(k+1)
	# 					else:
	# 						args[5] = str(year) +"-"+str(month(j+1))+'-'+day(k+1)
	# 				else:
	# 					args[5] = str(year) +"-"+str(month(j))+'-'+day(k+1)
	# 				args[9] = "TwiSent_"+str(args[3])+"_"+str(args[5])+"_"+str(count)+".csv"
	# 				if not os.path.exists(args[9]):
	# 					Exporter.main(args)
	# 					time_to_sleep = random.randint(20,100)
	# 					print("Day = Sleeping for "+str(time_to_sleep))
	# 					time.sleep(time_to_sleep)
	# 			# time_to_sleep = random.randint(5,30)
	# 			# print("Month = Sleeping for "+str(time_to_sleep))
	# 			# time.sleep(time_to_sleep)
	# 				#print args
	# 			# sys.exit(1)
	# 			# args[3] = str(year) +"-"+str(month(j))+"-01"
	# 			# if j == 11:
	# 			# 	args[5] = str(year+1) +"-"+str(month(j+1))+"-01"
	# 			# else:
	# 			# 	args[5] = str(year) +"-"+str(month(j+1))+"-01"
	# 			# args[9] = "TwiSent_"+str(args[3])+"_"+str(args[5])+".csv"
	# 			# main(args)
	# 			# time_to_sleep = random.randint(5,60)
	# 			# print("Month = Sleeping for "+str(time_to_sleep))
	# 			# time.sleep(time_to_sleep)
	# 			# #print args
	# 		time_to_sleep = random.randint(60,100)
	# 		print("Year = Sleeping for "+str(time_to_sleep))
	# 		time.sleep(time_to_sleep)
	# # for j in xrange(12)

	# # 	args.append(year+"-"+month(j))
	# # 	args.append(queries[0])
	# # 	args.append(queries[0])
	# # 	main(sys.argv[1:])