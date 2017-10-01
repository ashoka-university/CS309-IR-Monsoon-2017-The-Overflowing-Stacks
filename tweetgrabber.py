import time 
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome()
burl = u'https://twitter.com/search?'
query = 'narendra modi hot pics'
processedquery = urllib.parse.urlencode({'q':query})

browser.get(burl+processedquery)
time.sleep(1)

body = browser.find_element_by_tag_name('body')

for _ in range(5):
	body.send_keys(Keys.PAGE_DOWN)
	time.sleep(0.2)

tweets = browser.find_elements_by_class_name('tweet-text')
uname = browser.find_elements_by_class_name('username')

for i in range(len(tweets)):
	print('= = =')
	print(uname[i].text,":")
	print(tweets[i].text)
	i=i+1




# for tweet in tweets:
# 	print('= = =')
# 	print(unames.text)
# 	print(tweet.text)


