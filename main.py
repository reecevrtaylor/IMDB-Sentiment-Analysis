import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import re
import time
# tokenizer
from nltk.tokenize.toktok import ToktokTokenizer
# identify stopwords
from nltk.corpus import stopwords
## tfid
from sklearn.feature_extraction.text import TfidfVectorizer
# split up the data for training and testing
from sklearn.model_selection import train_test_split
# using support vector machine algorithm / model
from sklearn.svm import LinearSVC
# report on the accuracy of model
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup

csv = 'IMDB Dataset.csv'

print(f"importing data from: '{csv}'")
data = pd.read_csv(csv)

# text tokenization
tokenizer = ToktokTokenizer()
# setting up the list of english stopwords
stopword_list = nltk.corpus.stopwords.words('english')

print("########### STARTING PREPROCESSING OF DATA ###########\nEstimated time of completion: 5 mins\n")

# time how long the program takes
start_time = time.time()

print("removing noise...")

# removing html tags
def rem_html(text):
    parse = BeautifulSoup(text, "html.parser")
    return parse.get_text()

# removing space between characters
def rem_space(text):
    return re.sub('\[[^]]*\]', '', text)

# using the two above functions to remove the noise from text
def rem_noise(text):
    text = rem_html(text)
    text = rem_space(text)
    return text

# apply to reviews column from data
data['review'] = data['review'].apply(rem_noise)
print("noise removed\n")

# function for removing special characters
print("removing special characters...")
def rem_special(text, remove_digits=True):
    regex = r'[^a-zA-z0-9\s]'
    text = re.sub(regex,'',text)
    return text

# apply to reviews column from data
data['review'] = data['review'].apply(rem_special)
print("special characters removed\n")

# Stemming the text
print("starting stemming...\nthis may take a moment...")
def stem(text):
    porter = nltk.porter.PorterStemmer()
    text = ' '.join([porter.stem(word) for word in text.split()])
    return text

# apply to reviews column from data
data['review'] = data['review'].apply(stem)
print("stemming complete\n")

# removing the stopwords
print("removing stopwords...\nthis may take a moment...")
def rem_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filter = [token for token in tokens if token not in stopword_list]
    else:
        filter = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filter)    
    return filtered_text

# apply to reviews column from data
data['review'] = data['review'].apply(rem_stopwords)
print("stopwords removed\n")

# showing head preview to see if preprocessing has changed the data
# data.head()

# test to see if it has removed tags by printing review
# print(data['review'])

tfidf = TfidfVectorizer()
x = data['review']
y = data['sentiment']

x = tfidf.fit_transform(x)

## set test size to 0.2 (10000 instances), other 0.8 (40000) = training size 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# using support vector machine
svc = LinearSVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

# print out summary of training model using classification_report
print("########### SUMMARAY ###########\n", classification_report(y_test, y_pred))
print(f"Elapsed time {time.time() - start_time}\n")

print("########### TEST STRINGS ###########")
# custom tests
# cust_x = 'this film was really good! i loved it a lot'
# cust_y = 'this film was awful! I hated it'
# cust_z = 'some of this film was good, most of this film was bad'

# vec_cust_x = tfidf.transform([cust_x])
# vec_cust_y = tfidf.transform([cust_y])
# vec_cust_z = tfidf.transform([cust_z])
# print(f"The sentence '{cust_x}', gets the result: ", svc.predict(vec_cust_x))
# print(f"The sentence '{cust_y}', gets the result: ", svc.predict(vec_cust_y))
# print(f"The sentence '{cust_z}', gets the result: ", svc.predict(vec_cust_z))

while True:
    test_input = input("Please enter a test string\n")
    vec_test_input = tfidf.transform([test_input])
    print(f"The sentence '{test_input}', gets the result: ", svc.predict(vec_test_input))