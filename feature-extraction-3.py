import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)
print(fixed_text[0:2])

# Print counts for the first two tweets
# Only prints columns where count>0
print(counts[0:2])
print count_vect.vocabulary_.get(u'the')

print count_vect.vocabulary_.get(u'the')

# print(fixed_text[0])
# print count_vect.transform(["cerulean"])

print ""
print "Learning by trying stuff out...."

print "Pass in a new tweet"
print count_vect.transform(["i like sandwiches"])
# => maps words to existing features

print "If there's a word that's not in the training data, what does it do?"
# => it won't be able to map it into the bag of words
print count_vect.transform(["aaaaaaaaaaaaaaa"])
print "How does it handle punctuation?"
print count_vect.transform(["tacos tacos!"])
# => ignores it

print "How does it handle plural words?"
print count_vect.transform(["taco tacos"])
# => 2 different words
