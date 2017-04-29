import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

# This removes the _same_ rows from both text and target dataframes
# If we wrote
#   fixed_target = target[pd.notnull(target)]
# it would be a subtle mistake

# If it were a real problem..
# - Ensure you're not dropping lots of rows
# - Figure out why this data is null?
fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)] # would be

from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer() takes a text list.
# Each column in vector is a word, and each row contains
# the # of times the word appears in the corresponding text.
# Example:
# => Input: "hello world"
# => Output: {
#   "hello": 1,
#   "world": 1,
#   "all": 0,
#   "other": 0,
#   "words": 0,
#   ....
# }
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

# get the column that corresponds to 'the'
print count_vect.vocabulary_.get(u'the')

# how many vocab words total do we have?
print "Columns (words)=", len(count_vect.vocabulary_)
print "Rows (# tweets)=", len(fixed_text)
