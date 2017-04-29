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

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# nb = MultinomialNB()
# nb = GaussianNB() # doens't work immediately
nb = BernoulliNB()
nb.fit(counts, fixed_target)

print nb.predict(count_vect.transform(["iphone!!!"]))
print nb.predict(count_vect.transform(["iphone cost too much"]))
print nb.predict(count_vect.transform(["iphone is delicious and tasty and amazing"]))
print nb.predict(count_vect.transform(["I love my iphone"]))
print nb.predict(count_vect.transform(["iphone is not bad"]))

# ones that dont work so well...
print nb.predict(count_vect.transform(["this sucks"])) # positive
print nb.predict(count_vect.transform(["my iphone sucks"])) # negative
print nb.predict(count_vect.transform(["my iphone does not suck"])) # still negative

# Bag of words in Chinese or Japanese -- how does it work, if not spaces? Or encoding issues?
# Bag of words in German -- less words, since words are often joined
