# Machine Learning: TensorFlow, Keras, Sci-Kit Learn

Personal notes

## Subfields

- linear regression
- clustering
- anomaly detection - is reading high? Someone trying to hack me? Fraudulent cc transaction?
- reinforcement learning (speed up / slow down, raise/lower temp, Roomba; continuing vacuuming or stop)

## Project: judging sentiment on Twitter

* suggestion: learn to use Excel, if you want quick data analysis
* can use pandas to read in the data frame

=> Everyone should spend more time manually reviewing the data.

## Feature Extraction Examples:

How do we turn text into numbers?

Simplest = "bad of words" - count # of times any word appeared. Lots of these will have a count of 0!

Fit
- set everything up on a set of data ... it's always part of the pipeline
- looks at each row one by one

It's very important to dig into how your model works -- e.g. in the case of SciKitLearn bag-of-words, it drops symbols. This probably isn't good if we're analyzing Twitter, since emojis ";D" and hashtags "#foo" are very meaningful there.

## Choose an algorithm

http://scikit-learn.org/stable/tutorial/machine_learning_map/

Category? YES

- 3 options: Positive, Negative, Neutral

Though it could be an Ordering...

- Positive-negative-neutral (+1,0,-1)
- We could say full range from -1 to 1

Labeled data? YES

< 100k samples? YES

Linear SVC not working? YES

Text Data? YES

=> Naive Bayes

open `classifier.py`

=> ... MultinomialNB, GaussianNB, BernoulliNB? Just try em...

n.b. "deep learning" algos? Basic problem: algos are slow. so they have
lots of optimizations, but these make them hard to run locally.

Can do a lot with really simple algorithms, so long as you have more and more data.

## Improving your model

In order of leverage, you can use these three approaches:

1. better training data
2. better feature extraction
3. changing your model

## Let's compare algorithms!

We're using a simple accuracy measurement...
Other things like F-Score exist.

`test-algorithm-1.py`

...

`test-dummy-algorithm.py` - baseline. always guesses the most common answer! sometimes this is more accurate than things you build!

## Cross Validation

- test-algorithm-cross-validation.py
- test-algorithm-cross-validation-dummy.py

"We're data scientists, so we ignore deprecation errors. Leave that to the engineers to fix later..."

## Pipeline

- pipeline.py
- pipeline-bigrams.py
- pipeline-bigrams-cross-validation.py

## Grid Search

"Algorithm, and a way of life"

Sci-kit makes it easy to try many variations on a model at once.

## Deep Learning

Keras: it's built on top of Tensorflow. Experts in field use it, so recommend
learning it b/c it's easier and still useful if advanced.

**Perceptron**:

- simplest machine learning algorithm.
- also the first one.
- he literally made them with wires. neuron metaphors.
- building block for neural nets

multiple inputs.. "wires with different amounts of electricity". perceptron takes the inputs and then transforms them into an "output".

inputs -> weights -> net input function -> activiation function -> output

TODO: Find this slide and understand it better.

Check out `perceptron.py` to see a Perceptron in SciKitLearn.

## Image Processing

One example use case for "deep learning": image recognition.

Digit recognition. `keras-digits.py`

Tensor ~= vector in a matrix (2-dims), but with arbitrary dimensions

## Tensors

1 hot encoding - basically an identity matrix

Example:

```
White 	1	0	0
Red		0	1	0
Black	0	0	1
```

