The following general purpose text classification algorithms have been implemented for the purpose of email categorization
* [Bernoulli Naive Bayesian Method] (http://www.nltk.org/book/ch06.html#naive-bayes-classifiers)
* [Maximum Entropy Method / Log-Likelihood Method] (http://www.nltk.org/book/ch06.html#maximum-entropy-classifiers)


Both are ideal for classification of short-texts like emails, messageboards posts, short messages - usually text that is precise and to the point. Bayesian Method is supposed to be more robust and flexible, requiring less training; but it may not be as accurate as a well trained Maximum Entropy Method for multi-class classification. With enough training Maximum Entropy Method is expected and observed to have better prediction accuracy than Bayesian Method.


Tested on the following popular datasets:
* [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/) - (Accuracy of 88% using Maximum Entropy Method)
* [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) - (Accuracy of 91% using Maximum Entropy Method)


This classifier can be deployed as a webservice (using Flask) that supports training and classification using HTTP POST method.
