# Cooccurrence Vectorizer
## Installation
Open terminal in this directory and run either:
1. pip install the module itself
```
pip install .
```
2. setup.py (same as 1)
```
python setup.py install
```
3. docker-compose to set up an environment with module installed
```
docker-compose up -d
```
## Getting Started
```
import cooccurrence

vectorizer = cooccurrence.CooccurrenceVectorizer(max_features=None, context_window=2) # context window of size 2
vectorizer.fit(texts) # texts is a list of lists, it should look like [['He', 'is', 'not', 'lazy', 'He', 'is', 'intelligent', 'He', 'is', 'smart'], ["I", "love", "Singapore"]]
vectors = vectorizer.transform(texts) # vectors would be a list of numpy arrays, each array will have shape (number_tokens, vocab_size)
```

## Jupyter Notebook Server
To set up a notebook server, follow step 3 of **Installation** and assuming default settings are applied, head to *http://localhost:8889/tree* to view existing or create new notebooks to perform experiments with the module.