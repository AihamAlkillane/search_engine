import nltk
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
text = "aiham42 kilany, is going to a handsome@gmail.com414 guy"
text = text.lower()
words = word_tokenize(text)
words = [stemmer.stem(t) for t in words]
print(words)