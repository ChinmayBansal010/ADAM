import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# You may uncomment this if running for the first time
# nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """Tokenizes a sentence into words."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Stems a word to its root form."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag-of-words vector:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
