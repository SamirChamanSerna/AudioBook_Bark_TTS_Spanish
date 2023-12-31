import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.rand(10).device)

# Do this in a separate python interpreter session, since you only have to do it once
import nltk

nltk.download("punkt")

# Do this in your ipython notebook or analysis script
from nltk.tokenize import word_tokenize

sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week.",
]

sentences_tokenized = []
for s in sentences:
    sentences_tokenized.append(word_tokenize(s))
