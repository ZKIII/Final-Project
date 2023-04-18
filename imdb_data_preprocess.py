import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from collections import Counter

nltk.download('stopwords')
stop_words = stopwords.words("english")
filter_tokens = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@','\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]

def process(text):
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(filter_tokens), " ", text, flags=re.S)
    word_list = [w.strip().lower() for w in text.split()]    
    return word_list

imdb_movie = "imdb_dataset.csv"
imdb_data = pd.read_csv(imdb_movie)
imdb_data["cleaned_review"] = imdb_data["review"].map(process)
imdb_data["label"] = imdb_data["sentiment"].map(lambda x: 1.0 if x == "positive" else 0.0)

clean_content = imdb_data["cleaned_review"]
uniq_clean_content = clean_content.apply(lambda l: set(l))
vocab = list(reduce(lambda s1, s2: s1.union(s2), uniq_clean_content))
print(f"vocab size: {len(vocab)}")
vocab_dict = dict()
for i, w in enumerate(vocab):
    vocab_dict[w] = str(i)
review_idxes = clean_content.apply(lambda doc: " ".join([vocab_dict[w] for w in doc]))
imdb_data["processed_review"] = review_idxes
processed_data = imdb_data[["processed_review", "label"]]
processed_data.to_csv("./processed_imdb_data.csv", index=False)

with open("vocab.dat", "w", encoding="utf-8") as fpw:
    for w in vocab_dict:
        fpw.write(f"{w}\t{vocab_dict[w]}\n")

review_len = clean_content.apply(lambda doc: len(doc))
df = pd.DataFrame.from_dict(Counter(review_len.tolist()), orient='index', columns=['w_count'])
df.plot.bar()
plt.show()