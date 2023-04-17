import pandas as pd

imdb_file = "processed_imdb_data.csv"
imdb_reviews = pd.read_csv(imdb_file)
imdb_reviews["text_len"] = imdb_reviews["processed_review"].map(lambda doc: len(doc.split(" ")))
total_len = imdb_reviews["text_len"].sum()
avg_len = total_len / imdb_reviews.shape[0]
print(imdb_reviews.shape[0])
print(f"imdb average doc length: {avg_len}")

labels = imdb_reviews["label"]
one_labels = labels.sum()
print(one_labels)
