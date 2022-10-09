import pandas as pd
import models, config
import numpy as np
from training import song_embeddings

"""
Read the trained data
"""

# Bag of words
df_bow = pd.read_pickle("./Models/bow.pkl")

# Tf-idf
df_tfidf = pd.read_pickle("./Models/tf_idf.pkl")

# Word2Vec
vectors = pd.read_csv('./Models/vectors.tsv', sep='\t')
vocab = pd.read_csv('./Models/metadata.tsv', sep='\t')
df_embed = pd.read_pickle("./Models/word2vec.pkl")

"""
Set the input
"""
# Reading the data
df = pd.read_csv(config.lyric_data_path)

# Drop all the columns except Lyrics and Song Name
df = df.loc[:, df.columns.intersection(['Lyric','SName','language'])]

# Filter only English songs
df = df[df["language"] == "en"]
df = df.drop("language", axis=1)

test_input = df.iloc[config.input_song_id]
input_sName = test_input["SName"]
input_Lyric = test_input["Lyric"]

"""
Cosine distance
"""

def cosine_distance(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

"""
Calculate distance
"""

# BOW
bow = models.BagOfWords()
input_bow = bow([input_Lyric])

distances = []
for i in range(len(df_bow)):

    distance = cosine_distance(input_bow, df_bow.iloc[i])
    distance = distance, i
    distances.append(distance)

distances.sort()

print("Bag of Words:")
print(f"Closest Songs:\n{df['SName'].iloc[distances[0][1]]}\n"
      f"{df['SName'].iloc[distances[1][1]]}\n"
      f"{df['SName'].iloc[distances[2][1]]}\n"
      f"{df['SName'].iloc[distances[3][1]]}\n"
      f"{df['SName'].iloc[distances[4][1]]}")
print("-------------------------------")

# TF-idf
tf_idf = models.TfIdf()
input_tfidf = tf_idf([input_Lyric])

distances = []
for i in range(len(df_tfidf)):

    distance = cosine_distance(input_tfidf, df_tfidf.iloc[i])
    distance = distance, i
    distances.append(distance)

distances.sort()

print("Tf-idf:")
print(f"Closest Songs:\n{df['SName'].iloc[distances[0][1]]}\n"
      f"{df['SName'].iloc[distances[1][1]]}\n"
      f"{df['SName'].iloc[distances[2][1]]}\n"
      f"{df['SName'].iloc[distances[3][1]]}\n"
      f"{df['SName'].iloc[distances[4][1]]}")
print("-------------------------------")

# Word2Vec
input_embed = song_embeddings(input_Lyric)

distances = []
for i in range(len(df_tfidf)):

    distance = cosine_distance(input_embed, df_embed.iloc[i])
    distance = distance, i
    distances.append(distance)

distances.sort()

print("Tf-idf:")
print(f"Closest Songs:\n{df['SName'].iloc[distances[0][1]]}\n"
      f"{df['SName'].iloc[distances[1][1]]}\n"
      f"{df['SName'].iloc[distances[2][1]]}\n"
      f"{df['SName'].iloc[distances[3][1]]}\n"
      f"{df['SName'].iloc[distances[4][1]]}")
print("-------------------------------")


