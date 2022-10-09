import models, config
from preprocessing import Preprocessing
import io
import numpy as np
import tensorflow as tf
import pandas as pd

# Reading the data
df_all = pd.read_csv(config.lyric_data_path)

# Drop all the columns except Lyrics and Song Name
df = df_all.loc[:, df_all.columns.intersection(['Lyric','SName','language'])]

# Filter only English songs
df = df[df["language"] == "en"]
df = df.drop("language", axis=1)

doc_list = list(df["Lyric"])

"""
1. Bag of words
"""

bow = models.BagOfWords()

bowList = bow(doc_list)

df_bow = pd.DataFrame(bowList)

df_bow.to_pickle("./Models/bow.pkl")

"""
2. Tf-idf
"""

tf_idf = models.TfIdf()

df_tfidf = pd.DataFrame(tf_idf(doc_list))

df_tfidf.to_pickle("./Models/tf_idf.pkl")

"""
3. Word2Vec
"""

preprocessing = Preprocessing(df, config.lyric_text_path, config.vocab_size, config.sequence_length)

# Generating training data
targets, contexts, labels = preprocessing.generating_training_data(sequences=preprocessing.sequences,
                                                                 window_size=config.window_size,
                                                                 num_ns=config.num_ns,
                                                                 vocab_size=config.vocab_size)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

dataset = tf.data.Dataset.from_tensor_slices((targets, contexts), labels)
dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE,drop_remainder=True)
dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

# Initialize word2vec model
word2vec = models.Word2Vec(config.vocab_size, config.EMBEDDING_SIZE)

word2vec.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])

word2vec.fit(dataset, epochs=20)

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = preprocessing.vectorize_layer.get_vocabulary()


def song_embeddings(song):

    song = song.split()
    all_embeds = []

    for word in song:

        index = vocab.find(word)

        if index != -1:

            all_embeds.append(weights[index])

        else:

            all_embeds.append(weights[0])

    return sum(all_embeds) / len(all_embeds)


df["Lyric"] = df["Lyric"].apply(song_embeddings)
df.to_pickle("./Models/word2vec.pkl")

out_v = io.open('./Models/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('./Models/metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")

out_v.close()
out_m.close()

