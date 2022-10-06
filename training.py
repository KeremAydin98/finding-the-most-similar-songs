import models
from preprocessing import Preprocessing
import config, io
import numpy as np
import tensorflow as tf
from models import Word2Vec
import pandas as pd

# Reading the data
df_all = pd.read_csv(config.lyric_data_path)

# Drop all the columns except Lyrics and Song Name
df = df_all.loc[:, df_all.columns.intersection(['Lyric','SName','language'])]

# Filter only English songs
df = df[df["language"] == "en"]
df = df.drop("language", axis=1)

"""
1. Bag of words
"""

bow = models.BagOfWords()

songs_bow = []

for i in range(len(df)):

    keys, vectors = bow(df["Lyric"][i])

    songs_bow.append((keys,vectors))

"""
2. Tf-idf
"""


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
word2vec = Word2Vec(config.vocab_size, config.EMBEDDING_SIZE)

word2vec.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])

word2vec.fit(dataset, epochs=20)

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = preprocessing.vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")

out_v.close()
out_m.close()

