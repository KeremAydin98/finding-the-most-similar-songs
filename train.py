from preprocessing import Preprocessing
import config
import numpy as np
import tensorflow as tf
from models import Word2Vec

preprocessing = Preprocessing(config.lyric_data_path,config.lyric_text_path, config.vocab_size, config.sequence_length)

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
