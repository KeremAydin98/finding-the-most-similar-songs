from preprocessing import Preprocessing
import config
import numpy as np
import tensorflow as tf

preprocessing = Preprocessing()

# Generating training data
targets, contexts, labels = preprocessing.generating_training_data(sequences=preprocessing.sequences,
                                                                 window_size=config.window_size,
                                                                 num_ns=config.num_ns,
                                                                 vocab_size=config.vocab_size,
                                                                 seed=42)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

dataset = tf.data.Dataset.from_tensor_slices((targets, contexts), labels)
dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE,drop_remainder=True)
dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

