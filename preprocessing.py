import pandas as pd
import tensorflow as tf
import tqdm


class Preprocessing():

    def __init__(self, df, lyric_text_path, vocab_size, sequence_length):

        with open(lyric_text_path, 'w', encoding="UTF-8") as f:
            for i in range(len(df) // 10):
                f.write(str(df["Lyric"].iloc[i]))
                f.write('\n')

        # Using non-empty lines to construct a tf.data.TextLineDataset object
        text_ds = tf.data.TextLineDataset(lyric_text_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        # Number of words in the vocabulary
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # Text Vectorization
        self.vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size,
                                                                output_mode='int',
                                                                output_sequence_length=self.sequence_length)

        # Adapting to the data set
        self.vectorize_layer.adapt(text_ds.batch(1024))

        # Vectorization of the data in text_ds
        self.text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(self.vectorize_layer).unbatch()

        # Turns the elements into numpy arrays
        self.sequences = list(self.text_vector_ds.as_numpy_iterator())

    def generating_training_data(self, sequences, window_size, num_ns, vocab_size):
        """
        Generating skip-gram pairs with negative sampling for a list of sequences based on window size,
        number of negative samples and vocabulary size
        """

        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab size tokens
        # The sampling probabilities are generated according to the sampling distribution used in word2vec
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        for sequence in tqdm.tqdm(sequences):

            # Generate positive skip-gram pairs for a sequence
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0
            )

            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples
            for target_word, context_word in positive_skip_grams:

                # Generate positive skip-gram pairs for a sequence (sentence).
                context_class = tf.expand_dims(tf.constant([context_word],dtype="int64"), 1)

                # Iterate over each positive skip-gram pair to produce training examples
                # with a positive context word and negative samples.
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=42,
                    name="negative_sampling"
                )

                # Build context and label vectors(for one target word
                negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

