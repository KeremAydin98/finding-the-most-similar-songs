import sklearn
import tensorflow as tf


class Word2Vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, num_ns):

        super(Word2Vec, self).__init__()

        self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          input_length=1,
                                                          name="w2v_embedding")

        self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                                           embedding_dim,
                                                           input_length=num_ns+1)

    def call(self, pair):

        target, context = pair

        if len(target.shape) == 2:

            target = tf.squeeze(target, axis=1)

        word_emb = self.target_embedding(target)

        context_emb = self.context_embedding(context)

        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots


class BagOfWords(tf.keras.model):
    """
    Bag-of-words representation for the document(for us song). This representation return the number of each word in a
    certain document. In other words, it is a statistical approach to similarity, therefore it has no information on the
    position of the words or their context.
    """
    def __init__(self):

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()

    def call(self, text):

        self.tokenizer.fit_on_texts(text)

        return list(self.tokenizer.word_index.keys()), self.tokenizer.texts_to_matrix(text, mode="count")


class TfIdf(tf.keras.model):
    """
    TF-idf uses the frequency of the words according to two values: tf and idf. TF is the abbreviation for term
    frequency, and it is calculated with the division of number of repetitions of word in a document by number of words
    in a document. IDF stands for inverse document frequency, and it is calculated with the log of number of documents
    divided by the number of documents containing the word.

    TF =  (Number of repetitions of word in a document) / (# of words in a document)

    IDF = Log[ (Number of documents) / (# of words in a document)]
    """

    def __init__(self):

        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()

    def call(self, text):

        self.vectorizer.fit(text)

        return self.vectorizer


