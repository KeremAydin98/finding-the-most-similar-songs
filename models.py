import string
import numpy as np
import tensorflow as tf
import tqdm
from nltk.corpus import stopwords


class BagOfWords(tf.keras.Model):
    """
    Bag-of-words representation for the document(for us document is a song). This representation return the number of
    each word in a certain document. In other words, it is a statistical approach to similarity, therefore it has no
    information on the position of the words or their context.
    """

    def __init__(self):

        super().__init__()

        self.bowList = []

    def call(self, docList):

        for doc in tqdm.tqdm(docList):

            # Remove punctuation
            doc = doc.translate(str.maketrans("", "", string.punctuation))

            words = doc.split()

            # Removing stop words
            stop_words = set(stopwords.words('english'))

            words = [w for w in words if not w.lower() in stop_words]

            # Start the dictionary with the words of the doc
            wordDict = dict.fromkeys(words, 0)

            # Count the number of words
            for word in words:
                wordDict[word] += 1

            self.bowList.append(wordDict)

        return self.bowList


class TfIdf(tf.keras.Model):
    """
    TF-idf uses the frequency of the words according to two values: tf and idf. TF is the abbreviation for term
    frequency, and it is calculated with the division of number of repetitions of word in a document by number of words
    in a document. IDF stands for inverse document frequency, and it is calculated with the log of number of documents
    divided by the number of documents containing the word.

    TF =  (Number of repetitions of word in a document) / (# of words in a document)

    IDF = Log[ (Number of documents) / (# of words in a document)]
    """

    def __init__(self):

        super().__init__()

        self.tfList = []
        self.IDF_list = []
        self.final_doc_dict = {}

    def call(self, docList):

        """
        Computes tf values for each document

        """

        for doc in tqdm.tqdm(docList):

            # Remove punctuation
            doc = doc.translate(str.maketrans("", "", string.punctuation))

            # Split on whitespaces
            s_doc = doc.split()

            stop_words = set(stopwords.words('english'))

            s_doc = [w for w in s_doc if not w.lower() in stop_words]

            # Form a dictionary with the words in the document
            s_dict = dict.fromkeys(s_doc, 0)

            # Number of words in the document
            n_of_words = len(s_doc)

            # Counting the number of each word
            for word in s_doc:
                s_dict[word] += 1

            # Divide the counts of words by the number of words in the document
            s_dict.update((x, y / n_of_words) for x, y in s_dict.items())

            self.tfList.append(s_dict)

        """
        Now we will calculate idf and finally tf-idf
        """

        # Number of documents
        n_doc = len(docList)

        """
        1. Calculating IDF
        """

        for i in range(len(docList)):

            # bag of words representation
            BOW = dict.fromkeys(docList[i], 0)

            for word, val in BOW.items():
                BOW[word] = np.log10(n_doc / (val + 1))

            self.IDF_list.append(BOW)

        """
        2. Calculating TF-IDF
        """

        final_doc_list = []

        for i, doc in enumerate(docList):
            final_doc = {}

            for word in doc:
                final_doc[word] = self.tfList[i][word] * self.IDF_list[i][word]

            final_doc_list.append(final_doc)

        return self.final_doc_dict(final_doc_list)


class Word2Vec(tf.keras.Model):
    """
    Word2Vec is just another technique to train word embeddings. We are going to use skip gram method, therefore we will
    train the word embeddings by predicting context from the target word.
    """
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

        # output[b,c] = sum_c word_emb[b,e] * context_emb[b,c,e]
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots








