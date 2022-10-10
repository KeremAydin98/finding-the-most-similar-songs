# semantic-shazam

Normally the app Shazam uses syntaxtic data on figuring out which song is playing. While I was working on text representations, I wondered if I can build a model that can find out the closest songs to the input song in terms of the semantics. In other words, the model finds the closest songs which have lyrics with the most similar meaning. That's why I called this project "Semantic Shazam".

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/194806554-280ccb9f-af95-4323-8285-d1454840bed1.png" />
</p>

There are three models which are getting more complex in order: bag of words, tf-idf, word2vec. All three have been developed from scratch. 

The basic intuition behind bag of words method is the most similar texts or documents would contain the same words. Therefore all the algorithm does is counting every single word in text and form a table for every single text. However, since the most common words like "the" or "a" have no effect on the meaning of the text. These were removed before forming the table. In literature, these kind of words are called "stop words".

Tf-idf can be considered as the improved version of bag-of-words method. Tf-idf gives more importance to a word in terms on the frequency of the word in the document but also gives less importance when the word is located in other documents as well. These two concepts are called term frequency and inverse document frequency in order and combination of them is the tf-idf method. 

Word2Vec is the real model of the project since it actually creates some kind of representation for the semantic of a word. Word2Vec uses word embeddings which are basically vectors. The key idea of the Word2Vec is to figuring out the word meaning from the neighboring words. This is actually what people do when they do not understand single word in a sentence. We try to figure out the meaning of the word by examining the context. There are two ways to create train set for the word2vec algorithm: skip-gram and continuous bag of words. I have used skip-gram which makes the model predict the context words from the target word unlike the continuous bag of words method which makes the model predict the target word from the context words. In this way, I have trained word embedding for the words that are inside the song lyrics. Then I replaced the words with their word embeddings and created embedding for the song by averaging its words' embedding vectors.

In these ways, I represented the songs and found the closest songs by calculating the distance with the cosine similarity method.
