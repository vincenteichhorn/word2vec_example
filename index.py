from epoch_logger import EpochLogger
from sentence_iterator import SentenceIterator
from gensim import models

epoch_logger = EpochLogger()
# corpus from https://wortschatz.uni-leipzig.de/de/download
# Load Sentences from corpus
sentences = SentenceIterator('.\corpora\eng_news_2020_10K\eng_news_2020_10K-sentences.txt')
# Train Word2Vec model with sentences, add callback for training epochs
model = models.Word2Vec(sentences=sentences, callbacks=[epoch_logger], workers=4, epochs=15)

# Get top 10 most similar words (similar embeddings) for words in list
words = ['time', 'year', 'COVID19', 'government', 'works']
for word in words:
    most_sim = model.wv.most_similar(positive=[word], topn=10)
    print(word, [sim[0] for sim in most_sim])


# Als Metrik für die sematische Ähnlichkeit eignet sich die Cosine Similarity besser als die Euclidean Distance,
# da word2vec die Ähnlichkeit bzw. Unterschiedlichkeit der Word Embeddings in der Richtung der Vektoren (Winkel zwischen zwei Vektoren) 
# abbildet und der Betrag nicht von großer Bedeutung ist. Dies kommt durch die Aktivierung der OutputUnit des word2vec-Models 
# mit der Softmax-Funktion. Hierbei werden sehr große Beträge gegenüber größeren ignoriert, da die Softmax-Funktion für
# x->infinity gegen 1 konvergiert und somit für gleiche Aktivitäten in der OutputUnit sorgt. Weiterhin ist die Cosine Similarity
# weitesgehend immun gegen "Inkonsistenzen" im Textkorpus. So haben Wörter mit einer hohen häufigkeit auch Vektoren mit kleineren
# Beträgen, was zu einer kleineren Eclidean Distance zu Wörtern mit geringerem Vorkommen im Korpus führt, obwohl diese von der
# Bedetung ähnlich sind.
# Beispielsweise haben die Vektoren (8, 6) und (16, 12) einen unterschiedlichen Betrag, jedoch die gleiche Richtung. 
# Also einen Euclcidean Distance von >>0, jedoch eine Cosine Similarity von 1. Wenn diese Vektoren als Embedding in der
# HiddenUnit vorliegen erzeugen sie eine ähnliche Aktivität in der OutputUnit, was zu einer ähnlichen Fehlerbeurteilung kommt.

# Das Word2Vec-Model bringt den Vorteil, dass man eine mathematische Representation von Wörtern erzeugen kann. Dabei erhalten
# ähnliche Worte eine ähnliche mathematische Representation. Diese Eigenschaft kann man sich zu nutze machen und nicht nur dei
# Ähnlichkeit von zwei Wörtern evaluieren, sondern auch von ganzen Sätzen oder Texten. Die einzelnen Vektoren jedes Wortes kann
# man zu einem Vektor für einen Satz oder Text zusammenfassen. Die Eigenschaft, dass die Ähnlichkeit über die Richtung des Vektors 
# (Winkel) abgebildet wird bleibt dabei erhalten.
