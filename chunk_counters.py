from collections import Counter
from nltk.corpus import stopwords

def np_chunk_counter(chunked_sentences, rmvStopWords = True):
    """
    Pulls chunks out of chunked sentence and finds the most common chunks
    """
    chunks = list()

    # extract NP chunks
    for chunked_sentence in chunked_sentences:
        for subtree in chunked_sentence.subtrees(filter = lambda t: t.label() == "NP"):
            chunks.append(tuple(subtree))

    chunk_counter = Counter()

    for chunk in chunks:
        # increase counter of specific chunk by 1
        chunk_counter[chunk] += 1

    print("chunk_counter:", chunk_counter)
    # return 30 most frequent chunks
    return chunk_counter.most_common(30)



def vp_chunk_counter(chunked_sentences, rmvStopWords = True):
    """
    Pulls chunks out of chunked sentence and finds the most common chunks
    """
    chunks = list()

    # extract VP chunks
    for chunked_sentence in chunked_sentences:
        for subtree in chunked_sentence.subtrees(filter = lambda t: t.label() == "VP"):
            chunks.append(tuple(subtree))

    chunk_counter = Counter()

    for chunk in chunks:
        # increase counter of specific chunk by 1
        chunk_counter[chunk] += 1
    # remove stop words from statistical result
    print("chunk_counter:", chunk_counter)
    # return 30 most frequent chunks
    return chunk_counter.most_common(30)
