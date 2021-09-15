"""
Information Retrieval using Chunking
------------------------------------
Author: Friedrich Cheng
Text data source: https://www.bbc.com/news/technology-58570353
Creation Date: Sep 15 2012
"""
from nltk import pos_tag, RegexpParser
from tokenise_words import word_sentence_tokenise
from chunk_counters import np_chunk_counter, vp_chunk_counter


def import_n_preprocess(filepath):
    """
    Import clean text file and preprocess it.
    """
    with open(filepath, encoding = "utf-8") as f:
        text = f.read().lower()
    word_tokenised_text = word_sentence_tokenise(text)
    return word_tokenised_text



def POS_tag(tokenised_text):
    """
    Part-of-speech tagging each word tokenised sentence in the text
    """
    pos_tagged_text = list()
    for word_tokenised_sent in tokenised_text:
        # part-of-speech tag each sentence and append to list of pos-tagged sentences here
        pos_tagged_text.append(pos_tag(word_tokenised_sent))
    return pos_tagged_text

  

def shallow_parsing(pos_tagged_text):
    """
    NP-chunking and VP-chunking POS tagged text
    """
    # define chunk grammars
    np_chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
    vp_chunk_grammar = "VP: {(<DT>?<JJ>*<NN>|<VB.?>)<RB.?>?}"

    # creates chunk parsers
    np_chunk_parser = RegexpParser(np_chunk_grammar)
    vp_chunk_parser = RegexpParser(vp_chunk_grammar)

    np_chunked_text = list()
    vp_chunked_text = list()

    for pos_tagged_sentence in pos_tagged_text:
        np_chunked_text.append(np_chunk_parser.parse(pos_tagged_sentence))
        vp_chunked_text.append(vp_chunk_parser.parse(pos_tagged_sentence))

    return np_chunked_text, vp_chunked_text


if __name__ == "__main__":
    tokenised_text = import_n_preprocess("data/bbc_news.txt")
    pos_tagged_text = POS_tag(tokenised_text)
    np_chunked_text, vp_chunked_text = shallow_parsing(pos_tagged_text)

    # List the most common 30 NP/ VP chunks
    rmvStopWords = True
    most_common_np_chunks = np_chunk_counter(np_chunked_text, rmvStopWords = rmvStopWords)
    print(most_common_np_chunks, end = "\n\n\n")

    most_common_vp_chunks = vp_chunk_counter(vp_chunked_text, rmvStopWords = rmvStopWords)
    print(most_common_vp_chunks)



