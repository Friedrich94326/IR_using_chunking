from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

def word_sentence_tokenise(text):
    """
    text: clean text
    -----------------------------
    Step 1: sentence segmenation
    Step 2: tokenisation
    """
    sentence_tokeniser = PunktSentenceTokenizer(text)

    # sentence tokenise text
    sentence_tokenised = sentence_tokeniser.tokenize(text)

    # create a list to hold word tokenised sentences
    word_tokenised = list()

    # for-loop through each tokenized sentence in sentence_tokenized
    for tokenized_sentence in sentence_tokenised:
        # word tokenize each sentence and append to word_tokenized
        word_tokenised.append(word_tokenize(tokenized_sentence))
    return word_tokenised
