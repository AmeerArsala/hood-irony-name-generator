from roman_numeral_formatting import roman_to_integer, integer_to_roman


SOS_TOKEN = "<SOS>"     # Start of Sentence Token
EOS_TOKEN = "<EOS>"     # End of Sentence Token
OOV_TOKEN = "<UNK>"     # Out of Vocabulary Token
NOTHING_TOKEN = ""  # Literally nothing 


def tokenize_to_texts(full_text: str):
    corpus = full_text.split("\n")

    for i in range(len(corpus)):  # for each sample
        corpus[i] = SOS_TOKEN + tokenize_romans_to_ints(corpus[i]) + EOS_TOKEN

    return corpus


def detokenize_texts(texts: []):
    for i, text in enumerate(texts):
        # go back to roman numerals and get rid of SOS and EOS tokens
        texts[i] = detokenize_ints_to_romans(text[len(SOS_TOKEN):-len(EOS_TOKEN)])

    return texts


def tokenize_romans_to_ints(s: str):
    delimiter = " "
    words = s.split(delimiter)
    tokenized_text = ""

    # replace integer with roman numeral
    for term in words:
        try:
            term = roman_to_integer(term)
        except:
            pass

        tokenized_text += term + delimiter
    
    # don't return with last delimiter
    return tokenized_text[0:-1]


def detokenize_ints_to_romans(s: str):
    delimiter = " "
    words = s.split(delimiter)
    detokenized_text = ""

    # replace integer with roman numeral
    for term in words:
        try:
            term = integer_to_roman(term)
        except:
            pass

        detokenized_text += term + delimiter
    
    # don't return with last delimiter
    return detokenized_text[0:-1]