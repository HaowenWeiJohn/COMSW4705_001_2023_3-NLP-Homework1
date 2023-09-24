
######### Part 1 - extracting n-grams from a sentence (20 pts) #########
def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1

    #     >>> get_ngrams(["natural","language","processing"],1)
    # [('START',), ('natural',), ('language',), ('processing',), ('STOP',)]
    # >>> get_ngrams(["natural","language","processing"],2)
    # ('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]
    # >>> get_ngrams(["natural","language","processing"],3)
    # [('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]

    """
    sequence_pad = ["START"] + sequence + ["STOP"]
    if n > 2:
        sequence_pad = ["START"]*(n-2) + sequence_pad

    ngrams = []
    for i in range(len(sequence_pad)-n+1):
        ngrams.append(tuple(sequence_pad[i:i+n]))



    return ngrams


sequence = ["natural","language","processing"]
n = 3

print(get_ngrams(sequence, n))

