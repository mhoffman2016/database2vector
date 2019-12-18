"""
TODO
"""

import argparse
import numpy as np
from db2v import db2v
from pyspark import SparkContext
from gensim.models import Word2Vec, KeyedVectors
from difflib import get_close_matches

parser = argparse.ArgumentParser(description = 'TODO',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input",help="Input file or directory to be used as corpus.")
parser.add_argument("--output",default="",help="Output file for the keyed vector to be saved (end with '.kv' extension).")
parser.add_argument("--min_count",default=1,help="Minimum frequency before the term is introduced into the vocab.")
parser.add_argument("--size",default=1000,help="Dimensionality of the vectors produced.")
parser.add_argument("--window",default=10,help="How many items in the context before and after the target are considered.")
parser.add_argument("--cbow",default=False,help="Create Word2Vec using the continuous bag of words model.",action="store_true")
parser.add_argument("--pyspark",help="Run the script with Pyspark.",action="store_true")
parser.add_argument("--master",default="local[20]",help="Spark Master.")
args = parser.parse_args()


if __name__ == "__main__":
    """
    TODO
    """

    # Creating db2v with SparkContext if specified through command line
    if args.pyspark:
        sc = SparkContext(args.master, "SparkContext used to tokenize.")
        sc.setLogLevel("ERROR")
        db2v = db2v(sc)
    else:
        db2v = db2v()

    # Loading keyed vector model or creating one from input
    if ".kv" in args.input:
        db2v.load(args.input)
    else:
        db2v.createKeyedVector(args.input, args.min_count, args.size, args.window, args.cbow)
        # Save keyed vector if given an output path
        if args.output != "":
            db2v.save(args.output)
    vocab = list(db2v.kv.vocab.keys())

    # Begin custom user loop
    print("#------------------------------------------#")
    print("| Type 'vocab' to get sample terms.        |")
    print("| Type a term to find other similar terms. |")
    print("| Type 'quit' or 'exit' when finished!     |")
    print("#------------------------------------------#")
    while(True):
        uin = input("INPUT>").lower()
        if "quit" in uin or "exit" in uin:
            exit(0)
        elif "vocab" in uin:
            print("INFO  Listing sample vocabulary.")
            print(vocab[:25])
        else:
            try:
                similar_terms = [pair[0].lower() for pair in db2v.kv.most_similar(uin,topn=10)]
                print("INFO  Terms most similar to '%s':" % uin)
                for rank, term in enumerate(similar_terms, 1):
                    print("%d. %s" % (rank, term))
            except KeyError:
                print("ERROR  Term not found.")
                close_terms = get_close_matches(uin, vocab, 5, .3)
                if len(close_terms) != 0:
                    print("INFO  Consider the following terms from the vocabulary:")
                    print(close_terms)
