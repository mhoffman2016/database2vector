"""
TODO
"""

import argparse
import numpy as np
from db2v import db2v
from pyspark import SparkContext
from gensim.models import Word2Vec, KeyedVectors

parser = argparse.ArgumentParser(description = 'TODO',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input",help="Input file or directory to be used as corpus.")
parser.add_argument("--output",default="",help="Output file for the keyed vector to be saved (end with '.kv' extension)")
parser.add_argument("--pyspark",help="Run the script with Pyspark",action="store_true")
parser.add_argument("--master",default="local[20]",help="Spark Master")
args = parser.parse_args()


if __name__ == "__main__":
    """
    TODO
    """
    print("INFO  Starting 'Database2Vector' main process.")
    # Creating db2v with SparkContext if allowed
    if args.pyspark:
        sc = SparkContext(args.master, "SparkContext used to tokenize.")
        sc.setLogLevel("ERROR")
        print("INFO  Parellel option selected.")
        db2v = db2v(sc)
    else:
        db2v = db2v()
    # Loading keyed vector or creating one from input
    if ".kv" in args.input:
        db2v.kv = KeyedVectors.load(args.input)
        print("INFO  Using model from '%s'." % args.input)
    else:
        data = db2v.tokenize(args.input)
        param = (1, 1000, 10, 1)
        db2v.kv = Word2Vec(data,
                           min_count = param[0],
                           size = param[1],
                           window = param[2],
                           sg = param[3]).wv
        # Save keyed vector if given an output path
        if args.output != "":
            print("INFO  Saving model in '%s'." % args.output)
            db2v.kv.save(args.output)
    # Begin user loop
    uin = ""
    vocab = list(db2v.kv.vocab.keys())
    print("#------------------------------------------#")
    print("| Type 'vocab' to get sample terms.        |")
    print("| Type a term to find other similar terms. |")
    print("| Type 'exit' when finished!               |")
    print("#------------------------------------------#")
    while("exit" not in uin.lower()):
        uin = input("DB2V#:")
        if "vocab" in uin.lower():
            print(vocab[:25])
        else:
            try:
                similar_terms = [pair[0].lower() for pair in db2v.kv.most_similar(uin.lower(),topn=10)]
                print("INFO  Terms most similar to '%s'." % uin)
                for rank, term in enumerate(similar_terms, 1):
                    print("%d. %s" % (rank, term))
            except:
                print("ERROR  Term not found.")
