"""
TODO
"""

import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
warnings.filterwarnings(action = 'ignore')   
import pandas as pd
from tqdm import tqdm
from pyspark import SparkContext
from gensim.models import Word2Vec, KeyedVectors

def stripNonAlpha(s):
    """ 
    Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana'
    """
    return ''.join([c for c in s if c.isalpha()])


class db2v:
    """
    TODO
    """
    def __init__(self, sc=None):
        """
        TODO
        """
        print("INFO  Starting 'Database2Vector' main process.")
        self.sc = sc
        self.kv = None


    def save(self, output_path):
        """
        TODO
        """
        self.kv.save(output_path)
        print("INFO  Saving model in '%s'." % output_path)
        return


    def load(self, input_path):
        """
        TODO
        """
        try:
            self.kv = KeyedVectors.load(input_path)
            print("INFO  Using model from '%s'" % input_path)
        except:
            print("ERROR  Cannot open input file. Exiting...")
            exit(-1)
        return


    def createKeyedVector(self, input_path, min_count, size, window, cbow):
        """
        TODO
        """
        data = self.tokenize(input_path)
        param = (min_count, size, window, not cbow)
        self.kv = Word2Vec(data,
                           min_count = param[0],
                           size = param[1],
                           window = param[2],
                           sg = param[3]).wv
        return


    def tokenize(self, input_path):
        """
        TODO
        """
        if "xlsx" in input_path:
            return self.xlsxTokenize(input_path)
        return self.miscTokenize(input_path)


    def xlsxTokenize(self, input_path):
        """
        TODO
        """
        print("INFO  Excel file given, reading file (user will select fields to use for target/context).")
        temp_data = pd.read_excel(input_path)
        temp_data.dropna(inplace=True)
        print("#------------------------------------------#")
        print(temp_data.head())
        print("#------------------------------------------#")
        while(True):
            target_field = input("INPUT>Enter 'target field' to base history/context on (case sensitive):")
            context_field = input("INPUT>Enter 'context field' to build history of (case sensitive):")
            try:
                temp_data[target_field] = temp_data[target_field].astype(str)
                temp_data[context_field] = temp_data[context_field].astype(str)
                target_ids = temp_data[target_field].unique().tolist()
                break
            except KeyError:
                print("ERROR  Invalid field names, returning to prompt.")
        print("INFO  Fields accepted, beginning tokenization.")
        data = []
        if self.sc != None:
            target_ids = self.sc.parallelize(target_ids)
            data = target_ids.map(lambda i: temp_data[temp_data[target_field] == i][context_field].tolist()) \
                             .map(lambda history: [word.lower() for word in history]) \
                             .collect()
        else:
            for i in tqdm(target_ids):
                history = temp_data[temp_data[target_field] == i][context_field].tolist()
                data.append([word.lower() for word in history])
        return data


    def miscTokenize(self, input_path):
        """
        TODO
        """
        print("INFO  Reading '%s' file as sentences" % input_path)
        try:
            temp_data = open(input_path)
        except:
            print("ERROR  Cannot open input file. Exiting...")
            exit(-1)
        if self.sc != None:
            print("INFO  Tokenizing data in parallel")
            data = self.sc.parallelize(temp_data)
            data = data.map(lambda doc: doc.replace("\n", " ")) \
                       .flatMap(lambda doc: sent_tokenize(doc)) \
                       .map(lambda sent: word_tokenize(sent)) \
                       .map(lambda sent: [(word.lower()) for word in sent]).cache()
            data = data.collect()
        else:
            print("INFO  Tokenizing data in serial")
            data = []
            for doc in tqdm(temp_data):
                doc_temp = []
                for sent in sent_tokenize(doc):
                    sent_temp = []
                    sent.replace("\n", " ")
                    for word in word_tokenize(sent):
                        word = stripNonAlpha(word.lower())
                        if word != "":
                            sent_temp.append(word)
                    doc_temp.append(sent_temp)
                data.extend(doc_temp)
        return data
