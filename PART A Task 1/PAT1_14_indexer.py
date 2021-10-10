import os
import re
import sys
import pickle
import tarfile
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()
DATAFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
PICKLEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data\model_queries_14.pth")

if os.path.exists(DATAFILE) != True:
    print("No such file exists!")
    print("Aborting the program...")
    sys.exit(0)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def preprocess_data(filename):
    print("Extracting text from documents...")
    pattern = r"en_BDNews24/[0-9]+/"
    tar = tarfile.open(filename, "r:gz")
    corp = []
    for fname in tar.getnames():
        if re.match(pattern, fname):
            f = tar.extractfile(fname)
            corp.append(f.read().decode('utf-8'))

    print("Extracting specific parts of text...")

    sub1 = "<TEXT>"
    sub2 = "</TEXT>"
    sub3 = "<DOCNO>"
    sub4 = "</DOCNO>"
    sub5 = "<TITLE>"
    sub6 = "</TITLE>"
    corpus = []
    for i in range(len(corp)):
        idx1 = corp[i].index(sub1)
        idx2 = corp[i].index(sub2)
        idx3 = corp[i].index(sub3)
        idx4 = corp[i].index(sub4)
        idx5 = corp[i].index(sub5)
        idx6 = corp[i].index(sub6)
        text = ''
        for idx in range(idx1 + len(sub1) + 1, idx2):
            text  = text + corp[i][idx]
        docno = ''
        for idx in range(idx3 + len(sub3) + 1, idx4):
            docno  = docno + corp[i][idx]
        title = ''
        for idx in range(idx5 + len(sub5) + 1, idx6):
            title  = title + corp[i][idx]
        res = {
            "DOCNO" : docno,
            "TITLE" : normalizeString(title),
            "TEXT" : normalizeString(text),
            "TOKENS" : []
            }
        corpus.append(res)
    
    print("Tokenizing, removing stopwords and lemmatizing...")
    
    for dict in corpus:
        text = dict["TEXT"]
        dict["TOKENS"] = [word for word in word_tokenize(text) if word not in STOPWORDS]
        dict["TOKENS"] = [LEMMATIZER.lemmatize(word) for word in dict["TOKENS"]]
        dict["TOKENS"] = list(set(dict["TOKENS"]))
    
    dictionary = {}
    print("Building inverted index...")

    for dict in corpus:
        docno = dict["DOCNO"]
        tokens = dict["TOKENS"]
        for tk in tokens:
            if tk in dictionary.keys():
                dictionary[tk].append(docno)
            else:
                dictionary[tk] = []
                dictionary[tk].append(docno)
    
    dictionary = {k : v for k, v in sorted(dictionary.items())}

    print("Saving file to destination...")
    with open(PICKLEFILE, "wb") as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved file!")

def save_pickle_file(filename=DATAFILE):
    if os.path.isfile(PICKLEFILE) == False or os.path.getsize(PICKLEFILE) == 0:
        preprocess_data(filename)
    else:
        print("Saved file!")

save_pickle_file()