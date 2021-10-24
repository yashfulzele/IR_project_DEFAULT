import os
import re
import sys
import pickle
import tarfile
import unicodedata
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import collections

STOPWORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()
DATAFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
PICKLEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_queries_14.pth")

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

def preprocess_data(folderlocation):
    print("Extracting text from documents...")
    pattern = r"en_BDNews24/[0-9]+/"
    # tar = tarfile.open(filename, "r:gz")

    c=0
    dictionary = {}

    for foldername in os.listdir(folderlocation):
        folder = os.path.join(folderlocation, foldername)
        if os.path.isdir(folder) == False:
            continue
        c+=1
        c1=0
        for filename_ in os.listdir(folder):
            filename = os.path.join(folder, filename_)
            c1+=1
            if os.path.isfile(filename):    
                with open(filename, "r") as f:
                    doc = f.read()

                    docno_pattern = r"<DOCNO>(.*?)</DOCNO>"
                    docno = re.search(docno_pattern, doc).group(1)
                    st = doc.find('<TEXT>') + len('<TEXT>')
                    en = doc.find('</TEXT>')
                    text = normalizeString(doc[st:en])
                    TOKENS = [word for word in word_tokenize(text) if word not in STOPWORDS]
                    TOKENS = [LEMMATIZER.lemmatize(word) for word in TOKENS]
                    tokens_count = collections.Counter(TOKENS)
                    # printing the element and the frequency
                    for key, value in tokens_count.items():
                        if key in dictionary.keys():
                            dictionary[key].append((docno,value))
                        else:
                            dictionary[key] = []
                            dictionary[key].append((docno,value))
            print("Done, Directory:",c," ",foldername," Doc: ",c1," ",docno)     
    dictionary = {k : v for k, v in sorted(dictionary.items())}

    print("Saving file to destination...")
    with open(PICKLEFILE, "wb") as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved file!")

def save_pickle_file(filename=DATAFILE):
    if os.path.isfile(PICKLEFILE) == False or os.path.getsize(PICKLEFILE) == 0:
        preprocess_data(filename)
    else:
        print("Saved file already!")

save_pickle_file()