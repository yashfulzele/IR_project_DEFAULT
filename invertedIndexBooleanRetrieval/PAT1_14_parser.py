import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()
RAW_TEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
TEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries_14.txt")

if os.path.exists(RAW_TEXT_FILE) != True:
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
    s = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def preprocess_queries(filename):
    print("Parsing the raw_queries.txt...")
    topics = ET.parse(filename)

    id_query_tks = []
    for top in topics.getroot():
        for i in top:
            if i.tag == 'num':
                num = i.text
            if i.tag == 'title':
                query = i.text
        dictionary = {'num' : num, 'title' : query, 'tokens' : []}
        id_query_tks.append(dictionary)

    for dict in id_query_tks:
        dict['title'] = normalizeString(dict['title'])
        for word in word_tokenize(dict['title']):
            if word not in STOPWORDS and len(word)>1:
                dict['tokens'].append(word)
        dict['tokens'] = [LEMMATIZER.lemmatize(word) for word in dict['tokens']]
        dict['tokens'] = list(set(dict['tokens']))

    print("Saving file to destination...")
    with open(TEXT_FILE, 'w') as f:
        for dict in id_query_tks:
            query_id = dict['num']
            tokens = dict['tokens']
            f.write(query_id + ",")
            for tk in tokens:
                f.write("\t" + tk)
            f.write("\n")
    
    print("Saved file!")

def save_text_file(filename=RAW_TEXT_FILE):
    if os.path.isfile(TEXT_FILE) == False or os.path.getsize(TEXT_FILE) == 0:
        preprocess_queries(filename)
    else:
        print("Saved file already!")

save_text_file()