import os
import re
import sys
import pickle
import unicodedata

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAT1_14_results.txt")

if os.path.exists(MODEL_FILE) != True or os.path.exists(QUERY_FILE) != True:
    print("No such file(s) exists!")
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

def results(model_file, query_file):
    print("Loading the files...")
    with open(model_file, 'rb') as handle:
        invertedIndex = pickle.load(handle)
    
    with open(query_file, 'r') as f:
        lines = f.readlines()
    
    queries = {}
    for line in lines:
        r = line.split(',')
        id = r[0]
        rest = re.split(r'\t+', r[1])
        query = []
        for i in range(len(rest)):
            if len(rest[i]) != 0:
                query.append(normalizeString(rest[i]))
        queries[id] = query
    
    print("Calculating results...")
    result = {}
    for id, query in queries.items():
        docs = {}
        for q in query:
            if q in invertedIndex.keys():
                docs[q] = invertedIndex[q]
        docs = {k : v for k, v in sorted(docs.items(), key=lambda x : len(x[1]))}
        list_of_docs = [set(v) for _, v in docs.items()]
        final_docs = list_of_docs[0].intersection(*list_of_docs)
        result[id] = final_docs

    print("Saving the file...")
    with open(RESULTS_FILE, 'w') as f:
        for id, docs in result.items():
            f.write(id + ":")
            for d in docs:
                f.write("\t" + d)
            f.write("\n")
    
    print("Saved file!")

def save_results_file(model_file=MODEL_FILE, query_file=QUERY_FILE):
    if os.path.isfile(RESULTS_FILE) == False or os.path.getsize(RESULTS_FILE) == 0:
        results(model_file, query_file)
    else:
        print("Saved file already!")

save_results_file()