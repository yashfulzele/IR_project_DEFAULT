import os
import re
import sys
import pickle
import math
import unicodedata
import csv

CORPUS_SIZE = 0
DATAFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
PICKLEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[3])

OUTPUT_FILE_A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAT2_14_ranked_list_A.csv")
OUTPUT_FILE_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAT2_14_ranked_list_B.csv")
OUTPUT_FILE_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAT2_14_ranked_list_C.csv")

docno_indexed_dict = {}
query_indexed_dict = {}
tfSumDoc = {}
tfMaxDoc = {}
tfDocTerm = {}

if os.path.exists(DATAFILE) != True:
    print("No such file exists!")
    print("Aborting the program...")
    sys.exit(0)

if os.path.exists(PICKLEFILE) != True:
    print("No such pickle file exists!")
    print("Aborting the program...")
    sys.exit(0)


with open(PICKLEFILE, "rb") as f:
    invertedIndex = pickle.load(f)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def preprocess_data():
    print("preprocessing invertedIndex file...")
    for key,value in invertedIndex.items():
        for val in value:
            if val[0] in docno_indexed_dict.keys():
                docno_indexed_dict[val[0]].append(key)
            else:
                docno_indexed_dict[val[0]] = []
                docno_indexed_dict[val[0]].append(key)
            tfDocTerm[(key,val[0])] = int(val[1])
            if val[0] in tfSumDoc.keys():
                tfSumDoc[val[0]]+=int(val[1])
            else:
                tfSumDoc[val[0]]=int(val[1])
            if val[0] in tfMaxDoc.keys():
                tfMaxDoc[val[0]]=max(tfMaxDoc[val[0]],int(val[1]))
            else:
                tfMaxDoc[val[0]]=int(val[1])

    global CORPUS_SIZE
    CORPUS_SIZE = len(docno_indexed_dict)
    print("Total Number of Documents is", CORPUS_SIZE) 
    
def preprocess_queries(filename):
    print("Parsing the preprocessed queries file...")
    with open(filename, "r") as f:
        Lines = f.readlines()
        for text in Lines:
            c=0
            queryId = 0
            for token in text.split():
                if c:
                    query_indexed_dict[queryId].append(token)
                else:
                    queryId = int(token[:-1]) 
                    query_indexed_dict[queryId]=[]
                c+=1
    print("Completed!")

def IDF(t,c = 't'):
    if c == 'n':
        return 1
    if t in invertedIndex.keys():
        if c == 't':
            idf = CORPUS_SIZE/len(invertedIndex[t])
            return math.log(idf)
        
        if c == 'p':
            idf = (CORPUS_SIZE - len(invertedIndex[t]))/len(invertedIndex[t])
            return max(math.log(idf), 0)
    else:
        return 0

def TF(t,d,c = 'n',type='d'):
    if type == 'q':
        if t in query_indexed_dict[d]:
            return 1
        else:
            if c=='a':
                return 0.5
            else:
                return 0

    count = 0 
    if (t,d) in tfDocTerm.keys():
        count = tfDocTerm[(t,d)]
    max_count= tfMaxDoc[d]
    avg_tf= tfSumDoc[d] 
    avg_tf /= len(docno_indexed_dict[d])
    
    if c == 'l':
        if count==0:
            return 0
        return (1 + math.log(count))
    
    if c == 'L':
        if count==0:
            return 0
        ret = (1 + math.log(count))
        ret /= (1 + math.log(avg_tf))
        return ret
    
    if c == 'a':
        ans = 0.5
        ans += (0.5*count)/max_count
        return ans
    return 0

def TF_IDF(term,id_,c,type = 'd'): #
    
    tf = TF(term,id_,c[0],type)

    if tf == 0:
        return 0
    idf = IDF(term,c[1])
    return tf*idf


normalization_coeff={}
def compute_normalization_coeff(c1):  
    for docno in docno_indexed_dict.keys():
        normalization_coeff[(docno,c1)] = 0
        for val in docno_indexed_dict[docno]:
            normalization_coeff[(docno,c1)] += (TF_IDF(val, docno, c1)**2)
        
        normalization_coeff[(docno,c1)] = math.sqrt(normalization_coeff[(docno,c1)])



def COMPUTE_SCORE(query, c1, c2):   
    
    TFIDF_query = []
    for t in query_indexed_dict[query]:
        TFIDF_query.append(TF_IDF(t,query,c2,'q'))

    norm_coefficient = 0                                         #normalization
    for i in range(len(TFIDF_query)):
        norm_coefficient += TFIDF_query[i]**2

    norm_coefficient = math.sqrt(norm_coefficient)
    if norm_coefficient!=0:
        for i in range(len(TFIDF_query)):
            TFIDF_query[i] /= norm_coefficient
   
    # num_of_doc = 0
    DocScores = []
    for docno in docno_indexed_dict.keys():
        
        # num_of_doc+=1
        # if num_of_doc%10000==0:
        #     print("computed score for ",num_of_doc," docs.")

        TFIDF_doc = []
        for term in query_indexed_dict[query]:
            TFIDF_doc.append(TF_IDF(term,docno,c1))
        norm_coefficient = normalization_coeff[(docno,c1)]
        if c1[0]=='a':
            norm_coefficient = 1

        dq_score = 0
        for i in range(len(TFIDF_doc)):
            if norm_coefficient!=0:
                TFIDF_doc[i] /= norm_coefficient
            dq_score += (TFIDF_doc[i]*TFIDF_query[i])

        DocScores.append((dq_score,docno))

    DocScores.sort(reverse = True)
    top50docs =  DocScores[:50]
    return top50docs

def main():
    # INC ITC 
    if os.path.isfile(OUTPUT_FILE_A) == True:
        print("Output file A already exists!")
    else:
        print("Computing normalization_coeff for all docs(\"lnc\")...")
        compute_normalization_coeff("lnc")
        print("Completed!")

        with open(OUTPUT_FILE_A, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['query_id', 'document_id']
            writer.writerow(header)
            num_of_query=0
            for queryno in query_indexed_dict.keys():
                num_of_query+=1
                print("computing doc scores for ",num_of_query,"/",len(query_indexed_dict)," query.")
                res = COMPUTE_SCORE(queryno, "lnc", "ltc")
                for row in res:
                    writer.writerow([queryno, row[1]])
    print("Part A(\"lnc\", \"ltc\") Done")


    # LNC LPC
    if os.path.isfile(OUTPUT_FILE_B) == True:
        print("Output file B already exists!")
    else:
        print("Computing normalization_coeff for all docs(\"Lnc\")...")
        compute_normalization_coeff("Lnc")
        print("Completed!")
        with open(OUTPUT_FILE_B, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['query_id', 'document_id']
            writer.writerow(header)
            num_of_query=0
            for queryno in query_indexed_dict.keys():
                num_of_query+=1
                print("computing doc scores for ",num_of_query,"/",len(query_indexed_dict)," query.")
                res = COMPUTE_SCORE(queryno, "Lnc", "Lpc")
                for row in res:
                    writer.writerow([queryno, row[1]])
    print("Part B(\"lnc\", \"Lpc\") Done")

    # ANC APC
    if os.path.isfile(OUTPUT_FILE_C) == True:
        print("Output file C already exists!")
    else:
        print("Computing normalization_coeff for all docs(\"anc\")...")
        compute_normalization_coeff("anc")
        print("Completed!")

        with open(OUTPUT_FILE_C, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['query_id', 'document_id']
            writer.writerow(header)
            num_of_query=0
            for queryno in query_indexed_dict.keys():
                num_of_query+=1
                print("computing doc scores for ",num_of_query,"/",len(query_indexed_dict)," query.")
                res = COMPUTE_SCORE(queryno, "anc", "apc")
                for row in res:
                    writer.writerow([queryno, row[1]])
    print("Part C(\"anc\", \"apc\") Done.")

preprocess_data()
preprocess_queries(QUERY_FILE)
main()