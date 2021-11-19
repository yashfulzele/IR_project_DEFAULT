import os
import sys
import pickle
import math
import csv
import collections

CORPUS_SIZE = 0
DATAFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
QUERYFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
PICKLEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[3])
RANKERFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[4])
OUTPUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PB_14_important_words.csv")

query_indexed_dict = dict()
docno_indexed_dict = dict()
tfSumDoc = {}
tfMaxDoc = {}
tfDocTerm = {}

with open(PICKLEFILE, "rb") as f:
	invertedIndex = pickle.load(f)


def PREPROCESS_DATA():
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

def PREPROCESS_QUERIES():
	print("Parsing the preprocessed queries file...")
	with open(QUERYFILE, "r") as f:
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
	print("Completed preprocessing Queries!")

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

def TF(t,d,c = 'n',type='d'):	#l
	if type == 'q':
		if t in query_indexed_dict[d]:
			return 1
		else:
			return 0
	count = 0 
	if (t,d) in tfDocTerm.keys():
		count = tfDocTerm[(t,d)]
	max_count= tfMaxDoc[d]
	avg_tf= tfSumDoc[d] 
	avg_tf /= len(docno_indexed_dict[d])
	
	if count==0:
		return 0
	return (1 + math.log(count))
	

def TF_IDF(term,id_,c,type = 'd'): #computing Tfidf
	
	tf = TF(term,id_,c[0],type)

	if tf == 0:
		return 0
	idf = IDF(term,c[1])
	return tf*idf


def TFIDF_vector(top10docs):
	
	final_vector={}
	for doc in top10docs:
		doc_vector=[]
		normalization_coeff = 0
		for term in invertedIndex.keys():
			tfidf = TF_IDF(term, doc, 'ltc')
			doc_vector.append(tfidf)
			normalization_coeff += (tfidf**2)
		normalization_coeff = math.sqrt(normalization_coeff)

		termno=0
		for term in invertedIndex.keys():

			if term not in final_vector:
				final_vector[term] = 0
			final_vector[term] += (doc_vector[termno]/normalization_coeff)
			termno+=1
	
	top5words= sorted(final_vector, key=final_vector.get, reverse=True)[:5]
	print(top5words)
	return top5words

def evaluate():

	with open(OUTPUTFILE, "w") as f:
		writer = csv.writer(f)

		
		for query_id in query_indexed_dict.keys():
			print(query_id)
			relevant_docs = set()
			non_relevant_docs = set()
			
			rel_count=10
			for doc in queryRetrievedDocs[query_id]:
				if rel_count<=0:
					break
				relevant_docs.add(doc)
				rel_count-=1
			top5words = TFIDF_vector(relevant_docs)
			writer.writerow([query_id,top5words[0]+','+top5words[1]+','+top5words[2]+','+top5words[3]+','+top5words[4]])


PREPROCESS_DATA()
PREPROCESS_QUERIES()
print("\npreprocessing Completed!")

queryRetrievedDocs = {}
with open(RANKERFILE, "r") as f:
	csvreader = csv.reader(f)

	rowno=0
	for query_no, docID in csvreader:
		if rowno:
			query_no = int(query_no)
			if query_no not in queryRetrievedDocs:
				queryRetrievedDocs[query_no] = []
			queryRetrievedDocs[query_no].append(docID)
		rowno+=1
evaluate()