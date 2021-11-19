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
GOLDFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[4])
RANKERFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[5])
OUTPUTFILE_RF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PB_14_rocchio_RF_metrics.csv")
OUTPUTFILE_PsRF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PB_14_rocchio_PsRF_metrics.csv")

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

normalization_coeff={}
doc_vector = {}

def COMPUTE_NORMALIZATION_COEFF(c1):
	print("preprocessing normalization_coeff for all docs:", end =" ", flush=True)  
	count=0
	for docno in docno_indexed_dict.keys():
		if count%10000 == 0:
			print("+", end ="", flush=True)
		count+=1
		normalization_coeff[docno] = 0
		doc_vector[docno]=[]
		for term in docno_indexed_dict[docno]:
			tfidf = TF_IDF(term, docno, c1)
			doc_vector[docno].append(tfidf)
			normalization_coeff[docno] += (tfidf**2)
		normalization_coeff[docno] = math.sqrt(normalization_coeff[docno])

def COMPUTE_QUERY_VECTOR(query, c):   
	
	TFIDF_query = {}
	norm_coefficient = 0 
	for t in query_indexed_dict[query]:
		tfidf = TF_IDF(t,query,c,'q')
		TFIDF_query[t]=(tfidf)
		norm_coefficient += (tfidf**2)
		
	norm_coefficient = math.sqrt(norm_coefficient)
	if norm_coefficient!=0:
		for term in TFIDF_query.keys():
			TFIDF_query[term] /= norm_coefficient

	return TFIDF_query

def ROCCHIO(query_id, relevant_docs, non_relevant_docs, alpha, beta, gamma):
	queryVector = COMPUTE_QUERY_VECTOR(query_id, 'lnc')
	index = 0

	for term in queryVector.keys():
		queryVector[term] *= alpha

	for doc in relevant_docs:
		for term,tfidf in zip(docno_indexed_dict[doc], doc_vector[doc]):

			tfidf = tfidf/normalization_coeff[doc]
			if term not in queryVector: queryVector[term] = 0
			queryVector[term] += ((beta*tfidf)/len(relevant_docs))
	for doc in non_relevant_docs:
		for term,tfidf in zip(docno_indexed_dict[doc], doc_vector[doc]):
			tfidf = tfidf/normalization_coeff[doc]
			if term not in queryVector: queryVector[term] = 0
			queryVector[term] -= ((gamma*tfidf)/len(non_relevant_docs))
	return queryVector



def COMPUTE_SCORE(query_vector1, query_vector2, query_vector3):   
	# num_of_doc = 0
	DocScores1 = []
	DocScores2 = []
	DocScores3 = []


	for docno in doc_vector.keys():
		dqScore1 = 0
		dqScore2 = 0
		dqScore3 = 0
		for term, tfidf in zip(docno_indexed_dict[docno],doc_vector[docno]):
			tfidf = tfidf/normalization_coeff[docno]
			if term in query_vector1:
				dqScore1 += query_vector1[term]*tfidf
			if term in query_vector2:
				dqScore2 += query_vector2[term]*tfidf
			if term in query_vector3:
				dqScore3 += query_vector3[term]*tfidf

		DocScores1.append((dqScore1,docno))
		DocScores2.append((dqScore2,docno))
		DocScores3.append((dqScore3,docno))


	DocScores1.sort(reverse = True)
	DocScores2.sort(reverse = True)
	DocScores3.sort(reverse = True)
	top20docs1 = DocScores1[:20]
	top20docs2 = DocScores2[:20]
	top20docs3 = DocScores3[:20]
	return (top20docs1, top20docs2, top20docs3)

def Avg_Precision(query_id, top20docs, k):
	
	sumAvgPrecision = 0
	relevantDocsCount = 0
	for i in range(k):
		doc_id = top20docs[i][1]
		if (query_id, doc_id) in golden_relevant_docs:
			relevantDocsCount += 1
			sumAvgPrecision += relevantDocsCount/(i + 1)
	if relevantDocsCount==0:
		return 0
	meanAvgPrecision = sumAvgPrecision/relevantDocsCount
	return meanAvgPrecision


def DCG(relevanceList):
	score = relevanceList[0]
	for i in range(1, len(relevanceList)):
		score += relevanceList[i]/math.log(i + 1,2)
	return score

def NDCG(query_id, top20docs, k):
	relevanceList = []

	for i in range(k):
		doc_id = top20docs[i][1]
		relevanceScore = 0
		if (query_id, doc_id) in golden_relevant_docs: 
			relevanceScore = golden_relevant_docs[(query_id, doc_id)]
		relevanceList.append(relevanceScore)

	DCGValue = DCG(relevanceList)
	relevanceList.sort(reverse = True)
	idealDCGValue = DCG(relevanceList)

	if idealDCGValue==0:
		return 0
	NDCGValue = DCGValue/idealDCGValue
	return NDCGValue


def evaluate(scheme):
	OUTPUTFILE = OUTPUTFILE_RF
	if scheme=='PsRF':
		OUTPUTFILE = OUTPUTFILE_PsRF

	with open(OUTPUTFILE, "w") as f:
		writer = csv.writer(f)
		header = ['alpha', 'beta', 'gamma', 'mAP@20', 'NDCG@20']
		writer.writerow(header)
		sumAP_20_1 = 0
		sumNDCG_20_1 = 0
		sumAP_20_2 = 0
		sumNDCG_20_2 = 0
		sumAP_20_3 = 0
		sumNDCG_20_3 = 0
		
		print("Evaluating for", scheme, "Scheme:",end=" ", flush=True)
		for query_id in query_indexed_dict.keys():
			print("+", end="", flush=True)
			relevant_docs = set()
			non_relevant_docs = set()
			
			if scheme == 'RF':
				rel_count=20
				for doc in queryRetrievedDocs[query_id]:
					if rel_count<=0:
						break
					if (query_id,doc) in golden_relevant_docs.keys() and golden_relevant_docs[(query_id,doc)]==2:
						relevant_docs.add(doc)
					else:
						non_relevant_docs.add(doc)
					rel_count-=1
			else:
				rel_count=10
				for doc in queryRetrievedDocs[query_id]:
					if rel_count<=0:
						break
					relevant_docs.add(doc)
					rel_count-=1

			alpha,beta,gamma = 1,1,0.5
			queryVector1 = ROCCHIO(query_id, relevant_docs, non_relevant_docs, alpha, beta, gamma)
			alpha,beta,gamma = 0.5,0.5,0.5
			queryVector2 = ROCCHIO(query_id, relevant_docs, non_relevant_docs, alpha, beta, gamma)
			alpha,beta,gamma = 1,0.5,0
			queryVector3 = ROCCHIO(query_id, relevant_docs, non_relevant_docs, alpha, beta, gamma)
			top20docs = COMPUTE_SCORE(queryVector1, queryVector2, queryVector3)

			sumAP_20_1 += Avg_Precision(query_id, top20docs[0], 20)
			sumAP_20_2 += Avg_Precision(query_id, top20docs[1], 20)
			sumAP_20_3 += Avg_Precision(query_id, top20docs[2], 20)

			sumNDCG_20_1 += NDCG(query_id, top20docs[0], 20)
			sumNDCG_20_2 += NDCG(query_id, top20docs[1], 20)
			sumNDCG_20_3 += NDCG(query_id, top20docs[2], 20)
		print("")
		sumAP_20_1 /= len(query_indexed_dict)
		sumAP_20_2 /= len(query_indexed_dict)
		sumAP_20_3 /= len(query_indexed_dict)

		sumNDCG_20_1 /= len(query_indexed_dict)
		sumNDCG_20_2 /= len(query_indexed_dict)
		sumNDCG_20_3 /= len(query_indexed_dict)


		writer.writerow([1,1,0.5,sumAP_20_1,sumNDCG_20_1])
		writer.writerow([0.5,0.5,0.5,sumAP_20_2,sumNDCG_20_2])
		writer.writerow([1,0.5,0,sumAP_20_3,sumNDCG_20_3])


PREPROCESS_DATA()
PREPROCESS_QUERIES()
COMPUTE_NORMALIZATION_COEFF('ltc')
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


golden_relevant_docs = {}
with open(GOLDFILE, "r") as f:
	csvreader = csv.reader(f)

	rowno=0
	for row in csvreader:                    #(query,doc)--> score
		if rowno:
			golden_relevant_docs[(int(row[0]),row[1])]=int(row[2])
		rowno+=1

evaluate('RF')
evaluate('PsRF')