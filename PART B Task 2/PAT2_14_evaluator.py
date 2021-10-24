import os
import re
import sys
import csv
import math
import unicodedata
import collections
from collections import defaultdict

GOLD_STANDARD_RANKED_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
RANKED_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAT2_14_metrics_")

query_ranked_docs={}
def defaultvalue():
	return 0
query_relevant_docs = defaultdict(defaultvalue)

if os.path.exists(GOLD_STANDARD_RANKED_LIST) != True:
    print("Gold Standard file not exist!")
    print("Aborting the program...")
    sys.exit(0)

if os.path.exists(RANKED_LIST_FILE) != True:
    print("Ranked file doesn't exist!")
    print("Aborting the program...")
    sys.exit(0)

with open(RANKED_LIST_FILE, "r") as f:					#retrieved query ranked docs in dict
	csvreader = csv.reader(f)

	rowno=0
	for row in csvreader:
		rowno+=1
		if rowno==1:
			continue

		if row[0] in query_ranked_docs:
			query_ranked_docs[row[0]].append(row[1])
		else:
			query_ranked_docs[row[0]]=[]
			query_ranked_docs[row[0]].append(row[1])
	for key in query_ranked_docs.keys():
		query_ranked_docs[key]= query_ranked_docs[key][:20]

print("Completed, query and their ranked docs retrieval.")

with open(GOLD_STANDARD_RANKED_LIST, "r") as f:		   #stored gold standard data in query_relevant_docs
	csvreader = csv.reader(f)
	rowno=0
	for row in csvreader:									#(query_id,doc_id) ---> relevance_score
		rowno+=1
		if rowno==1:
			continue
		query_relevant_docs[(row[0],row[1])]= int(row[2])

print("Completed, Retrieval of Gold Standard ranked list.")
		
def Avg_Precision(query_id, k):
	
	sumAvgPrecision = 0
	relevantDocsCount = 0
	for i in range(k):
		doc_id = query_ranked_docs[query_id][i]


		if query_relevant_docs[(query_id, doc_id)]!=0:
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

def NDCG(query_id, k):
	relevanceList = []

	for i in range(k):
		doc_id = query_ranked_docs[query_id][i]
		relevanceScore =	query_relevant_docs[(query_id, doc_id)]
		relevanceList.append(relevanceScore)

	DCGValue = DCG(relevanceList)
	relevanceList.sort(reverse = True)
	idealDCGValue = DCG(relevanceList)

	if idealDCGValue==0:
		return 0
	NDCGValue = DCGValue/idealDCGValue
	return NDCGValue


def main():
	K= RANKED_LIST_FILE[-5]
	global OUTPUT_FILE
	OUTPUT_FILE	= OUTPUT_FILE+str(K)+".csv"
	with open(OUTPUT_FILE, "w") as f:					   #Opened output file
		writer = csv.writer(f)
		header = ["query_id", "AP@10", "AP@20", "NDCG@10", "NDCG@20"]
		writer.writerow(header)
		sumAP_10 = 0
		sumAP_20 = 0
		sumNDCG_10 = 0
		sumNDCG_20 = 0
		count=0
		for query_id in query_ranked_docs.keys():
			
			AP_10 = Avg_Precision(query_id, 10)
			sumAP_10 += AP_10 
			
			AP_20 = Avg_Precision(query_id, 20) 
			sumAP_20 += AP_20 

			NDCG_10 = NDCG(query_id, 10) 
			sumNDCG_10 += NDCG_10
			NDCG_20 = NDCG(query_id, 20)
			sumNDCG_20 += NDCG_20
			writer.writerow([query_id,AP_10,AP_20,NDCG_10,NDCG_20])
			count+=1
			print("Completed  ",count,"/",len(query_ranked_docs), " query." )

		sumAP_10 /= len(query_ranked_docs)
		sumAP_20 /= len(query_ranked_docs)
		sumNDCG_10 /= len(query_ranked_docs)
		sumNDCG_20 /= len(query_ranked_docs)

		writer.writerow(["Average:",sumAP_10,sumAP_20,sumNDCG_10,sumNDCG_20])
		print("Evalution Completed for", sys.argv[2] )

main()