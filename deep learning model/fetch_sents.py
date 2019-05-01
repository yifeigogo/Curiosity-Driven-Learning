import sys,os,math,re
import json
import urllib.parse
import requests
from bs4 import BeautifulSoup, Comment, SoupStrainer
import nltk
import time
#nltk.download()


#... usage: fetchRecords("Earthquake", ["because"], 1)
#... the Google Search API only picks up 
sentlist = []
regmatches = []
totalSents = 20

# protects against invalid unicode characters
def rectify(selstr):
	val = ''.join([x for x in selstr if ord(x)<128])
	return str(val)


senthandle = open("sentence_results.txt", "w+")
# modified: startIndex error. MUST start at 1 at minimum. resolved
def fetchRecords(userquery, necessary_words, startIndex=1):
	global senthandle, sentlist, regmatches, totalSents
	queryterms = [userquery]
	queryterms = [userquery] + [necessary_words]
	queryterms = [x.lower() for x in queryterms]
	userquery = " ".join(queryterms)
	#userquery = userquery + " " + " ".join(necessary_words)
	liveMode = True
	if startIndex%10==0:
		startIndex = startIndex + 1 # API thing

	if liveMode==True:
		apikey = "AIzaSyAGw6MwyZe-H68Kt3vIeOk9T3620g0aRVY"
		##apikey = "AIzaSyDiuRRyDbrCXqsy02yE4Vhzp7ny7FG52m4"
		#apikey = "AIzaSyD13KEY4zNRONAR4vR0eL0YmqvT1eI7Cv4" # this is my key
		# fetch the data from the API
		selquery = chr(34) + urllib.parse.quote_plus(userquery) + chr(34)
		# &searchType=image
		selurl = "https://www.googleapis.com/customsearch/v1?key="+apikey+"&cx=007865715089351041060:znf9injp6vi&q="+selquery+"&num=10&start="+str(startIndex)
		#selurl = "https://www.googleapis.com/customsearch/v1?key="+apikey+"&cx=000460833272819085236:zptf8kyliuc&q="+selquery+"&num=10&start="+str(startIndex)

		# we now have the JSON results
		ch = requests.get(selurl)
		res = rectify(ch.text)
		jstr = res
		jstruct = json.loads(res)
		
		# store the raw results
		#handle = open("apires_folder/apires_surprise.json", "w")
		#handle.write(res)
		#handle.close()
		
	
	allresults = []
	userquery_lower = userquery.lower().strip()

	# begin processing
	print ("BEGIN:"+str(startIndex)+":"+userquery)
	if not "items" in jstruct:
		return
	for i in range(0,len(jstruct["items"])):
		linkRef = jstruct["items"][i]["link"]
		linkTitle = jstruct["items"][i]["title"]
		linkSnippet = jstruct["items"][i]["snippet"]
		linkSize = 1
		allresults.append({"title":linkTitle, "link":linkRef, "snippet":linkSnippet, "size":linkSize})
		#print (linkTitle,":::", linkRef,":::",str(len(sentlist)))
		
		if len(sentlist)>=totalSents:
			break
		if 5<6:
		#try:
			ch = requests.get(linkRef)
			res = rectify(ch.text).lower()
			soup2 = BeautifulSoup("<suprema>"+res+"</suprema>", "html.parser")
			for s in soup2.findAll("script"):
				s.extract()
			for s in soup2.findAll("style"):
				s.extract()
			for s in soup2(text=lambda text:isinstance(text,Comment)):
				s.extract()

			res = soup2.findAll(["p","h1","h2","h3","body","section"])
			res = " ".join([". "+x.text for x in res])
			res = re.sub(r"\n+", ". ", res)
			res = re.sub(r" {2,}", ". ", res)
			res = re.sub(r"[^A-Za-z0-9\-\.\,\:\; ]+", "",res)
			#res = soup2.texts
			allsents = nltk.sent_tokenize(res)
			#print (allsents)
			#break
			tempregs = []
			for p in allsents:
				tempregs.append(re.sub(r"[^A-Za-z0-9 ]+","",p))
			#print (tempregs)
			#for p in range(0,len(allsents)):
			#	print (p)
			#	print (allsents[p])
			#sys.exit()
			termset = userquery
			for q in range(0,len(allsents)):
				cursent = allsents[q]
				isvalid = True
				for ele in queryterms:
					if not ele in tempregs[q]:
						isvalid = False
						break
				if not isvalid:
					continue

				if tempregs[q] not in regmatches:
					if len(cursent)>(len(userquery_lower)+5) and len(cursent)<200:
						sentlist.append(cursent)
						regmatches.append(tempregs[q])
						senthandle.write(cursent+"\n")
						print ("FOUND: ", tempregs[q], ": ", queryterms)
			if len(sentlist)>=totalSents:
				break
			#handle.write(res)
			#handle.close()
			time.sleep(1)
		#except:
		#	print ("Couldn't handle: ", linkRef)
		#	continue
		#break
	#handle.close()
	return allresults

def main():
	topics = ["Wikipedia"]
	keywords = ["CNN", "RNN", "LSTM", "machine learning"]
	#keywords = ["type", "types", "classified", "is", "use", "used", "apply", "are"]

	for word in topics:
		for k_w in keywords:
			print (fetchRecords(word, k_w, 1))

if __name__== "__main__":
    main()

#print (fetchRecords("Earthquake", ["because"], 1))