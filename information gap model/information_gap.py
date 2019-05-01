from array import *
import math
import numpy as np
import numpy.random
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
'''
1. How to run this model?
	a. The input to this model are:
		a matrix of strings contains answears for each question
		a vector of strings contains the question
	b. The output of this model is:
		a number means the utility value
'''

'''
variable defination:
	A should be a matrix of M * N, M correspondings to M questions, N correspondings to N possible answears for Qi
	pi_A should be a matrix of M * N M correspondings to M questions, pi_A[i][j] means the probability of answear A[i][j]
	X should be a vectore of K, means K possible different prizes
	pi_X should be a vector of size K, pi_x[k] means the probability of prize X[k]
	w shoudl be a vector of size M w[i] means the weight for questions[i]
	v_A should be a matrix of size M * N is the valence of answear Ai

'''

# Helper function for get_sim()
def cos_sim(s1, s2):
	dot_product = np.dot(s1, s2)
	norm_s1 = np.linalg.norm(s1)
	norm_s2 = np.linalg.norm(s2)
	return dot_product / (norm_s1 * norm_s2)

# Caculate the similarity of two strings
def get_sim(s1, s2):
	vocab = []
	s1 = s1.split()
	s2 = s2.split()
	ss1 = {}
	ss2 = {}

	i = 0
	for key in s1:
		vocab.append(key)
		ss1[key] = i
		i += 1
	i = 0
	for key in s2:
		vocab.append(key)
		ss2[key] = i
		i += 1
	vocab_len = len(vocab)
	v1 = np.zeros(vocab_len)
	v2 = np.zeros(vocab_len)
	i = 0
	for key in vocab:
		v1[i] = ss1.get(key, 0)
		v2[i] = ss2.get(key, 0)
		i += 1
	return cos_sim(v1, v2)

# Randomly generate the prior information user have for each question
def get_prior(M):
	'''
	tmp = np.random.dirichlet(np.ones(M),size=1)
	prior = np.zeros((M, 1))
	for i in range(M):
		prior[i][0] = tmp[0][i]
	'''
	prior = np.zeros((M, 1))
	prior[0][0] = 0.1
	prior[1][0] = 0.5
	return prior


def get_conditional(M):
	'''
	tmp = np.random.dirichlet(np.ones(M),size=1)
	conditional = np.zeros((M, 1))
	for i in range(M):
		conditional[i][0] = tmp[0][i]
	'''
	conditional = np.zeros((M, 1))
	conditional[0][0] = 0.5
	conditional[1][0] = 0.7

	return conditional

def get_salience(p_prior, p_network, p_relative):
	'''
	To measure salience, it can be divide into three parts:
	1. the influence of your social networks
	2. the influence of relatives, which means the reletiveness between the questions and the answears, to start with, use cos similarity to measure this
	3. the influence of your prior knowledge
	inputs:
	p_prior: a matrix of size M * L
	p_network: a vector of size M
	p_relative: a vector of size M

	'''
	
	M = p_prior.shape[0]
	
	salience = np.zeros((M, 1))
	prior = np.zeros((M, 1))
	for i in range (M):
		prior[i] = get_entropy(p_prior[i])
	#salience = prior + p_network + p_relative
	correct = np.array([1, 0])
	
	salience = prior + p_relative + correct
	return salience



# Calculate the relativeness of each question with its answears
def get_relative(questions, answears):
	'''
	inputs:
	questions: a list of string of size M includes M questions
	answears: a matrix of strings of size M * N
	'''
	M = len(answears)
	#print(M)
	relative = []
	i = 0
	for q in questions:
		tmp_answears = answears[i]
		tmp = 0
		for an in tmp_answears:
			print(an)
			tmp += get_sim(q, an)
		relative.append(1.0 * tmp / M)
		i += 1

	p_relative = np.zeros((M, 1))
	#print("re : ", relative)
	for i in range (M):
		p_relative[i][0] = relative[i] / np.sum(relative) 
	'''
	for i in range (M):
		tmp = 0
		for j in range (N):
			tmp += get_sim(questions[i], answears[i][j])
		relativea.append(1.0 * tmp / M)
	p_relative = np.zeros((M, 1))
	for i in range (M):
		p_relative[i] = relative[i] / np.sum(relative) 
	'''
	return p_relative

def get_piX(k):
	
	piX = np.zeros((k,1))
	#print(piX.shape())
	tmp = np.random.dirichlet(np.ones(k),size=1)
	for i in range (k):
		piX[i][0] = tmp[0][i]
	#print("piX (probility of x) : ", piX)
	
	return piX 

def get_piA(M, N):
	piA = np.zeros((M, N))
	#print(piA.shape())
	for i in range(M):
		tmp = np.random.dirichlet(np.ones(N),size=1)
		#print(tmp.shape)
		for j in range (N):
			piA[i][j] = tmp[0][j]
	#print(piA.shape())
	#print("piA (probility of A) : ", piA)
	return piA


def get_vX(k):
	
	vX = np.zeros((k,1))
	tmp = np.random.normal(50, 20, k)
	for i in range (k):
		vX[i][0] = abs(tmp[i])
	#print("vX (value of prize) : ", vX)
	return vX

def get_vA(M, N):
	tmp = np.random.normal(50, 20, M * N)
	vA = np.zeros((M, N))
	#print(M, N)
	for i in range(M):
		for j in range(N):
			vA[i][j] = abs(tmp[i * N + j])
	#print("vA (answer A) : ", vA)
	return vA

def get_importance(piA):
	'''
	According to the definetion in the paper. Thus, a question is important to the extent that one’s utility 
	depends on the answer. Raising the stakes increases importance. On the other hand, if an answer is known 
	with certainty, then by this definition nothing is at stake, so the underlying question is no longer important. 
	These we can use the information entropy to meansure importance.
	'''
	M = piA.shape[0]
	importance = np.zeros((M, 1))
	for i in range (M):
		importance[i][0] = np.sum(-piA[i] * np.log(piA[i]))
	return importance

def get_network(M):
	p_network = np.zeros((M, 1))
	tmp = np.random.dirichlet(np.ones(M),size=1)
	for i in range(M):
		p_network[i][0] = tmp[0][i]
	return p_network



def get_surprise(p_prior, p_conditional):
	'''
	The model for surprise I adapted the Beyesian Surprise model introduced by paper 
	(http://ilab.usc.edu/publications/doc/Itti_Baldi06nips.pdf).
	Inputs:
	p_prior = P(M) should be a matrix of size M * L(L is the length of all prior for Qi)
	p_conditional = P(D|M) should be a matrix of size M * L
	outputs:
	S(D, M) = sum(P(M|D) * log(P(M|D) / P(M))) should be a vector of size M
	'''
	M = p_prior.shape[0]
	surprise = np.zeros((M, 1))
	for i in range(M):
		tmp = p_conditional[i] * np.log(p_conditional[i] / p_prior[i])
		surprise[i][0] = np.sum(tmp)
	return surprise


def get_weight(piA, questions, answears):
	#related with importance, salience, surprise
	#w = get_importance(M) + get_salience(M) + get_surprise(M)
	#print(piA)
	M, N = piA.shape[0], piA.shape[1]

	w = np.zeros((M, 1))
	importance =  get_importance(piA)
	p_prior = get_prior(M)
	p_conditional = get_conditional(M)
	surprise = get_surprise(p_prior, p_conditional)
	p_network = get_network(M)
	p_relative = get_relative(questions, answears)
	salience = get_salience(p_prior, p_network, p_relative)
	print("importance : ", importance)
	print("surprise : ",surprise)
	print("saliance : ", salience)
	w = importance + surprise + salience
	return w

def get_entropy(pi):
	entropy = 0
	#print(len(pi))
	#print(pi)
	for i in range(len(pi)):
		if(pi[i] == 0 or pi[i] < 0):
			continue
		entropy -= pi[i] * np.log(pi[i])
	return entropy



def utility_func():
	M = 2
	N = 4
	k = 10
	'''
	piX = get_piX(k)
	vX = get_vX(k)
	vA = get_vA(M, N)
	piA = get_piA(M, N)
	
	'''
	#piX = get_piX(k)
	piX = np.array([[0.13069837],[0.00785529],[0.07507671],[0.08857213],[0.03158515],[0.04539351],[0.28160479],[0.11404032],[0.113052  ],[0.11212173]])
	#print(piX)

	piA = np.array([[2.0/15, 2.0/15, 9.0/15, 2.0/15], [2.0/15, 6.0/15, 6.0/15, 1.0/15]])
	vA = np.array([[0.16, 0.21, 0.13, 0.5], [0.29, 0.11, 0.35, 0.25]])
	vX = np.array([[13.33220515],[18.35698654],[14.43482627],[16.4645992 ],[9.93202005],[12.50067072],[6.45649751],[5.71824374],[19.2250781 ],[7.69954935]])
	#vX = get_vX(k)
	#print(vX)
	
	questions = ["fairs in Philadelphia were held ___." ,"new markets opened in Philadelphia because ___."]
	answears = [["on the same day as market says", "as often as possible", "a couple of times a year", "whenever the government allowed it"], ["they provided more modem facilities than older markets", "the High Street Market was forced to close", "existing markets were unable to serve the growing population", "farmers wanted markets that were closer to the farms"]]

	utility = 0
	#print(piX.type())
	#print(vX)
	for i in range (k):
		utility += piX[i][0] * vX[i][0]
	#utility += (piX) * (vX)
	#print(utility)
	w = get_weight(piA, questions, answears)

	for i in range(M):

		entropy = get_entropy(piA[i])
		
		for j in range(N):
			utility += w[i][0] * ( entropy + piA[i][j] * (vA[i][j]))
	print ("utility : ", utility)

	return utility

def main():
	utility_func()


if __name__ == "__main__":
    main()
	
