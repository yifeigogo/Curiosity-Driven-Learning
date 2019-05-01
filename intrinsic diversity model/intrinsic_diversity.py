from array import *
import math
import numpy as np
import numpy.random
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def greedy(Q, answers, index, piA, vA):
	'''
	The main section for the greedy algorithm part.
	Parameters:
	delta: a small value to garantee converge
	Input: 
	Whole question set Q
	Output:
	Result question set Qr 
	'''
	delta = 0
	Qr = set()
	initial_q = ""
	initial_utility = 0
	cur_utility = 0
	i = 0
	for q in Q:
		q_u = utility(Qr, q, answers, index, piA, vA)
		if (q_u > initial_utility):
			initial_q = q
			initial_utility = q_u
	Qr.add(initial_q)
	Q.remove(initial_q)
	cur_utility = initial_utility
	utility_gain = 0
	next_q = ""
	FLAG = True
	while(utility_gain > delta or FLAG):
		FLAG = False
		if(next_q in Q):
			Qr.add(next_q)
			Q.remove(next_q)
		cur_utility += utility_gain
		utility_gain = 0
		for q in Q:
			q_u = utility(Qr, q, answers, index, piA, vA)
			if(q_u - cur_utility > utility_gain):
				utility_gain = q_u - cur_utility
				next_q = q
	return Qr

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

def get_snip(q1, q2):
	return 10
# Randomly generate the prior information user have for each question
def get_prior(M):
	
	tmp = np.random.dirichlet(np.ones(M),size=1)
	print("prior: ", tmp)
	prior = np.zeros((M, 1))
	for i in range(M):
		prior[i][0] = tmp[0][i]
	
	return prior


def get_conditional(M):
	
	tmp = np.random.dirichlet(np.ones(M),size=1)
	conditional = np.zeros((M, 1))
	print("conditional: ", tmp)
	for i in range(M):
		conditional[i][0] = tmp[0][i]
	
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
def get_relative(questions, answers):
	'''
	inputs:
	questions: a list of string of size M includes M questions
	answears: a matrix of strings of size M * N
	'''
	M = answers.shape[1]
	#print(M)
	relative = []
	i = 0
	for q in questions:
		tmp_answers = answers[i]
		tmp = 0
		for an in tmp_answers:
			print(an)
			tmp += get_sim(q, an)
		relative.append(1.0 * tmp / M)
		i += 1

	p_relative = np.zeros((M, 1))
	print("re: ", relative)
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


def get_weight(piA, questions, answers):
	#related with importance, salience, surprise
	#w = get_importance(M) + get_salience(M) + get_surprise(M)
	#print(piA)
	M, N = answers.shape[0], answers.shape[1]

	w = np.zeros((M, 1))
	importance =  get_importance(piA)
	p_prior = get_prior(M)
	p_conditional = get_conditional(M)
	surprise = get_surprise(p_prior, p_conditional)
	p_network = get_network(M)
	p_relative = get_relative(questions, answers)
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


def information_gain(Qr, answers, piA, vA):
	utility = 0
	#print(piX.type())
	#print(vX)
	M, N = answers.shape[0], answers.shape[1]
	piX = np.array([[0.13069837],[0.00785529],[0.07507671],[0.08857213],[0.03158515],[0.04539351],[0.28160479],[0.11404032],[0.113052  ],[0.11212173]])
	#print(piX)

	#piA = np.array([[0.33, 0.33, 0.33], [0.25, 0.35, 0.4], [0.27, 0.5, 0.23], [0.14, 0.52, 0.34]])
	#vA = np.array([[0.16, 0.13, 0.71], [0.29, 0.46, 0.25], [0.05, 0.38, 0.57], [0.19, 0.46, 0.35]])
	vX = np.array([[13.33220515],[18.35698654],[14.43482627],[16.4645992 ],[9.93202005],[12.50067072],[6.45649751],[5.71824374],[19.2250781 ],[7.69954935]])
	for i in range (10):
		utility += piX[i][0] * vX[i][0]
	#utility += (piX) * (vX)
	#print(utility)
	w = get_weight(piA, Qr, answers)

	for i in range(M):

		entropy = get_entropy(piA[i])
		
		for j in range(N):
			utility += w[i][0] * ( entropy + piA[i][j] * (vA[i][j]))
	return utility
	#print ("utility : ", utility)

def utility(Qr, q, answers, index, piA, vA):
	'''
	parameters:
	lamda: balance the trade off between similarity and utility gain
	belta: control the rate at which returns diminish from additional coverage
	Input:
	A set of questions
	Output:
	A number represent the utility value
	'''
	lamda = 0.1
	belta = 0.05
	#Qr = {"How many of my students liked my teaching?", "Did they applaud on the last day of class?", "How good a teacher am I?", "Will I get tenure?"}
	#Qr_a = ["How many of my students liked my teaching?", "Did they applaud on the last day of class?", "How good a teacher am I?", "Will I get tenure?"]
	#answers = [["perhaps 10", "perhaps 20", "perhaps 50"], ["Nearly all of them", "Almost None of them", "Some of them"], ["good", "bad", "just so so"],  ["absolutely yes", "I am afraid not", "I don't know"]]
	u = 0
	for qr in Qr:
		u -= lamda * get_sim(qr, q)
	
	Qr.add(q)

	ans = [answers[index[q]]]
	piA_ = [piA[index[q]]]
	vA_ = [vA[index[q]]]
	for q in Qr:
		print(ans)
		print(index[q])
		ans = np.row_stack(answers[index[q]])
		piA_= np.row_stack(piA[index[q]])
		vA_ = np.row_stack(vA[index[q]])

	u += information_gain(Qr, ans, piA_, vA_)
	Qr.remove(q)


	u = math.exp(belta * u)
	return u




def main():
	Q = {"How many of my students liked my teaching?", "Did they applaud on the last day of class?", "How good a teacher am I?", "Will I get tenure?"}
	index = {"How many of my students liked my teaching?":0, "Did they applaud on the last day of class?":1, "How good a teacher am I?":2, "Will I get tenure?":3}
	answers = [["perhaps 10", "perhaps 20", "perhaps 50"], ["Nearly all of them", "Almost None of them", "Some of them"], ["good", "bad", "just so so"],  ["absolutely yes", "I am afraid not", "I don't know"]]
	piA = np.array([[0.33, 0.33, 0.33], [0.25, 0.35, 0.4], [0.27, 0.5, 0.23], [0.14, 0.52, 0.34]])
	vA = np.array([[0.16, 0.13, 0.71], [0.29, 0.46, 0.25], [0.05, 0.38, 0.57], [0.19, 0.46, 0.35]])

	Qr = greedy(Q, answers, index, piA, vA)
	for q in Qr:
		print(q)
	return Qr

if __name__ == "__main__":
    main()
	

