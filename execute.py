from conllu import parse
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import manhattan_distances
import re, sys, numpy 


reload(sys)
sys.setdefaultencoding('utf8')

file1 = "dataset/en_ewt-ud-train.conllu"
file2 = "dataset/en_ewt-ud-test.conllu"

data1 = open(file1, 'r').read()
data1 = re.sub(r" +", r"\t", data1)

data2 = open(file2, 'r').read()
data2 = re.sub(r" +", r"\t", data2)


# The dictionary is 2D list of dictionaries, list of sentences *  list of dictionaries belonging to the words in sentences  
data_train = parse(data1)
data_test = parse(data2)

# Parsing
def left_arc(x, stack, c):
	if len(stack) < 3:
		return 0
	e1 = stack[-1]
	e2 = stack[-2]
	c = 0
	if x[e2[0]-1]['head'] == e1[0]:
		for y in range(len(x)):
			if x[y]['head'] == e2[0]:
				c += 1
		if c == 0:
			return 1				
	return 0
	
def right_arc(x, stack, c):
	if c == 0 and len(stack) == 2:
                return 1
	if len(stack) < 3:
		return 0
		
	e1 = stack[-1]
        e2 = stack[-2]
        c = 0
        if x[e1[0]-1]['head'] == e2[0]:
                for y in range(len(x)):
                        if x[y]['head'] == e1[0]:
                                c += 1
        	if c == 0:
                	return 1
	return 0


def dependency_parsing(x):
	stack = [[0,'ROOT']]; buffer_ = []; op = []; deps = [] 
	for y in x:
		if y['id'] == None or type(y['id']) != int:
			y['id'] = 0
		buffer_.append([y['id'], y['lemma']])
		#print y
	while(len(stack) + len(buffer_) > 1):
	
		if left_arc(x, stack, len(buffer_)):
			var = stack.pop(-2)
			op.append('LEFT-ARC')
			x[var[0]-1]['head'] = -1	 
			deps.append([stack[-1][1], x[stack[-1][0]-1]['xpostag'], x[var[0]-1]['deprel'], var[1], x[var[0]-1]['xpostag']])
		
		elif right_arc(x, stack, len(buffer_)):
			var = stack.pop()
                        op.append('RIGHT-ARC')    
			x[var[0]-1]['head'] = -1
                        deps.append([stack[-1][1], x[stack[-1][0]-1]['xpostag'],x[var[0]-1]['deprel'], var[1], x[var[0]-1]['xpostag']])
		else:
			if len(buffer_) == 0:
				break
			stack.append(buffer_.pop(0))
                        op.append('SHIFT')
			#print stack[-2][0], stack[-1][0] 
			deps.append([stack[-2][1], x[stack[-2][0]-1]['xpostag'], 'SHIFT', stack[-1][1], x[stack[-1][0]-1]['xpostag']])
	return deps, op



# Wordvectors
def wordvec(data):
	corpus = []; pos = {}; out = {}; count1 = 0
	for x in data:
		sentence = []
		for y in x:
			sentence.append(y['lemma'])
			if y['xpostag'] not in pos:
				pos[y['xpostag']] = count1
				count1 += 1
		corpus.append(sentence)
	
	vec1 = numpy.zeros(count1)
	count1 = 0
	for x in pos:
		pos[x] = vec1
		pos[x][count1] = 1
		count1 += 1 
	
	out['SHIFT'] = 1
	out['LEFT-ARC'] = 2
	out['RIGHT-ARC'] = 3
	
	wordvectors = Word2Vec(corpus, size = 40, min_count = 1)
	return wordvectors, pos, out




# Config FeatureVectors
# concatenated {Top of the stack and start of buffer (word lemma word2vec vector and pos tag one hot vector)}

def feature_vector(sentence, vectors, pos, out):
	dependency , operations = dependency_parsing(sentence)
	vector_ = []
	output_ = []
	for x in range(len(operations)):
		vector1 = [] ; output1 = []
		y = dependency[x]
		if y[0] == 'ROOT':
			vector1.extend(numpy.zeros(40))
		else:
			vector1.extend(vectors.wv[y[0]])
		vector1.extend(pos[y[1]])
		vector1.extend(vectors.wv[y[3]])
		vector1.extend(pos[y[4]])
		output1 = out[operations[x]]
	
		vector_.extend([vector1])
		output_.extend([output1])
	return vector_, output_



# Prediction
def classifier(feature, tag):
	length = int(0.9*len(feature))
	train_input = feature[:length]	
	train_output = tag[:length]
	test_input = feature[length:]
	test_output = tag[length:]
	
	nn_classifier = MLPClassifier().fit(train_input, train_output)
	prediction = nn_classifier.predict(test_input)
	accuracy = 0
	for x in range(len(test_output)):
		if prediction[x] == test_output[x]:
			accuracy += 1
	print 'Accuracy of The MLPClassifier is :' ,float(accuracy)/len(test_output)
	

# Program
data_train.extend(data_test)
vectors, pos , out= wordvec(data_train)
feature = []
tag = []
for sentence in data_train:
	id1, id2 = feature_vector(sentence, vectors, pos, out)
	feature.extend(numpy.array(id1))
	tag.extend(numpy.array(id2))

classifier(numpy.array(feature), numpy.array(tag))
