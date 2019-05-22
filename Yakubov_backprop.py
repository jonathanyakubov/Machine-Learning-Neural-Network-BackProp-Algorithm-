#!/usr/local/bin/python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/Users/jonathanyakubov/Desktop/MachineLearning/MLhw3/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.


    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    model = (w1,w2)
    # raise NotImplementedError #TODO: delete this once you implement this function
    return model

def forwardprop(model,train_ys,train_xs):    #for one value
	w1,w2=model
	prediction=np.matmul(w1,train_xs)
	act_1=sigmoid(prediction)
	# print(act_1)
# 	print(act_1.shape)
	a=np.array([[1]])
	act1=np.concatenate((act_1,a))
	# print(act1)
# 	print(act1.shape)
	prediction2=np.matmul(w2,act1)
	act2=sigmoid(prediction2)
	# print(act2)
# 	print(act2.shape)
	
	return act2   #last prediction

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    #TODO: Implement training for the given model, respecting args
	#forward propagation
	# from matplotlib import pyplot as plt
# 	iterations=[]
# 	accuracies=[]
# 	dev_accuracies=[]
	w1,w2=model
	for n in range(args.iterations):
		# iterations.append(n+1)
		for i in range(len(train_xs)):
			yhat=forwardprop(model,train_ys[i],train_xs[i])
			# print("yhat is", yhat)
			prediction=np.matmul(w1,train_xs[i])  #get the previous layers activation function
			# print(prediction.shape)
			# print("Prediction after weight 1 is", prediction)
			act1=sigmoid(prediction)
			# print("Activation after weight 1 is",act1)
# 			print(act1.shape)
			error=(yhat-train_ys[i])**2
			# print(yhat)
			# print(train_ys[i])
# 			print("The error after yhat is", error)
			# print(error.shape)
			a=np.array([[1]])
			act_1=np.concatenate((act1,a))
			delta2=(yhat-train_ys[i])*yhat*(1-yhat)
			dEdW2=delta2*act_1
			delta1=w2.T[:-1]*delta2*act1*(1-act1)
			dEdW1=np.matmul(delta1,train_xs[i].T)
			# dEdW2=act_1*(yhat-train_ys[i])*yhat*(1-yhat)
			# print(dEdW2)
# 			print(dEdW2.shape)
			# dEdW1=(yhat-train_ys[i])*yhat*(1-yhat)*np.matmul(act1*(1-act1),train_xs[i].T)
			# print(dEdW1)
			w1-=(args.lr*dEdW1)
			w2-=(args.lr*dEdW2.T) # i might need to transpose this
			model=(w1,w2)
			
	# 	if not args.nodev:
# 			dev_accuracy=test_accuracy(model,dev_ys,dev_xs)
# 			dev_accuracies.append(dev_accuracy)
# 			acc_it=test_accuracy(model, train_ys, train_xs)
# 			accuracies.append(acc_it)
# 		
# 	if not args.nodev:
# 		plt.plot(iterations,accuracies, label="Training Data")
# 		plt.plot(iterations,dev_accuracies, label="Dev Data")
# 		plt.xlabel("Iterations")
# 		plt.ylabel("Accuracy")
# 		plt.title("Accuracy vs. Iterations")
# 		plt.legend()
# 		plt.show()
	model=(w1,w2)
# 	print(w1)
# 	raise NotImplementedError #TODO: delete this once you implement this function
	return model

def sigmoid(point):
	activation=(np.exp(point)/(1+np.exp(point)))
	
	return activation 

def predict(model,test_ys,test_xs):  #prediction using backprop on the testing set 
	w1,w2=model   #using the weights after backprop
	s=0
	if float(forwardprop(model,test_ys,test_xs)[0][0])>=0.5:
		if test_ys==1:
			s+=1
	elif float(forwardprop(model,test_ys,test_xs)[0][0])<0.5:
		if test_ys==0:
			s+=1
	return s


def test_accuracy(model, test_ys, test_xs):
	total=[]
	for i in range(len(test_xs)):
		output=predict(model,test_ys[i],test_xs[i])
		total.append(output)
	accuracy=sum(total)/len(total)
	return accuracy
    	
    #TODO: Implement accuracy computation of given model on the test data
   #  raise NotImplementedError #TODO: delete this once you implement this function


def extract_weights(model):
	w1,w2=model
	return (w1,w2)
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
   #raise NotImplementedError #TODO: delete this once you implement this function
   

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
