README

The data provided in a7a.train and a7a.test are from the “Census Income” Dataset. The objective of the assignment was to predict whether the income of each tuple exceeded $50k/yr. The first column of the data was the class label which took the value of 1 or 0; as the sigmoid function was used for this assignment. For more information, please visit: http://archive.ics.uci.edu/ml/datasets/Adult

Each tuple has a total of 123 features. A neural network was implemented with backpropagation that would capture the change in error with respect to the weight at each layer, and was minimized using stochastic gradient descent. 

REQUIREMENTS:

Please have the numpy library installed and imported to run this program. The use of arrays is essential to complete matrix multiplication and array summations. Data for this is stored in /u/cs246/data/adult/ on the csug machines. 

FUNCTIONS:

The first is the parse_line which will parse each line in the data and add another value into the list to incorporate the bias. Thus, each tuple will have a total of 124 values. 

The second is parse_data which will return an array of the class labels of each tuple in the data and second, an array of the actual tuples within arrays. 

The third function is called init_model which will take in the weight files provided to us in w1.txt and w2.txt and initialize weights prior to feeding it into the neural network for training. 

The third function called forwardprop goes through the two layers, the hidden layer and the output layer, and returns the yhat prediction at the output layer. Given that this neural network has only one output node, this is a binary classification which will return the prediction of whether the tuple is part of one class or the other. The forwrdprop takes in the model weights and determines the activation at each layer using the sigmoid function and provides the respective “z.” It then returns the yhat. 

The train_model function takes in the initialized model and training data, and runs through a given iteration, updating the weights using stochastic gradient descent. It calls the forwardprop function at every iteration to have the yhat for each tuple and to calculate the respective deltas. Upon the completion of training, it returns the weights of the model. 

The sigmoid function takes in every point at each layer and “activates” it by implementing the sigmoid activation function. This activation results in a probability between 0 and 1. 

The predict function takes the corresponding model returned from the train_model function and compares the actual class of the test tuple to the prediction. If the prediction is correct with comparison to the actual class label, it will return a 1. 

The test_accuracy function iterates through each tuple in the test data set and calls the predict function, and records the outputs into a list. The accuracy is then calculated by summing all of the correct predictions over the length of the total points in the list. 

The extract_weights function returns the model’s weight 1 and weight 2, if the argument —print weights is provided. 

TESTS:

The data includes a development set, a training set, and a testing set. To run the algorithm without the development set using the weights given, please run as follows: ./Yakubov_backprop.py —nodev —iterations ? —weights_files w1.txt w2.txt —lr ? —train_file. Replace the ? questions marks with the iterations and learning rate desired. 

There is a commented section in the code that imports matplotlib to graph the development data and training data with respect to accuracy, to measure if there is any overfitting. More on this is provided in the discussion. If this code is uncommented, and is run with the development data as an argument (by suppressing nodev), you would get the graph plotted. 

DISCUSSION: 

Neural Networks can be used with any given amount of hidden layers to learn a specific function. Neural Networks are very good at learning a specific function given enough data and enough hidden layers. As a result, it is important to figure out if any overfitting occurs. To do this, my implementation used matplotlib to graph the accuracies with respect to the dev data and training data. 

The first plot shows the Accuracy vs. Iterations given a learning rate of .05, and using the weight files. As can be observed, the training data is overfitting. The development data seems to level off at some accuracy point, while the training data accuracy continues to grow at the number of iterations grows. This was run on 200 iterations to get a closer view of the overfitting that was occurring. Please see this in image 2. I used the same learning rate, and decided to cut off the number of iterations at 50. As you can see, this seems like a good numbers of iterations for the training data, as anything more than this tends to start the overfitting trend. This is seen in the third graph. 

In summation, while neural networks can be used to learn any continuous function and/or classification problem, it is important to note that they are prone to overfitting, as the amount of hidden layers increases, and the amount of iterations increases. The exact error of the training data is learned and the patterns of the data become well-known. This can be controlled for by using the validation set to determine the cutoff iteration for the model learned and whether it is a strong one. 





   