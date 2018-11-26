from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def run_nn(train_X, train_y, test_X, test_y):

        #### START CODE HERE ####
        # Initialize the MLPClassifier with the parameters 
        # specified in the assignment description (≈ 1 line)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1) 
        # Create a model by fitting the training data to the classifier (≈ 1 line)
        # y needs to be flattened before training by using the ravel() function  
        clf.fit(train_X, train_y.ravel())


        # Predict on new testing data (≈ 1 line)
        pred_y = clf.predict(test_X)
        
        ### END CODE HERE
        
        print("Accuracy on testing data = " + str(clf.score(test_X, test_y)))
        print("Confusion Matrix:")
        print(confusion_matrix(test_y,pred_y))
        print(classification_report(test_y,pred_y))

        return 
