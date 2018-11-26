from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def run_svm(train_X, train_y, test_X, test_y):

        #### START CODE HERE ####
        # Initialize the SVM classifier (SVC) with the parameters 
        # specified in the assignment description (≈ 1 line)

        svclassifier = SVC(kernel='linear')

        # Create a model by fitting the training data to the classifier (≈ 1 line)
        # y needs to be flattened before training by using the ravel() function 
        svclassifier.fit(train_X, train_y.ravel())

        # Predict on new testing data (≈ 1 line)
        pred_y = svclassifier.predict(test_X)

        ### END CODE HERE

        print("Accuracy on testing data = " + str(svclassifier.score(test_X, test_y)))
        print("Confusion Matrix:")
        print(confusion_matrix(test_y,pred_y))
        print(classification_report(test_y,pred_y))

        return

