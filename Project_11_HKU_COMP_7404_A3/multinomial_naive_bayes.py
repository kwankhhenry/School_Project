import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        #### START CODE HERE ####
        # initialization of the prior and likelihood variables
        prior = np.array([[ 0.0 for i in range(n_classes)]]).T
        likelihood = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])

        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables named "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature.

        print ("x_count = %d, word_count = %d, num_classes = %d, y_count=%d" % (
            n_docs, n_words, n_classes, y.shape[0]))

        # compute prior = prob( class )
        prior[0] = (y == 0).sum()/float(n_docs)                         # negative class (y=0) = positive review
        prior[1] = (y == 1).sum()/float(n_docs)                         # positive class (y=1) = negative review

        # compute likelihood = prob( word | class )
        # by considering all documents and all words in each document (≈ 7 lines)
        frequency = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])
        for i in range(n_docs):
            if y[i] == 0:
                frequency[0] += x[i]
            elif y[i] == 1:
                frequency[1] += x[i]

        for j in range(n_classes):
            likelihood[j] = (frequency[j] + 1) / (np.sum(frequency[j]) + n_words + 1)

        likelihood = likelihood.T
        
        #### END CODE HERE ####

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
