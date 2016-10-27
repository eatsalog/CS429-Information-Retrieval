"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):

    def __init__(self):
        self.prior = defaultdict(lambda: 0.0)
        self.cond_spam = defaultdict(lambda: 0.0)
        self.cond_ham = defaultdict(lambda: 0.0)

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        
        if label == 'spam':
            return self.cond_spam[term]
        else:
            return self.cond_ham[term]
        

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        
        odds_ratio = []

        if label == 'spam':
            for term in self.cond_spam:
                try:
                    odds_ratio.append((self.cond_spam[term]/self.cond_ham[term],term))
                except ZeroDivisionError:
                    continue
        else:
            for term in self.cond_ham:
                try:
                    odds_ratio.append((self.cond_ham[term]/self.cond_spam[term],term))
                except ZeroDivisionError:
                    continue
        
        return sorted(odds_ratio, key = lambda x: x[0], reverse = True)[:n]

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        vocab = []
        num = len(documents)

        for d in documents:
            for tok in d.tokens:
               if tok not in vocab:
                   vocab.append(tok)
        
        num_s = 0
        num_h = 0

        prior = defaultdict(lambda: 0.0)
        T_spam = defaultdict(lambda: 0.0)
        T_ham = defaultdict(lambda: 0.0)
        cond_spam = defaultdict(lambda: 0.0)
        cond_ham = defaultdict(lambda: 0.0) 
        spam_sum = 0
        ham_sum = 0

        for d in documents:
            if d.label == 'spam':
                num_s += 1
                for tok in d.tokens:
                    T_spam[tok] += 1
            else:
                num_h += 1
                for tok in d.tokens:
                    T_ham[tok] += 1
        
        self.prior['spam'] = float(num_s/num)
        self.prior['ham'] = float(num_h/num)
            
        for tok in vocab:
            spam_sum += 1 + T_spam[tok]

            ham_sum += 1 + T_ham[tok]

        for tok in vocab:
            self.cond_spam[tok] = float((T_spam[tok] + 1)/spam_sum)
            self.cond_ham[tok] = float((T_ham[tok] + 1)/ham_sum)
            

    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        
        label_class = []
        score_spam = 0
        score_ham = 0

        for doc in documents:
            score_spam = math.log10(self.prior['spam'])
            score_ham = math.log10(self.prior['ham'])

            for token in doc.tokens:
                if self.cond_spam[token] > 0:
                    score_spam += math.log10(self.cond_spam[token])
                if self.cond_ham[token] > 0:    
                    score_ham += math.log10(self.cond_ham[token])
	
            if score_spam > score_ham:
                label_class.append('spam')
            else:
                label_class.append('ham')
        return label_class
	
	
def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    Z = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    
    x = 0
    corrx = 0
    y = 0
    incorry = 0
    z = 0
    incorrz = 0
    
    for i in range(len(predictions)):
        if documents[i].label == predictions[i]:
            corrx += 1
    x = float(corrx/len(documents))

    for i in range(len(predictions)):
        if documents[i].label == 'ham' and predictions[i] == 'spam':
            y += 1

    for i in range(len(documents)):
        if documents[i].label == 'spam' and predictions[i] == 'ham':
            z += 1
	    
    return (x,y,z)
    
def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
