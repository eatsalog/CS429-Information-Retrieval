""" Assignment 2
"""
import abc

import numpy as np


class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.

        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        precise = 0.0

        for x in hits:
            if x in relevant:
                precise += 1.0

        if len(hits) == 0:
            return 0
        else:
            res = precise/len(hits)
	
        return res

    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.

        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        recall = 0.0

        for x in hits:
            if x in relevant:
                recall += 1.0

        if len(relevant) == 0:
            return 0

        else:    
            res = recall/len(relevant)
	
        return res


    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.

        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        precise = 0.0

        for x in hits:
            if x in relevant:
                precise += 1.0
        
        if len(hits) == 0:
            precise = 0
        else:
            precise /= len(hits)
        
        recall = 0.0

        for x in hits:
            if x in relevant:
                recall += 1.0

        if len(relevant) == 0:
            recall = 0
        else:
            recall /= len(relevant)
	
        if precise == 0 and recall == 0:
            return 0
        
        else:
            res = (2 * precise * recall)/(precise + recall)
	
        return res


    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.

        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """
        cnum = 0
        emap = 0.0

        for x, ind in enumerate(hits):
            if ind in relevant:
                cnum += 1.0
                emap += cnum/(x + 1)
	
        if len(relevant) == 0:
            return 0
        else:
            res = emap/len(relevant)

        return res

    def __repr__(self):
        return 'MAP'

