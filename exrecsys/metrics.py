import numpy as np

from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_distances

class SystemMetrics():
    def __init__(
        self,
        rel: List[Dict[Any, Any]]=None
    ):
        super(SystemMetrics, self).__init__()
        """Initialize a EMetrics class for calculate metrics for a recommendation 
        system in adaptive learning. """

        self.rel = rel

    def jaccdist(self, seq1, seq2):
        """Function to calculate jaccard similarity between two array like. """
        set1, set2 = set(seq1), set(seq2)

        return 1 - len(set1 & set2) / float(len(set1 | set2))

    def accuracy(
        self,
        rel: List[Dict[Any, Any]],
        **kwargs
    ):
        """Calculate metric Accuracy

        :param rel: The recommended exercises list
        :param delta: The difficulty score of exercises with the a given user 
        """
        if not rel:
            rel = self.rel

        delta = [v.get('delta') for v in rel]
        accuracy = 1 - np.average(delta)
        
        return accuracy

    def novelty(
        self,
        rel: List[Dict[Any, Any]],
        histories: Tuple,
        **kwargs
    ):
        """Calculate metric Novelty

        :param rel: The recommended exercises list
        :param correctly_answered: The answered exercises list, 1 if correctly answered, else 0
        """
        if not rel:
            rel = self.rel

        kcrl = []
        
        concepts, res = histories[1], histories[2]
        for i in range(len(res)):
            if res[i] == 1:
                kcrl.extend(concepts[i])
        
        dists = []
        for i in range(len(rel)):
            kce = rel[i].get('concepts', [])
            jaccdist = self.jaccdist(kce, kcrl)
            dists.append(jaccdist)

        novelty = np.average(dists)

        return novelty

    def embed(self, exercise: Dict[Any, Any]):
        """Function to embeding exercises e into a vector embedding with length is the 
        length of knowledge concepts.

        :param exercise: The exericise of user interaction
        
        :returns: A vector embedding
        """
        embedding = [0]*len(self.knowledge_concepts)

        for i in range(len(self.knowledge_concepts)):
            if self.knowledge_concepts[i] in exercise.get('concepts', []):
                embedding[i] = 1
        
        return np.array(embedding).reshape(1, -1)

    def diversity(
        self,
        rel: List[Dict[Any, Any]],
        knowledge_concepts: Dict[Any, Any],
        **kwargs
    ):
        """Calculate metric Accuracy

        :param rel: The recommended exercises list
        :param knowledge_concepts: A dictionary mapping index to concept
        """
        if not rel:
            rel = self.rel
        
        U = len(rel)
        self.knowledge_concepts = knowledge_concepts

        different = []
        for i in range(U):
            for j in range(U):
                if i != j:
                    exercise1 = self.embed(rel[i])
                    exercise2 = self.embed(rel[j])
                    diff = cosine_distances(exercise1, exercise2)

                    different.append(diff)

        diversity = np.sum(different) / (U**2 - U)

        return diversity
