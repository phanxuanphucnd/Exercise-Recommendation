import random
import numpy as np

from matplotlib import pyplot
from typing import List, Any, Dict
from scipy.constants import Boltzmann
from exrecsys.utils.print_utils import *
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

class RELGenerator():
    def __init__(
        self,
        knowledge_concepts: Dict[Any, Any]=None
    ):
        super(RELGenerator, self).__init__()

        self.knowledge_concepts = knowledge_concepts

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

    def distance(self, array_1, array_2):
        return cosine_distances(array_1, array_2)[0]

    def objective(self, input):
        n_samples = len(input)
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i][j] = self.distance(self.embed(input[i]), self.embed(input[j]))

        return np.mean(distance_matrix)

    def replace(self, curr, exercise_set, number: int=2):
        
        candidate = curr[:]
        temp_list = np.random.choice(exercise_set, number)

        for id, value in zip(np.random.choice(range(len(curr)), size=len(temp_list), replace=False), temp_list):
            candidate[id] = value

        return candidate

    def generate(
        self,
        exercise_set: List[Dict[Any, Any]]=None,
        n_samples: int=5,
        eps: float=1e-5,
        n_iterations: int=100,
        temperature: float=100,
        reduction_factor: float=0.095,

        **awargs
    ):
        best = random.sample(exercise_set, n_samples)
        best_eval = self.objective(best)

        curr, curr_eval = best, best_eval
        scores = list()

        for i in range(n_iterations):
            # Take a step
            candidate = self.replace(curr, exercise_set)
            # Eval candidate point
            candidate_eval = self.objective(candidate)

            if candidate_eval > best_eval:
                # Store new best point
                best, best_eval = candidate, candidate_eval

                # Keep track of scores
                scores.append(best_eval)

                # Report progress
                # print_style_notice(message=f"Updated step: {i} - {best_eval}")

            # Difference between candidate and current point 
            diff = candidate_eval - curr_eval

            # Calculate metropolis acceptable criterion
            metropolis = np.exp(-diff / temperature)

            # Check if we should keep the new point
            gamma = np.random.rand()

            if diff >= 0 or gamma >= metropolis:
                # Store the new current point
                curr, curr_eval = candidate, candidate_eval

            # Calculate temperature for current epoch
            temperature = temperature * 0.95

        return [best, best_eval, scores]

    def plot(self, scores):
        pyplot.plot(scores, '.-')
        pyplot.xlabel('Improvement Number')
        pyplot.ylabel('Evaluation function')
        pyplot.show()