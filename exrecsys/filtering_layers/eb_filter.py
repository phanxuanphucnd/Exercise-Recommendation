import numpy as np

from datetime import datetime
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity

from exrecsys.learner_models.learner import AKTLearner

class EBFilter():
    def __init__(
        self,
        kt_group: Any=None,
        kt_learner: AKTLearner=None,
        exercise_bank: List[Dict[Any, Any]]=None,
        knowledge_concepts: Dict[Any, Any]=None
    ):
        super(EBFilter, self).__init__()

        if not exercise_bank:
            raise ValueError(f"Exercise bank must be not empty!")

        self.kt_group = kt_group
        self.kt_learner = kt_learner
        self.exercise_bank = exercise_bank
        self.size_exercise_bank = len(exercise_bank)
        self.knowledge_concepts = knowledge_concepts
 
    def difficulty_estimate(
        self,
        exercise_id: int,
        prob_answering_exercises: Dict[Any, Any]
    ):
        """Calculate the difficulty of exercises

        :param prob_answering_exercises: The probability of correctly answering exercises
        """
        prob = prob_answering_exercises.get(exercise_id, -1)
        difficulty = 1 - prob

        return difficulty, prob

    def embed(self, exercise: Dict[Any, Any]):
        """Function to embeding exercises e into a vector embedding with length is the 
        length of knowledge concepts.

        :param exercise: The exericise of user interaction
        
        :returns: A vector embedding
        """
        embedding = [0] * len(self.knowledge_concepts)

        for i in range(len(self.knowledge_concepts)):
            if self.knowledge_concepts[i] in exercise.get('concepts', []):
                embedding[i] = 1
        
        return np.array(embedding).reshape(1, -1)

    def filters(
        self,
        n_samples: int=30,
        user_id: int=None,
        desired_difficulty: float=0.5,
        exercises_status: Dict[Any, Any]=None,
        prob_next_concepts: List[float]=None,
        prob_answering_exercises: Dict[Any, Any]=None,
        **kwargs
    ):
        """Function to filter exercises
        
        :param n_samples: The number of sampled exercises
        :param user_id: The id of user interaction
        :param desired_difficulty: The desired difficulty of user
        :param exercises_status: The status of exercises against the user. 
                                 If True, the user has done that exercise
        :param prob_next_concepts: The probability of the knowledge concept next time
        :param prob_answering_exercises: The probability of correctly answering exercises

        :returns: K recommended exercises and the weight corresponding
        """
        omegas = []
        distances = []
        probs_correctly_answered = []
        for i in range(self.size_exercise_bank):
            e = self.exercise_bank[i]
            e_id = e.get('content_id', None)
            e_emebedding = self.embed(e)
            
            # get similarity between the next concept and the embedding concepts of exercise
            prob_next_concepts = np.array(prob_next_concepts).reshape(1, -1)
            sim = cosine_similarity(e_emebedding, prob_next_concepts)[0][0]

            # get the the distance of difficulty
            e_difficulty, prob = self.difficulty_estimate(
                exercise_id=e_id, 
                prob_answering_exercises=prob_answering_exercises
            )
            dis = desired_difficulty - e_difficulty
            
            distances.append(dis)
            probs_correctly_answered.append(prob)
            omegas.append(np.sqrt(sim**2 + dis**2))


        # Sort exercises by weight omegas
        sorted_index = np.argsort(omegas)[::-1]

        rid = sorted_index[:n_samples]
        rel = np.array(self.exercise_bank)[rid]
        rel_distances = np.array(distances)[rid]
        rel_probs = np.array(probs_correctly_answered)[rid]
        
        return rel, rel_distances

    def express_filters(
        self,
        n_samples: int=100,
        user_id: int=None,
        desired_difficulty: float=0.5,
        exercises_status: Dict[Any, Any]=None,
        prob_next_concepts: List[float]=None,
        **kwargs
    ):
        omegas = []
        exercises_set = []

        weighted_concepts = self.weighted_concepts_through_time(user_id=user_id)

        prob = [a * b for a, b in zip(weighted_concepts, prob_next_concepts)]

        # Get 3 knowledge concepts highest 
        # next_concepts = [np.argsort(prob).tolist()[0]]
        next_concepts = np.argsort(prob).tolist()[:3]
        next_concepts = [self.knowledge_concepts[i] for i in next_concepts]
        
        for i in range(self.size_exercise_bank):
            if any(c in self.exercise_bank[i].get('concepts') for c in next_concepts):
                exercises_set.append(self.exercise_bank[i])

        for i in range(len(exercises_set)):
            content_id = exercises_set[i].get('content_id', None)
            out = self.kt_learner.infer(
                self.kt_group,
                self.kt_learner.model.n_question,
                user_id=user_id,
                content_id=content_id
            )
            exercises_set[i]['delta'] = abs(desired_difficulty - (1 - out))
            dis = abs(desired_difficulty - (1 - out))
            omegas.append(dis)
        
        K = min(n_samples, len(exercises_set))
        sorted_index = np.argsort(omegas)[::-1]

        rid = sorted_index[:K]
        rel = np.array(exercises_set)[rid]
        rel_distances = np.array(omegas)[rid]
        
        return rel, rel_distances

    def weighted_concepts_through_time(
        self,
        user_id: int=None
    ):
        """Return a vector with demision as same as the length of knowledge concepts,
        present the weighted of the knowledge concepts.
        """
        weighted_concepts = []
        count_concepts = {}
        
        for k, v in self.knowledge_concepts.items():
            count_concepts[k] = {}
            count_concepts[k]['0'] = 0 
            count_concepts[k]['1'] = 0
            count_concepts[k]['SUM'] = 0

        content_ids, concepts, res = self.kt_group[user_id]

        for i in range(len(content_ids)):
            for j in range(len(concepts[i])):
                kj = int(concepts[i][j])
                count_concepts[kj]['SUM'] += 1
                count_concepts[kj][str(res[i])] += 1

        for i in range(len(self.knowledge_concepts)):
            # Get the concept ki
            ki = int(self.knowledge_concepts[i])
            # ri is the number of correct answers to the concept ki
            ri = count_concepts[ki].get('1')
            # ci is the occurence number of the concept ki
            ci = count_concepts[ki].get('SUM')

            if ci == 0:
                weighted_concepts.append(1)
            elif ci > 0:
                tmp = 1 - ri / ci
                weighted_concepts.append(tmp)
            else:
                raise ValueError(f"ci must be not negative numbers !")

        return weighted_concepts
        