import os
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from pandas.core.series import Series
from typing import Union, List, Dict, Any

from torch.nn import parameter
from exrecsys.learner_models import model

from exrecsys.utils import *
from exrecsys.learner_models.model import AKT
from exrecsys.domain_models.model import LSTM, VAKT
from exrecsys.learner_models.learner import AKTLearner
from exrecsys.filtering_layers import EBFilter, RELGenerator
from exrecsys.domain_models.learner import KCCPLearner, VAKTLearner

class ExercisesRecommendInterpreter():
    def __init__(
        self,
        kt_group: Any=None,
        kccp_group: Any=None,
        kt_learner: AKTLearner=None,
        kccp_learner: Union[KCCPLearner, VAKTLearner]=None,
        eb_filter: EBFilter=None,
        rel_generator: RELGenerator=None
    ):
        self.kt_group = kt_group
        self.kccp_group = kccp_group
        self.kt_learner = kt_learner
        self.kccp_learner = kccp_learner
        self.eb_filter = eb_filter
        self.rel_generator = rel_generator

    def load_module_kt(
        self,
        config_params: Dict={},
        model_path: Union[str, Path]=None
    ):
        model_path = os.path.abspath(model_path)
        model_type = config_params.get('model_type', 'akt')

        if model_type.lower() == 'akt':
            kt_model = AKT(
                n_question=config_params.get('n_question', 13523),
                n_pid=config_params.get('n_pid', -1),
                n_blocks=config_params.get('n_blocks', 1),
                d_model=config_params.get('d_model', 256),
                kq_same=config_params.get('kq_same', 1),
                dropout=config_params.get('dropout', 0.05),
                l2=config_params.get('l2', 1e-5),
                n_heads=config_params.get('n_heads', 8),
                d_ff=config_params.get('d_ff', 2048),
                final_fc_dim=config_params.get('final_fc_dim', 512),
                model_type=model_type
            )
            self.kt_learner = AKTLearner(model=kt_model)
            self.kt_learner.load_model(model_path=model_path)
            self.n_question = self.kt_learner.model.n_question

    def load_module_kccp(
        self,
        config_params: Dict={},
        model_path: Union[str, Path]=None
    ):
        model_path = os.path.abspath(model_path)
        model_type = config_params.get('model_type', 'vakt')

        if model_type.lower() == 'lstm':
            kccp_model = LSTM(
                input_dim=config_params.get('input_dim', 100),
                hidden_dim=config_params.get('hidden_dim', 100),
                num_layers=config_params.get('num_layers', 2),
                n_concept=config_params.get('n_concept', 188),
                dropout=config_params.get('dropout', 0.2)
            )
            
            self.kccp_learner = KCCPLearner(model=kccp_model)
            self.kccp_learner.load_model(model_path)
            self.n_concept = self.kccp_learner.model.n_concept
        
        elif model_type.lower() == 'vakt':
            kccp_model = VAKT(
                n_concept=config_params.get('n_concept', 188),
                n_blocks=config_params.get('n_blocks', 1),
                d_model=config_params.get('d_model', 256),
                n_heads=config_params.get('n_heads', 8),
                dropout=config_params.get('dropout', 0.05),
                kq_same=config_params.get('kq_same', 1),
                l2=config_params.get('l2', 1e-5),
                final_fc_dim=config_params.get('final_fc_dim', 2048),
                d_ff=config_params.get('d_ff', 2048)
            )

            self.kccp_learner = VAKTLearner(model=kccp_model)
            self.kccp_learner.load_model(model_path)
            self.n_concept = self.kccp_learner.model.n_concept


    def load_module_filer(
        self,
        temperature: float=100,
        reduction_factor: float=0.095,
        n_iterations: int=100,
        exercise_bank: List[Dict[Any, Any]]=None,
        knowledge_concepts: Dict[Any, Any]=None
    ):
        self.eb_filter = EBFilter(
            kt_group=self.kt_group,
            kt_learner=self.kt_learner,
            exercise_bank=exercise_bank,
            knowledge_concepts=knowledge_concepts
        )

        self.rel_generator = RELGenerator(
            knowledge_concepts=knowledge_concepts
        )

    def load_group(
        self,
        data: Union[str, DataFrame],
        mode: str='kt'
    ):
        if isinstance(data, str):
            if not os.path.splitext(data)[-1] == '.pickle':
                raise ValueError("'data_path' must be a pickle file. ")  
            data_df = load_pickle(data_path=data)
        elif isinstance(data, DataFrame):
            data_df = data

        if mode.lower() == 'kt':
            self.kt_group, _ = group_by_user_id(data_df=data_df, mode='kt')
        elif mode.lower() == 'kccp':
            self.kccp_group, _ = group_by_user_id(data_df=data_df, mode='kccp')

    def process(
        self,
        user_id: int=None,
        m: int=100,
        k: int=5,
        desired_difficulty: float=0.5,
        exercises_status: Dict[Any, Any]=None,
        **kwargs
    ):
        """Process the input through the pipeline of modules and output 
        the output of pipeline as a list recommended exercises. 
        
        :param user_id: The id of user interaction
        :param m: The number of exercises set (ES)
        :param k: The number of the recommended exercises list
        :param desired_difficulty: The desired difficulty of user
        :param exercises_status: The status of exercises against the user. 
                                 If True, the user has done that exercise

        :returns: K recommended exercises and the weight corresponding
        """
        prob_answering_exercises = {}
        exercise_bank = self.eb_filter.exercise_bank

        now_1 = datetime.now()

        # Process get The next concepts
        prob_next_concepts, next_concept = self.kccp_learner.infer(
            group=self.kccp_group,
            user_id=user_id
        )

        # Process EBFilter
        sampled_exercises, delta = self.eb_filter.express_filters(
            n_samples=m,
            user_id=user_id,
            desired_difficulty=desired_difficulty,
            prob_next_concepts=prob_next_concepts,
        )

        # Process RELGenerator
        rel = self.rel_generator.generate(
            n_samples=k,
            eps=1e-5,
            n_iterations=100,
            temperature=100,
            exercise_set=sampled_exercises.tolist()
        )

        return rel

    def update_interaction(
        self,
        user_id: int=None,
        next_concept: int=None,
        content_id: int=None,
        answered_correctly: int=None
    ):
        if user_id in self.kt_group.index:
            self.kt_group[user_id] = (
                np.append(self.kt_group[user_id][0], content_id),
                np.append(self.kt_group[user_id][1], answered_correctly)
            )

            self.kccp_group[user_id] = (
                np.append(self.kccp_group[user_id][0], next_concept)
            )