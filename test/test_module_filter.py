
import numpy as np
from exrecsys.utils import *
from exrecsys.learner_models import AKT
from exrecsys.learner_models import AKTLearner
from exrecsys.filtering_layers import EBFilter

exercise_bank = [
    {
        'content_id': 10685,
        'concepts': ['a', 'b', 'e'],
        'difficulty': 0.7
    }, 
    {
        'content_id': 471,
        'concepts': ['b', 'd'],
        'difficulty': 0.6
    }, 
    {
        'content_id': 1056,
        'concepts': ['a', 'd'],
        'difficulty': 0.2
    }, 
    {
        'content_id': 592,
        'concepts': ['e', 'c'],
        'difficulty': 0.4
    }
]

knowledge_concepts = {
    0: 'a', 
    1: 'b',
    2: 'c', 
    3: 'd',
    4: 'e'
}

eb_filter = EBFilter(exercise_bank=exercise_bank, knowledge_concepts=knowledge_concepts)

user_id = 46886

data_df = load_pickle(data_path='./data/train-all-subset.pkl')

group, data_df = group_by_user_id(data_df=data_df)

n_question = 13523
max_seq = 200
n_pid = -1
n_blocks = 1
d_model = 256
dropout = 0.05
kq_same = 1
n_heads = 8
d_ff = 2048
l2 = 1e-5
final_fc_dim = 512

model = AKT(
    n_question=n_question, n_pid=n_pid, n_blocks=n_blocks,
    d_model=d_model, n_heads=n_heads, dropout=dropout,
    kq_same=kq_same, model_type='akt', l2=l2,
    final_fc_dim=final_fc_dim, d_ff=d_ff
)    

learner = AKTLearner(model=model)
learner.load_model('./models/akt_model.pt')

prob_answering_exercises = {}
for i in range(len(exercise_bank)):
    content_id = exercise_bank[i].get('content_id')
    output = learner.infer(
        group,
        n_question,
        user_id=46886, 
        content_id=content_id
    )
    prob_answering_exercises[content_id] = output

sampled_exercises, delta, probs = eb_filter.filters(
    k=2,
    user_id=user_id,
    desired_difficulty=0.5,
    prob_next_concepts=[0.1, 0.7, 0.1, 0.02, 0.08],
    prob_answering_exercises=prob_answering_exercises
)

from pprint import pprint
pprint(sampled_exercises)
print(delta)
print(probs)
