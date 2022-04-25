import os
import time
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from exrecsys.utils import *
from exrecsys.metrics import SystemMetrics
from exrecsys.interpreter import ExercisesRecommendInterpreter

if __name__ == '__main__':

    # Parse Auguments
    parse = argparse.ArgumentParser(
        description="Argument to training the model"
    )
    
    # define type of module
    parse.add_argument("--mode", choices=['test', 'eval'], default='eval', help="MODE.")
    parse.add_argument("--user_id", type=int, default=1984659, help="The user_id interaction.")

    params = parse.parse_args()

    # load history data 
    data_df = load_pickle(data_path='./data/train-all-2.pkl')

    interpreter = ExercisesRecommendInterpreter()
    now = datetime.now()

    interpreter.load_group(data=data_df, mode='kt')
    interpreter.load_group(data=data_df, mode='kccp')

    print(f"\nLoading dataset in {datetime.now() - now} (s)")

    kt_config = {
        'model_type': 'akt',
        'n_question': 13523,
        'n_pid': -1,
        'n_blocks': 1,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.05,
        'kq_same': 1,
        'l2': 1e-5,
        'final_fc_dim': 512,
        'd_ff': 2048,
    }

    interpreter.load_module_kt(
        config_params=kt_config,
        model_path='./models/kt_model_best.pt'
    )
    
    # Use LSTM model
    # kccp_config = {
    #     'model_type': 'lstm',
    #     'n_concept': 188,
    #     'input_dim':100,
    #     'num_layers': 2,
    #     'hidden_dim': 1024,
    #     'dropout': 0.2,
    # }

    # Use VAKT model
    kccp_config = {
        'model_type': 'vakt',
        'n_concept': 188,
        'n_blocks': 1,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.05,
        'kq_same': 1,
        'l2': 1e-5,
        'final_fc_dim': 2048,
        'd_ff': 2048,
    }
    interpreter.load_module_kccp(
        config_params=kccp_config,
        model_path='./models/vakt_model_best.pt'
    )

    # get exercises_bank and knowledge concepts

    print(f"\nGET EXERCISES BANK AND KNOWLEDGE CONCEPTS...")
    exercises_bank = []
    concepts = []
    temp = []

    content_df = pd.read_csv('./data/questions.csv', encoding='utf-8')
    content_df['tags'] = content_df['tags'].str.split(" ")
    content_df = content_df.rename(columns={'question_id': 'content_id', 'tags': 'concepts'}, inplace=False)
    content_df['diffculty'] = [1.0]*len(content_df)

    for i in tqdm(range(len(content_df))):
        excercise_info = {
            'content_id': content_df['content_id'][i],
            'concepts': content_df['concepts'][i],
            'difficulty': 1.0
        }
        try:
            concepts.extend(content_df['concepts'][i])
        except:
            print(i)
            print(content_df['concepts'][i])
        exercises_bank.append(excercise_info)

    knowledge_concepts = {
        int(i): str(i) for i in list(set(concepts)) 
    }


    print(f"\nLength of Knowledge concepts: {len(knowledge_concepts)}")
    print(f"Length of Exercise bank: {len(exercises_bank)}\n")

    interpreter.load_module_filer(
        exercise_bank=exercises_bank,
        knowledge_concepts=knowledge_concepts
    )

    system_metric = SystemMetrics()

    now = datetime.now()

    if params.mode.upper() == 'EVAL':
        # Calculate metrics system
        accuracy = []
        novelty = []
        diversity = []

        list_users = list(set(data_df['user_id'].tolist()))[:300]

        print(f"Evaluate on the numbers of list users: {len(list_users)}")
        
        for user_id in tqdm(list_users): 
            rel = interpreter.process(
                m=100,
                k=5,
                user_id=user_id,
                desired_difficulty=0.5
            )
            group = interpreter.kt_group
            histories = group[user_id]
            acc = system_metric.accuracy(rel=rel[0])
            nov = system_metric.novelty(rel=rel[0], histories=histories)
            div = system_metric.diversity(rel=rel[0], knowledge_concepts=knowledge_concepts)
            
            accuracy.append(acc)
            novelty.append(nov)
            diversity.append(div)

        print(f"SYSTEM METRICS: ")
        print(f"\t- Accuracy = {np.mean(accuracy)}")
        print(f"\t- Novelty = {np.mean(novelty)}")
        print(f"\t- Diversity = {np.mean(diversity)}")

    
    else:
        rel = interpreter.process(
            m=100,
            k=5,
            user_id=params.user_id,
            desired_difficulty=0.5
        )

        group = interpreter.kt_group
        histories = group[params.user_id]
        acc = system_metric.accuracy(rel=rel[0])
        nov = system_metric.novelty(rel=rel[0], histories=histories)
        div = system_metric.diversity(rel=rel[0], knowledge_concepts=knowledge_concepts)

        print(f"\n- Accuracy = {acc}")
        print(f"\n- Novelty = {nov}")
        print(f"- Diversity = {div}")

    print(f"\n> TOTAL TIME PROCESSING SYSTEM: {datetime.now() - now}")
