import gc
import os
import pandas as pd 

from pandas import DataFrame
from exrecsys.utils.print_utils import *


def load_pickle(data_path):
    """Load data from a pickle file

    :param data_path: Path to the data file (.pkl)
    """
    print("\nLoading pickle file...")
    data_df = pd.read_pickle(data_path)

    try:
        data_df = data_df[data_df['content_type_id'] == False]
    except:
        print("[Warning] Column `content_type_id` not exists !")
    
    # arrange by timestamp
    data_df = data_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    del data_df["timestamp"]
    del data_df["content_type_id"]

    return data_df

def group_by_user_id(data_df: DataFrame, mode='kt'):
    """Group a DataFrame by user_id

    :param data_df: A DataFrame
    :param mode: Mode 'kt' or 'kccp'. 
                 If mode='kt', process data for KT module
                 if mode='kccp', process data for KCCP module
    """

    if mode.lower() == 'kccp':
        data_df = data_df[['user_id', 'concepts', 'answered_correctly']].astype(
            {
                'user_id': 'int32',
                'answered_correctly': 'int8'
            }
        )
        data_df = data_df.explode("concepts")
        group = (
            data_df.groupby("user_id")
            .apply(lambda r: (r["concepts"].values, r['answered_correctly'].values))
        )
        
    elif mode.lower() == 'kt':
        data_df = data_df[['user_id', 'content_id', 'concepts', 'answered_correctly']].astype(
            {
                'user_id': 'int32',
                'content_id': 'int16',
                'answered_correctly': 'int8'
            }
        )
        group = (
            data_df.groupby("user_id")
            .apply(lambda r: (r["content_id"].values, r["concepts"].values, r["answered_correctly"].values))
        )
    else:
        raise ValueError(f"Mode '{mode}' not in ['kt', 'kccp'] !")

    return group, data_df

def train_test_split(data_df: DataFrame, pct: float=0.1, mode='kt'):
    """Split DataFrame into Train and Test subsets

    :param data_df: A DataFrame
    :param pct: The ratio to split train/test set
    :param mode: Mode 'kt' or 'kccp'. 
                 If mode='kt', process data for KT module
                 if mode='kccp', process data for KCCP module
    """
    train_percent = 1 - pct
    train = data_df.iloc[:int(train_percent * len(data_df))]
    test = data_df.iloc[int(train_percent * len(data_df)):]

    print_line(text="Dataset Info")

    print(f"- Shape of Train dataset: {train.shape}")
    print(f"- Shape of Test dataset: {test.shape}")

    if mode.lower() == 'kt':
        train_group = (
            train[['user_id', 'content_id', 'concepts', 'answered_correctly']]
            .groupby('user_id')
            .apply(lambda r: (r['content_id'].values, r["concepts"].values, r['answered_correctly'].values))
        )

        test_group = (
            test[['user_id', 'content_id', 'concepts', 'answered_correctly']]
            .groupby('user_id')
            .apply(lambda r: (r['content_id'].values, r["concepts"].values, r['answered_correctly'].values))
        )
    elif mode.lower() == 'kccp':
        train_group = (
            train[['user_id', 'concepts', 'answered_correctly']]
            .groupby('user_id')
            .apply(lambda r: (r['concepts'].values, r['answered_correctly'].values))
        )

        test_group = (
            test[['user_id', 'concepts', 'answered_correctly']]
            .groupby('user_id')
            .apply(lambda r: (r['concepts'].values, r['answered_correctly'].values))
        )
    else:
        raise ValueError(f"Mode '{mode}' not in ['kt', 'kccp'] !")

    return train_group, test_group


def get_n_question(data_df: DataFrame):
    """Get the numbers of the skills

    :param data_df: A DataFrame
    """
    skills = data_df["content_id"].unique()
    n_skills = len(skills)
    print(f"\n- Numbers of the questions: {n_skills}")

    return n_skills
