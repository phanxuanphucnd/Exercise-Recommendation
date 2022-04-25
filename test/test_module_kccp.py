import numpy
import pandas as pd

from exrecsys.utils import *
from exrecsys.domain_models.model import LSTM, VAKT
from exrecsys.domain_models.dataset import KCCPDataset, VAKTDataset
from exrecsys.domain_models.learner import KCCPLearner, VAKTLearner

def test_train():
    data_df = load_pickle(data_path='./data/train-all-2.pkl')

    group, data_df = group_by_user_id(data_df, mode='kccp')
    train_group, test_group = train_test_split(data_df=data_df, pct=0.1, mode='kccp')

    n_concept = 188
    max_seq = 200

    # train_dataset = KCCPDataset(
    #     train_group, n_concept=n_concept, max_seq=max_seq)
    # test_dataset = KCCPDataset(
    #     test_group, n_concept=n_concept, max_seq=max_seq)

    train_dataset = VAKTDataset(
        train_group, n_concept=n_concept, max_seq=max_seq
    )
    test_dataset = VAKTDataset(
        test_group, n_concept=n_concept, max_seq=max_seq
    )

    if isinstance(train_dataset, KCCPDataset):
        # Define parameters
        input_dim = 100
        num_layers = 2
        hidden_dim = 100
        dropout = 0.2

        batch_size = 36
        learning_rate = 0.001
        max_learning_rate = 2e-2
        n_epochs = 100

        model = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_concept=n_concept,
            dropout=dropout
        )

        learner = KCCPLearner(model=model)
        learner.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_learning_rate=max_learning_rate,
            n_epochs=n_epochs,
            save_path='./models',
            model_name='kccp_model'
        )
    
    elif isinstance(train_dataset, VAKTDataset):
            # Define paramerters
        max_seq = 200
        n_blocks = 1
        d_model = 256
        dropout = 0.05
        kq_same = 1
        n_heads = 8
        d_ff = 2048
        l2 = 1e-5
        final_fc_dim = 2048

        batch_size = 36
        learning_rate = 1e-5
        max_learning_rate = 2e-3
        n_epochs = 100

        model = VAKT(
            n_concept=n_concept,
            n_blocks=n_blocks,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kq_same=kq_same,
            l2=l2,
            final_fc_dim=final_fc_dim,
            d_ff=d_ff
        )

        learner = VAKTLearner(model=model)
        learner.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            max_learning_rate=max_learning_rate,
            save_path='./models',
            model_name='vakt_model'
        )

def test_inference():
    data_df = load_pickle(data_path='./data/train-all.pkl')

    group, data_df = group_by_user_id(data_df, mode='kccp')

    # Define parameters
    n_concept = 188
    input_dim = 100
    num_layers = 2
    hidden_dim = 100
    dropout = 0.2
    max_seq = 200
    
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_concept=n_concept,
        dropout=dropout
    )

    learner = KCCPLearner(model=model)
    learner.load_model(model_path='./models/kccp_model.pt')

    prob, pred = learner.infer(group=group, n_concept=n_concept, user_id=46886)

    # print(pred)
    # print(prob)


test_train()
# test_inference()
