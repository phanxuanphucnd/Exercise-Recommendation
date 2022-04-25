import os
import argparse

from exrecsys.utils import *
from exrecsys.filtering_layers import EBFilter
from exrecsys.learner_models.model import AKT
from exrecsys.domain_models.model import LSTM
from exrecsys.domain_models.dataset import KCCPDataset
from exrecsys.domain_models.learner import KCCPLearner
from exrecsys.learner_models.dataset import AKTDataset
from exrecsys.learner_models.learner import AKTLearner

if __name__ == '__main__':

    # Parse Auguments
    parse = argparse.ArgumentParser(
        description="Argument to training the model"
    )
    
    # define type of module
    parse.add_argument("--module_type",
                        choices=['kt', 'kccp'],
                        default='kt',
                        help="The type of module to training")

    # General Parameters
    parse.add_argument("--data_path",
                        type=str,
                        help="The path to train dataset (.pickle")
    parse.add_argument("--batch_size",
                        type=int,
                        default=36,
                        help="The batch size value")
    parse.add_argument("--max_seq",
                        type=int,
                        default=200,
                        help="The maximun of sequence length")
    parse.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="The learning rate value")
    parse.add_argument("--dropout",
                        type=float,
                        default=0.05,
                        help="The dropout value")
    parse.add_argument("--max_learning_rate",
                        type=float,
                        default=2e-3,
                        help="The maximun of learning rate value")
    parse.add_argument("--n_epochs",
                        type=int,
                        default=50,
                        help="The number of epochs")
    # Storage Parameters
    parse.add_argument("--base_path",
                        type=str,
                        default='./models',
                        help="The folder path to save the model")
    parse.add_argument("--model_name",
                        type=str,
                        default='akt_model',
                        help="The name file of model")
    
    # KT parameters
    parse.add_argument("--n_question",
                        type=int,
                        help="The number of questions")
    parse.add_argument("--n_pid",
                        type=int,
                        default=-1,
                        help="The pid value")
    parse.add_argument("--n_blocks",
                        type=int,
                        default=1,
                        help="The number of blocks")
    parse.add_argument("--d_model",
                        type=int,
                        default=256,
                        help="The d_model value")
    parse.add_argument("--kq_same",
                        type=int,
                        default=1,
                        help="The kq_same value")
    parse.add_argument("--n_heads",
                        type=int,
                        default=8,
                        help="The numbers of heads in Transformers architecture")
    parse.add_argument("--d_ff",
                        type=int,
                        default=2048,
                        help="The number unit of feed forward layer in Transformer architecture")
    parse.add_argument("--l2",
                        type=float,
                        default=1e-5,
                        help="The l2 value")
    parse.add_argument("--final_fc_dim",
                        type=int,
                        default=512,
                        help="The number unit of the final fully connected layer")

    
    # KCCP parameters
    parse.add_argument("--n_concept",
                        type=int,
                        default=188,
                        help="The number of concepts")
    parse.add_argument("--input_dim",
                        type=int,
                        default=100,
                        help="The input dimension")
    parse.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="The number of layers")
    parse.add_argument("--hidden_dim",
                        type=int,
                        default=100,
                        help="The hidden dimension")

    params = parse.parse_args()

    data_df = load_pickle(data_path=params.data_path)

    if params.module_type.upper() == 'KT':
        group, data_df = group_by_user_id(data_df=data_df)
        train_group, test_group = train_test_split(data_df=data_df, pct=0.1)

        train_dataset = AKTDataset(train_group, n_question=params.n_question, max_seq=params.max_seq)
        test_dataset = AKTDataset(test_group, n_question=params.n_question, max_seq=params.max_seq)

        kt_model = AKT(
            n_question=params.n_question,
            n_pid=params.n_pid,
            n_blocks=params.n_blocks,
            d_model=params.d_model,
            n_heads=params.n_heads,
            dropout=params.dropout,
            kq_same=params.kq_same,
            model_type='akt',
            l2=params.l2,
            d_ff=params.d_ff,
            final_fc_dim=params.final_fc_dim
        )

        learner = AKTLearner(model=kt_model)
        learner.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            n_epochs=params.n_epochs,
            max_learning_rate=params.max_learning_rate,
            save_path=params.base_path,
            model_name=params.model_name
        )

    elif params.module_type.upper() == 'KCCP':
        group, data_df = group_by_user_id(data_df, mode='kccp')
        train_group, test_group = train_test_split(data_df=data_df, pct=0.1, mode='kccp')

        train_dataset = KCCPDataset(
            train_group, n_concept=params.n_concept, max_seq=params.max_seq)
        test_dataset = KCCPDataset(
            test_group, n_concept=params.n_concept, max_seq=params.max_seq)

        model = LSTM(
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            n_concept=params.n_concept,
            dropout=params.dropout
        )

        learner = KCCPLearner(model=model)
        learner.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            max_learning_rate=params.max_learning_rate,
            n_epochs=params.n_epochs,
            save_path=params.base_path,
            model_name=params.model_name
        )

    else:
        print(f"The 'module_type' is not supported! "
              f"Please choices in [kt, kccp].")