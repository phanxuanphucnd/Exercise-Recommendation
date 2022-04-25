import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Tuple, Any
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from exrecsys.utils import *
from exrecsys.learner_models.dataset import AKTDataset

class AKTLearner():
    def __init__(self, model=None, device=None):
        super(AKTLearner, self).__init__()

        self.model = model
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _train(
        self, 
        train_dataloader, 
        optimizer, 
        scheduler,
        criterion,
    ):
        self.model.train()
        
        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outputs = []

        for item in tqdm(train_dataloader):
            q = item[0].to(self.device).long()
            qa = item[1].to(self.device).long()
            res = item[2].to(self.device).float()

            optimizer.zero_grad()
            output = self.model(q, qa)
            loss = criterion(output, res)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

            target_mark = (q != 0)
            output = torch.masked_select(output, target_mark)
            res = torch.masked_select(res, target_mark)
            pred = (torch.sigmoid(output) >= 0.5).long()

            num_corrects += (pred == res).sum().item()
            num_total += len(res)

            labels.extend(res.view(-1).data.cpu().numpy())
            outputs.extend(output.view(-1).data.cpu().numpy())

        loss = np.mean(train_loss)
        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outputs)

        return loss, acc, auc

    def _validate(
        self,
        valid_dataloader,
        criterion=None
    ):
        self.model.eval()

        valid_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outputs = []

        for item in tqdm(valid_dataloader):
            q = item[0].to(self.device).long()
            qa = item[1].to(self.device).long()
            res = item[2].to(self.device).float()

            output = self.model(q, qa)

            print(f"Size of output = {output.size()}")
            print(f"Size of res = {res.size()}")

            if criterion:
                loss = criterion(output, res)
                valid_loss.append(loss.item())
            
            target_mark = (q != 0)
            output = torch.masked_select(output, target_mark)
            res = torch.masked_select(res, target_mark)
            pred = (torch.sigmoid(output) >= 0.5).long()

            num_corrects += (pred == res).sum().item()
            num_total += len(res)

            labels.extend(res.view(-1).data.cpu().numpy())
            outputs.extend(output.view(-1).data.cpu().numpy())
        
        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outputs)
        
        loss = np.mean(valid_loss)
        
        return loss, acc, auc

    def train(
        self, 
        train_dataset: AKTDataset=None,
        test_dataset: AKTDataset=None,
        batch_size: int=48,
        learning_rate: float=1e-5,
        n_epochs: int=50,
        max_learning_rate: float=2e-3,
        eps: float=1e-8,
        betas: Tuple[float, float]=(0.9, 0.999),
        shuffle: bool=True,
        max_steps: int=10,
        num_workers: int=8,
        view_model: bool=True,
        save_path: Union[str, Path]='./models', 
        model_name: str='akt_model',
        **kwargs
    ):
        """Training the model

        :param train_dataset: An AKTDataset instance for train dataset
        :param test_dataset: An AKTDataset instance for test dataset
        :param batch_size: The batch size value
        :param learning_rate: The learning rate value
        :param n_epochs: The number of epochs to training
        :param max_learning_rate: The maximun value of learning rate
        :param eps: Term added to the denominator to improve numerical stability
        :param betas: Coefficients used for computing running averages of gradient and its square
        :param shuffle: If True, shuffle dataset before training
        :param num_works: The number of workers
        :param save_path: Path to the file to save the model
        :param model_nam: The name of model to storage
        """
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        valid_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=betas, eps=eps
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs
        )
        self.model.to(self.device)
        criterion.to(self.device)

        if view_model:
            # print the model architecture
            print_line(text="Model Info")
            print(self.model)

        print(f"\n- Using device: {self.device}")

        step = 0
        best_auc = 0
        max_steps = max_steps
        # n_question = train_dataset.n_question

        print_line(text="Training KT module")

        # check save path exists
        save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for epoch in range(n_epochs):
            train_loss, train_acc, train_auc = self._train(train_dataloader, optimizer, scheduler, criterion)
            valid_loss, valid_acc, valid_auc = self._validate(valid_dataloader, criterion)
            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n"
                        f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f}; auc = {train_auc:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; acc = {valid_acc:.4f}; auc = {valid_auc:.4f} "
            )

            if valid_auc > best_auc:
                best_auc = valid_auc
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': train_loss,
                        # 'acc': train_acc,
                        # 'auc': train_auc
                    }, 
                    os.path.join(save_path, f"{model_name}.pt")
                )
            else:
                step += 1
                if step >= max_steps:
                    break
        
        print(f"Path to the saved model: {save_path}/{model_name}.pt")

    def train_online(
        self, 
        dataset: AKTDataset=None
    ):
        raise NotImplementedError()

    def load_model(self, model_path):
        """Load the pretrained model

        :param model_path: The path to the model
        """
        # check model file exists?
        if not os.path.isfile(model_path):
            raise ValueError(f"Model file '{model_path}' is not exists or broken!")

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

    def infer(
        self, 
        group, 
        n_question, 
        user_id, 
        content_id, 
        max_seq=200
    ):
        """Inference a given sample
        
        :param group: A Group DataFrame by user_id from raw dataset
        :param n_question: The number of questions in data
        :param user_id: The user_id interaction
        :param content_id: The content_id for the user interaction
        :param max_seq: The number of input sequence pass into the model

        :returns: The scalar value represents the probability answered correctly of the user
        """
        q = np.zeros(max_seq, dtype=int)
        res = np.zeros(max_seq, dtype=int)
        qa = np.zeros(max_seq, dtype=int)

        if user_id in group.index:
            q_, _, res_ = group[user_id]
            q_ = q_ + 1
            seq_len = len(q_)

            if seq_len >= max_seq:
                q = q_[-max_seq:]
                res = res_[-max_seq:]
            else:
                q[-seq_len:] = q_
                res[-seq_len:] = res_

        q = np.append(q[2:], [content_id + 1])
        res = np.append(res[2:], [1])
        qa = res.astype(int) * n_question + q

        input_q = torch.from_numpy(q).to(self.device).long()
        input_qa = torch.from_numpy(qa).to(self.device).long()
        res = torch.from_numpy(res).to(self.device).float()

        input_q = torch.unsqueeze(input_q, dim=0)
        input_qa = torch.unsqueeze(input_qa, dim=0)

        with torch.no_grad():
            output = self.model(input_q, input_qa)
            
        output = torch.sigmoid(output)
        output = output[:, -1]
        output = output.view(-1).data.cpu().numpy()[0]

        return output
