import os
import torch
import numpy as np
from torch._C import clear_autocast_cache
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from exrecsys.utils import *
from exrecsys.domain_models.dataset import KCCPDataset, VAKTDataset


class KCCPLearner():
    def __init__(self, model=None, device=None):
        super(KCCPLearner, self).__init__()

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
        criterion
    ):
        self.model.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outputs = []

        for item in tqdm(train_dataloader):
            input = item[0].to(self.device).long()
            label = item[1].to(self.device).long()

            optimizer.zero_grad()
            output = self.model(input)

            output = output.view(-1, self.model.n_concept)
            label = label.view(-1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

            input = input.view(-1)
            target_mark = (input != 0)

            _, predicted = torch.max(output, dim=1)

            predicted = torch.masked_select(predicted, target_mark)
            label = torch.masked_select(label, target_mark)

            num_corrects += (predicted == label).sum().item()
            num_total += label.size(0)


            labels.extend(label.view(-1).data.cpu().numpy())
            outputs.extend(predicted.view(-1).data.cpu().numpy())

        loss = np.mean(train_loss)
        
        acc = num_corrects / num_total
        precision = precision_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        recall = recall_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        
        return loss, acc, precision, recall, f1

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
            input = item[0].to(self.device).long()
            label = item[1].to(self.device).long()

            output = self.model(input)

            output = output.view(-1, self.model.n_concept)
            label = label.view(-1)

            if criterion:
                loss = criterion(output, label)
                valid_loss.append(loss.item())

            input = input.view(-1)
            target_mark = (input != 0)

            _, predicted = torch.max(output, dim=1)

            predicted = torch.masked_select(predicted, target_mark)
            label = torch.masked_select(label, target_mark)

            num_corrects += (predicted == label).sum().item()
            num_total += label.size(0)


            labels.extend(label.view(-1).data.cpu().numpy())
            outputs.extend(predicted.view(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        precision = precision_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        recall = recall_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)

        loss = np.mean(valid_loss)
        
        return loss, acc, precision, recall, f1

    def train(
        self,
        train_dataset: KCCPDataset,
        test_dataset: KCCPDataset,
        batch_size: int=48,
        learning_rate: float=0.001,
        max_learning_rate: float=2e-3,
        eps: float=1e-8,
        betas: Tuple[float, float]=(0.9, 0.999),
        n_epochs: int=30,
        shuffle: bool=True,
        max_steps: int=10,
        num_workers: int=8,
        view_model: bool=True,
        save_path: str='./models',
        model_name: str='kccp_model',
        **kwargs
    ):
        """Training the model

        :param train_dataset: An KCCPDataset instance for train dataset
        :param test_dataset: An KCCPDataset instance for test dataset
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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs
        )

        criterion = nn.CrossEntropyLoss()
        
        self.model.to(self.device)
        criterion.to(self.device)

        if view_model:
            # print the model architecture
            print_line(text="Model Info")
            print(self.model)

        print(f"\n- Using device: {self.device}")


        step = 0
        best_f1 = 0
        max_steps = max_steps

        print_line(text='Training KCCP module')

        # check save path exists
        save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        for epoch in range(n_epochs):
            train_loss, _, _, _, train_f1 = self._train(train_dataloader, optimizer, scheduler, criterion)
            valid_loss, _, _, _, valid_f1 = self._validate(valid_dataloader, criterion=criterion)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n" 
                        f"\t- Train: loss = {train_loss:.4f}; f1 = {train_f1:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; f1 = {valid_f1:.4f} ")

            if valid_f1 >= best_f1:
                best_f1 = valid_f1
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': train_loss,
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
        dataset: KCCPDataset=None
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
        user_id,
        max_seq=200
    ):
        """Inference a given sample
        
        :param group: A Group DataFrame by user_id from raw dataset
        :param n_concept: The number of concepts in data
        :param user_id: The user_id interaction
        :param max_seq: The number of input sequence pass into the model

        :returns: The probability of the next concept
        """

        input = np.zeros(max_seq, dtype=int)
        res = np.zeros(max_seq, dtype=int)

        if user_id in group.index:
            input_, _ = group[user_id]
            input_ = input_.astype(int) + 1

            seq_length = len(input_)

            if seq_length >= max_seq:
                input = input_[-max_seq:]
            else:
                input[-seq_length:] = input_

        input = torch.from_numpy(input).to(self.device).long()
        input = torch.unsqueeze(input, dim=0)

        with torch.no_grad():
            output = self.model(input)

        # calculate softmax
        m = nn.Softmax(dim=2)
        
        output = m(output)
        v, pred = torch.max(output, dim=2)

        prob =  output[:, -1]
        prob = prob.view(-1).data.cpu().numpy()
        
        pred = pred[:, -1]
        pred = pred.view(-1).data.cpu().numpy()[0]

        return prob, pred



class VAKTLearner():
    def __init__(self, model=None, device=None):
        super(VAKTLearner, self).__init__()

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
        criterion
    ):
        self.model.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outputs = []

        for item in tqdm(train_dataloader):
            input = item[0].to(self.device).long()
            ca = item[1].to(self.device).long()
            label = item[2].to(self.device).long()

            optimizer.zero_grad()
            output = self.model(input, ca)

            output = output.view(-1, self.model.n_concept)
            label = label.view(-1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

            input = input.view(-1)
            target_mark = (input != 0)

            _, predicted = torch.max(output, dim=1)

            predicted = torch.masked_select(predicted, target_mark)
            label = torch.masked_select(label, target_mark)

            num_corrects += (predicted == label).sum().item()
            num_total += label.size(0)


            labels.extend(label.view(-1).data.cpu().numpy())
            outputs.extend(predicted.view(-1).data.cpu().numpy())

        loss = np.mean(train_loss)
        
        acc = num_corrects / num_total
        precision = precision_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        recall = recall_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        
        return loss, acc, precision, recall, f1

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
            input = item[0].to(self.device).long()
            ca = item[1].to(self.device).long()
            label = item[2].to(self.device).long()

            output = self.model(input, ca)

            output = output.view(-1, self.model.n_concept)
            label = label.view(-1)

            if criterion:
                loss = criterion(output, label)
                valid_loss.append(loss.item())

            input = input.view(-1)
            target_mark = (input != 0)

            _, predicted = torch.max(output, dim=1)

            predicted = torch.masked_select(predicted, target_mark)
            label = torch.masked_select(label, target_mark)

            num_corrects += (predicted == label).sum().item()
            num_total += label.size(0)


            labels.extend(label.view(-1).data.cpu().numpy())
            outputs.extend(predicted.view(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        precision = precision_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        recall = recall_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=outputs, average='weighted', zero_division=0)

        loss = np.mean(valid_loss)
        
        return loss, acc, precision, recall, f1

    def train(
        self,
        train_dataset: Union[KCCPDataset, VAKTDataset],
        test_dataset: Union[KCCPDataset, VAKTDataset],
        batch_size: int=48,
        learning_rate: float=0.001,
        max_learning_rate: float=2e-3,
        eps: float=1e-8,
        betas: Tuple[float, float]=(0.9, 0.999),
        n_epochs: int=30,
        shuffle: bool=True,
        max_steps: int=10,
        num_workers: int=8,
        view_model: bool=True,
        save_path: str='./models',
        model_name: str='kccp_model',
        **kwargs
    ):
        """Training the model

        :param train_dataset: An KCCPDataset instance for train dataset
        :param test_dataset: An KCCPDataset instance for test dataset
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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_learning_rate, steps_per_epoch=len(train_dataloader), epochs=n_epochs
        )

        criterion = nn.CrossEntropyLoss()
        
        self.model.to(self.device)
        criterion.to(self.device)

        if view_model:
            # print the model architecture
            print_line(text="Model Info")
            print(self.model)

        print(f"\n- Using device: {self.device}")


        step = 0
        best_f1 = 0
        max_steps = max_steps

        print_line(text='Training KCCP module')

        # check save path exists
        save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        for epoch in range(n_epochs):
            train_loss, _, _, _, train_f1 = self._train(train_dataloader, optimizer, scheduler, criterion)
            valid_loss, _, _, _, valid_f1 = self._validate(valid_dataloader, criterion=criterion)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n" 
                        f"\t- Train: loss = {train_loss:.4f}; f1 = {train_f1:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; f1 = {valid_f1:.4f} ")

            if valid_f1 >= best_f1:
                best_f1 = valid_f1
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': train_loss,
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
        dataset: KCCPDataset=None
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
        user_id,
        max_seq=200
    ):
        """Inference a given sample
        
        :param group: A Group DataFrame by user_id from raw dataset
        :param n_concept: The number of concepts in data
        :param user_id: The user_id interaction
        :param max_seq: The number of input sequence pass into the model

        :returns: The probability of the next concept
        """

        input = np.zeros(max_seq, dtype=int)
        res = np.zeros(max_seq, dtype=int)

        if user_id in group.index:
            input_, res_ = group[user_id]
            input_ = input_.astype(int) + 1

            seq_length = len(input_)

            if seq_length >= max_seq:
                input = input_[-max_seq:]
                res = res_[-max_seq:]
            else:
                input[-seq_length:] = input_
                res[-seq_length:] = res_

        ca =res.astype(int) * self.model.n_concept + input

        input = torch.from_numpy(input).to(self.device).long()
        input = torch.unsqueeze(input, dim=0)

        res = torch.from_numpy(res).to(self.device).float()
        res = torch.unsqueeze(res, dim=0)

        ca = torch.from_numpy(ca).to(self.device).long()
        ca = torch.unsqueeze(ca, dim=0)

        with torch.no_grad():
            output = self.model(input, ca)

        # calculate softmax
        m = nn.Softmax(dim=2)
        
        output = m(output)
        v, pred = torch.max(output, dim=2)

        prob =  output[:, -1]
        prob = prob.view(-1).data.cpu().numpy()
        
        pred = pred[:, -1]
        pred = pred.view(-1).data.cpu().numpy()[0]

        return prob, pred