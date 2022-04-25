import numpy as np

from torch.utils.data import Dataset

class KCCPDataset(Dataset):
    def __init__(self, group, n_concept, max_seq, min_samples=2):
        super(KCCPDataset, self).__init__()
        """Initialize a KCCPDataset class
        
        :param group: A Group DataFrame by user_id from raw dataset
        :param n_concept: The number of concept in dataset
        :param max_seq: The maximum number of input sequence
        :param min_samples: The minimun number of sample to pass into input sequence
        """
        self.n_concept = n_concept
        self.max_seq = max_seq
        self.min_samples = min_samples

        self.samples = {}
        self.user_ids = []

        for user_id in group.index:
            input_, _ = group[user_id]
            if len(input_) < min_samples:
                continue

            seq_length = len(input_)
            if seq_length > self.max_seq:
                initial = seq_length % self.max_seq
                
                if initial > min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    input = input_[: initial]
                    label = input_[1: initial+1] 
                    self.samples[f"{user_id}_0"] = (input, label)
                
                for seq in range(seq_length // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq + 1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    
                    if end >= seq_length:
                        input = input_[start: end - 1]
                        label = input_[start + 1: ]
                    else:
                        input = input_[start: end]
                        label = input_[start + 1: end + 1]

                    self.samples[f"{user_id}_{seq + 1}"] = (input, label)
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (input_[: -1], input_[1:])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        input_, label_ = self.samples[user_id]
        
        input_ = input_.astype(int) + 1
        seq_length = len(input_)

        input = np.zeros(self.max_seq, dtype=int)
        label = np.zeros(self.max_seq, dtype=int)

        if seq_length >= self.max_seq:
            input[:] = input_[-self.max_seq:]
            label[:] = label_[-self.max_seq:]
        else:
            input[-seq_length:] = input_
            label[-seq_length:] = label_

        return input, label


class VAKTDataset(Dataset):
    def __init__(self, group, n_concept, max_seq, min_samples=2):
        super(VAKTDataset, self).__init__()
        """Initialize a VAKTDataset class
        
        :param group: A Group DataFrame by user_id from raw dataset
        :param n_concept: The number of concept in dataset
        :param max_seq: The maximum number of input sequence
        :param min_samples: The minimun number of sample to pass into input sequence
        """
        self.n_concept = n_concept
        self.max_seq = max_seq
        self.min_samples = min_samples

        self.samples = {}
        self.user_ids = []

        for user_id in group.index:
            input_, res_ = group[user_id]
            if len(input_) < min_samples:
                continue

            seq_length = len(input_)
            if seq_length > self.max_seq:
                initial = seq_length % self.max_seq
                
                if initial > min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    input = input_[: initial]
                    res = res_[:initial]
                    label = input_[1: initial+1] 
                    self.samples[f"{user_id}_0"] = (input, res, label)
                
                for seq in range(seq_length // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq + 1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    
                    if end >= seq_length:
                        input = input_[start: end - 1]
                        res = res_[start: end - 1]
                        label = input_[start + 1: ]
                    else:
                        input = input_[start: end]
                        res = res_[start: end]
                        label = input_[start + 1: end + 1]

                    self.samples[f"{user_id}_{seq + 1}"] = (input, res, label)
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (input_[: -1], res_[: -1], input_[1:])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        input_, res_, label_ = self.samples[user_id]
        
        input_ = input_.astype(int) + 1
        seq_length = len(input_)

        input = np.zeros(self.max_seq, dtype=int)
        res = np.zeros(self.max_seq, dtype=int)
        label = np.zeros(self.max_seq, dtype=int)

        if seq_length >= self.max_seq:
            input[:] = input_[-self.max_seq:]
            res[:] = res_[-self.max_seq:]
            label[:] = label_[-self.max_seq:]
        else:
            input[-seq_length:] = input_
            res[-seq_length:] = res_
            label[-seq_length:] = label_

        ca =res.astype(int) * self.n_concept + input

        return input, ca, label