import torch
from torch.utils.data import DataLoader, Dataset


class Dataloader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        self.batch_size = args.batch_size

    def get_dataloaders(self):
        train_src_loader = DataLoader(
            TrainDataset_src(
                self.args,
                self.dataset["train_src"],
                self.dataset["train_all_seq_dict"],
                self.dataset["train_meta_seq_dict"],
            ),
            self.batch_size,
            True,
        )

        train_tgt_loader = DataLoader(
            TrainDataset_tgt(
                self.args,
                self.dataset["train_tgt"],
                self.dataset["train_meta"],
                self.dataset["train_all_seq_dict"],
                self.dataset["train_meta_seq_dict"],
            ),
            self.batch_size,
            True,
        )

        test_tgt_loader = DataLoader(
            TestDataset(
                self.args, self.dataset["test_tgt"], self.dataset["train_all_seq_dict"]
            ),
            self.batch_size,
            False,
        )

        data_loader = {
            "train_src": train_src_loader,
            "train_tgt": train_tgt_loader,
            "test_tgt": test_tgt_loader,
        }
        return data_loader


class TrainDataset_tgt(Dataset):
    def __init__(self, args, train_data_tgt, train_data_src, train_seq_dict, train_seq_guide_dict):
        self.device = args.device
        self.max_length = args.max_length
        self.train_data1 = train_data_tgt
        self.train_data2 = train_data_src
        self.train_seq_dict = train_seq_dict
        self.train_seq_guide_dict = train_seq_guide_dict
        self.train_data1 = self.train_data1.reset_index(drop=True)
        self.train_data2 = self.train_data2.reset_index(drop=True)

    def __len__(self):
        return len(self.train_data1)

    def __getitem__(self, index):
        """
         下面是我加的
        """
        if index >= len(self.train_data1):
            print(f"Index {index} exceeds train_data1 length {len(self.train_data1)}")
        data1 = self.train_data1.iloc[index]
        user_id1= data1["user_id"]
        item_id = data1["item_id"]
        rating = data1["rating"]
        train_seq = self.train_seq_dict.get(user_id1, []) 
        if len(train_seq) == 0:
            print(f"Warning: No sequence found for user_id {user_id1}")
        train_seq = train_seq[-self.max_length :]
        pad_length = self.max_length - len(train_seq)
        train_seq = [0] * pad_length + train_seq
        index2 = index % len(self.train_data2)  # 使用取余操作避免越界
        data2 = self.train_data2.iloc[index2]
        user_id2 = data2["user_id"]
        user_id2 = int(user_id2)
        train_guide = self.train_seq_guide_dict.get(user_id2, [0] * self.max_length)
        if len(train_guide) == 0:
            print(f"Warning: No guide found for user_id {user_id2}")
        train_guide = train_guide[-self.max_length :]
        pad_length = self.max_length - len(train_guide)
        train_guide = [0] * pad_length + train_guide

        item_id = torch.LongTensor([item_id]).to(self.device)
        rating = torch.FloatTensor([rating]).to(self.device)
        train_seq = torch.LongTensor(train_seq).to(self.device)
        train_guide = torch.LongTensor(train_guide).to(self.device)

        return train_seq, item_id, rating, train_guide

class TrainDataset_src(Dataset):
    def __init__(self, args, train_data, train_seq_dict, train_seq_guide_dict):
        self.device = args.device
        self.max_length = args.max_length
        self.train_data = train_data
        self.train_seq_dict = train_seq_dict
        self.train_seq_guide_dict = train_seq_guide_dict

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        data = self.train_data.iloc[index]
        user_id = data["user_id"]
        item_id = data["item_id"]
        rating = data["rating"]
        train_seq = self.train_seq_dict.get(user_id, [])
        train_seq = train_seq[-self.max_length :]
        pad_length = self.max_length - len(train_seq)
        train_seq = [0] * pad_length + train_seq
        train_guide = [0] * self.max_length

        item_id = torch.LongTensor([item_id]).to(self.device)
        rating = torch.FloatTensor([rating]).to(self.device)
        train_seq = torch.LongTensor(train_seq).to(self.device)
        train_guide = torch.LongTensor(train_guide).to(self.device)

        return train_seq, item_id, rating, train_guide



class TestDataset(Dataset):
    def __init__(self, args, test_data, train_seq_dict):
        self.device = args.device
        self.max_length = args.max_length
        self.test_data = test_data
        self.train_seq_dict = train_seq_dict

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        data = self.test_data.iloc[index]
        user_id = data["user_id"]
        item_id = data["item_id"]
        rating = data["rating"]

        train_seq = self.train_seq_dict.get(user_id, [])
        train_seq = train_seq[-self.max_length :]
        pad_length = self.max_length - len(train_seq)
        train_seq = [0] * pad_length + train_seq
        
        train_guide = [0] * self.max_length 

        item_id = torch.LongTensor([item_id]).to(self.device)
        rating = torch.FloatTensor([rating]).to(self.device)
        train_seq = torch.LongTensor(train_seq).to(self.device)
        train_guide = torch.LongTensor(train_guide).to(self.device)
         

        return train_seq, item_id, rating, train_guide
