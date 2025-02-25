import os
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


class Dataset:
    def __init__(self, args):
        self.ratio = args.ratio
        self.data_root = args.data_root
        self.src_path = args.src_path
        self.tgt_path = args.tgt_path
        self.min_item_interact = args.min_item_interact
        self.min_user_interact = args.min_user_interact
        self.process_data_path = os.path.join(
            self.data_root,
            f"Process_data_{self.src_path[:-6]}_{self.tgt_path[:-6]}.bin",
        )

        self.random_numeral_generator = random.Random(args.seed)

    def read_src_and_tgt(self):
        data_list = []
        columns_name = ["user_id", "item_id", "rating", "timestamp"]
        src_path = str(os.path.join(self.data_root, self.src_path))
        tgt_path = str(os.path.join(self.data_root, self.tgt_path))

        for file_name in (src_path, tgt_path):
            data = pd.read_csv(file_name, sep="	")  # data = pd.read_csv(file_name, sep="	")
            data.columns = columns_name
            data_list.append(data)

        return data_list[0], data_list[1]

    @staticmethod
    def mapper(src_data, tgt_data):
        user_id_dict = {}
        item_id_dict = {}

        src_user_id = set(src_data.user_id)
        src_item_id = set(src_data.item_id)
        tgt_user_id = set(tgt_data.user_id)
        tgt_item_id = set(tgt_data.item_id)

        all_user_id = src_user_id | tgt_user_id
        co_user_id = src_user_id & tgt_user_id
        un_user_id = all_user_id - co_user_id

        user_id_dict.update(dict(zip(co_user_id, range(len(co_user_id)))))
        user_id_dict.update(
            dict(
                zip(
                    un_user_id,
                    range(len(co_user_id), len(co_user_id) + len(un_user_id)),
                )
            )
        )
        item_id_dict.update(dict(zip(src_item_id, range(1, len(src_item_id) + 1))))
        item_id_dict.update(
            dict(
                zip(
                    tgt_item_id,
                    range(
                        len(src_item_id) + 1, len(src_item_id) + len(tgt_item_id) + 1
                    ),
                )
            )
        )

        src_data.user_id = src_data.user_id.map(user_id_dict)
        src_data.item_id = src_data.item_id.map(item_id_dict)
        tgt_data.user_id = tgt_data.user_id.map(user_id_dict)
        tgt_data.item_id = tgt_data.item_id.map(item_id_dict)

        scr_user_num = len(src_data.user_id.unique())
        scr_item_num = len(src_data.item_id.unique())
        tgt_user_num = len(tgt_data.user_id.unique())
        tgt_item_num = len(tgt_data.item_id.unique())

        user_num = scr_user_num + tgt_user_num
        item_num = scr_item_num + tgt_item_num
        co_user_num = len(co_user_id)

        return src_data, tgt_data, user_num, item_num, co_user_num

    def filter_triplets(self, src_data, tgt_data):
        triplets = []

        for data_frame in (src_data, tgt_data):
            user_count = [0, 0]
            item_count = [0, 0]
            while True:
                user_count[0] = len(set(data_frame["user_id"]))
                item_count[0] = len(set(data_frame["item_id"]))
                if self.min_item_interact > 0:
                    item_sizes = data_frame.groupby("item_id").size()
                    good_items = item_sizes.index[item_sizes >= self.min_item_interact]
                    data_frame = data_frame[data_frame["item_id"].isin(good_items)]
                if self.min_user_interact > 0:
                    user_sizes = data_frame.groupby("user_id").size()
                    good_users = user_sizes.index[user_sizes >= self.min_user_interact]
                    data_frame = data_frame[data_frame["user_id"].isin(good_users)]
                user_count[1] = len(set(data_frame["user_id"]))
                item_count[1] = len(set(data_frame["item_id"]))
                if user_count[0] == user_count[1] and item_count[0] == item_count[1]:
                    break
            triplets.append(data_frame)
        return triplets[0], triplets[1]

    def split(self, src_data, tgt_data, co_user_num):
        co_users = list(range(co_user_num))
        tgt_users = set(tgt_data.user_id)
        test_users = set(co_users)

        '''test_users = set(
            self.random_numeral_generator.sample(
                co_users, round(self.ratio[1] * len(co_users))
            )
        )'''

        train_src = src_data
        train_tgt = tgt_data[tgt_data.user_id.isin(tgt_users - test_users)]
        test_tgt = tgt_data[tgt_data.user_id.isin(test_users)]


        train_meta = pd.concat(
            [
                src_data[src_data.user_id.isin(co_users)],
                tgt_data[tgt_data.user_id.isin(set(co_users) - test_users)],
            ]
        )
        print("train_meta shape", train_meta['user_id'])

        train_all = pd.concat([train_src, train_tgt])
        print("train_all shape", train_all['user_id'])

        return train_src, train_tgt, train_all, train_meta, test_tgt

    @staticmethod
    def get_seq(train_all, train_meta):
        seq_dict_list = []
        tqdm.pandas(desc="Make sequence")
        for data_frame in (train_all, train_meta):
            train_user_group = data_frame.groupby("user_id")
            train_seq = train_user_group.progress_apply(
                lambda x: list(x.sort_values(by="timestamp")["item_id"])
            )
            seq_dict_list.append(train_seq)
        return seq_dict_list[0], seq_dict_list[1]

    @staticmethod
    def get_graph(train_meta_seq_dict, co_user_num):
        train_graph = np.zeros([co_user_num, co_user_num])
        for key, value in tqdm(
            train_meta_seq_dict.items(),
            total=len(train_meta_seq_dict),
            desc="Make graph",
        ):
            train_meta_seq_dict.drop(key, axis=0, inplace=True)
            for key_, value_ in train_meta_seq_dict.items():
                edge = len(set(value).intersection(set(value_)))
                if edge == 0:
                    continue
                train_graph[key, key_] += edge
                train_graph[key_, key] += edge
        return train_graph

    def process_data(self):
        src_data, tgt_data = self.read_src_and_tgt()

        src_data, tgt_data = self.filter_triplets(src_data, tgt_data)

        src_data, tgt_data, user_num, item_num, co_user_num = self.mapper(
            src_data, tgt_data
        )

        train_src, train_tgt, train_all, train_meta, test_tgt = self.split(
            src_data, tgt_data, co_user_num
        )

        train_all_seq_dict, train_meta_seq_dict = self.get_seq(train_all, train_meta)



        train_graph = self.get_graph(train_meta_seq_dict, co_user_num)

        dataset = {
            "train_src": train_src,
            "train_tgt": train_tgt,
            "test_tgt": test_tgt,
            "train_all_seq_dict": train_all_seq_dict,
            "train_graph": train_graph,
            "user_num": user_num,
            "co_user_num": co_user_num,
            "item_num": item_num,
            "train_meta_seq_dict": train_meta_seq_dict,
            "train_meta": train_meta,
        }

        with open(self.process_data_path, "wb") as file:
            pickle.dump(dataset, file)

        return dataset

    def get_datasets(self):
        if os.path.exists(self.process_data_path):
            with open(self.process_data_path, "rb") as file:
                dataset = pickle.load(file)
                return dataset
        else:
            dataset = self.process_data()
            return dataset
