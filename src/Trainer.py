import os

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, args, dataloader, my_model):
        self.args = args
        self.my_model = my_model
        self.lr = args.lr
        self.wd = args.wd
        self.epoch = args.epoch
        self.save_path = args.save_path
        self.early_stop = args.early_stop

        self.optimizer = torch.optim.Adam(
            params=self.my_model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        self.train_src = dataloader["train_src"]
        self.train_tgt = dataloader["train_tgt"]  # 目标域的dataloader
        self.test_tgt = dataloader["test_tgt"]

        self.criterion = torch.nn.MSELoss()

        self.results = {"tgt_mae": 999, "tgt_rmse": 999}

        self.path = os.path.join(self.save_path, "train_process.txt")

        # 现添加的
        self.theta = 0.1

    def result_print(self, phase):
        print_str = ""
        for m in ["_mae", "_rmse"]:
            metric_name = phase + m
            print_str += metric_name + ": {:.6f} ".format(self.results[metric_name])

        format_ = f"BEST: {print_str}"
        print(format_)

        with open(self.path, "a") as file:
            file.write(format_)

    @staticmethod
    def model_save(model, path, phase):
        torch.save(model.state_dict(), str(os.path.join(path, f"model_{phase}.pth")))

    @staticmethod
    def train_one_tgt(data_loader, model, criterion, optimizer, epoch, stage):
        tqdm_dataloader = tqdm(data_loader)
        for train_seq, item_id, rating, train_guide in tqdm_dataloader:
            losses = []
            model.train()

            pred = model(train_seq, item_id, train_guide, stage)
            loss = criterion(pred, rating.squeeze(1))
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            show_loss = sum(losses) / len(losses)
            tqdm_dataloader.set_description(
                f"Epoch: {epoch + 1}, Loss: {show_loss:.4f}"
            )

    @staticmethod
    def train_one_src(data_loader, model, criterion, optimizer, epoch, stage):
        tqdm_dataloader = tqdm(data_loader)
        for train_seq, item_id, rating, train_guide in tqdm_dataloader:
            losses = []
            model.train()

            pred = model(train_seq, item_id, train_guide, stage)
            loss = criterion(pred, rating.squeeze(1))
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            show_loss = sum(losses) / len(losses)
            tqdm_dataloader.set_description(
                f"Epoch: {epoch + 1}, Loss: {show_loss:.4f}"
            )

    def update_results(self, mae, rmse, phase):

        if mae < self.results[phase + "_mae"] and rmse < self.results[phase + "_rmse"]:
            self.results[phase + "_mae"] = mae
            self.results[phase + "_rmse"] = rmse
            self.model_save(self.my_model, self.save_path, "mae_rmse")
            flag = 1
        elif mae < self.results[phase + "_mae"]:
            self.results[phase + "_mae"] = mae
            self.model_save(self.my_model, self.save_path, "mae")
            flag = 2
        elif rmse < self.results[phase + "_rmse"]:
            self.results[phase + "_rmse"] = rmse
            self.model_save(self.my_model, self.save_path, "rmse")
            flag = 3
        else:
            flag = 0

        if flag == 1:
            print("Updated mae and rmse")
        elif flag == 2:
            print("Updated mae")
        elif flag == 3:
            print("Updated rmse")

        return flag

    def eval_mae(self, data_loader, model, epoch, stage):

        maes, rmses = [], []
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            tqdm_dataloader = tqdm(data_loader)
            for train_seq, item_id, rating, train_guide in tqdm_dataloader:
                model.eval()
                pred = model(train_seq, item_id, train_guide, stage)
                target = torch.tensor(rating.squeeze(1).tolist()).float()
                predict = torch.tensor(pred.tolist())

                mae = loss(target, predict).item()
                rmse = torch.sqrt(mse_loss(target, predict)).item()
                maes.append(mae)
                rmses.append(rmse)

                aver_mae = sum(maes) / len(maes)
                aver_rmse = sum(rmses) / len(rmses)

                tqdm_dataloader.set_description(
                    f"MAE: {aver_mae:.4f} RMSE: {aver_rmse:.4f}"
                )

            format_ = f"Epoch: {epoch}, MAE: {aver_mae:.4f}, RMSE: {aver_rmse:.4f} \n"
            if epoch == 1:
                with open(self.path, "w") as file:
                    file.write(format_)
            else:
                with open(self.path, "a") as file:
                    file.write(format_)

        return aver_mae, aver_rmse

    def trainer(self):
        stop = 0
        for i in range(self.epoch):
            
            self.train_one_tgt(
                self.train_tgt, self.my_model, self.criterion, self.optimizer, i, "train_tgt"
            )
            self.train_one_src(
                self.train_src, self.my_model, self.criterion, self.optimizer, i, "train_src"
            )
            aver_mae, aver_rmse = self.eval_mae(self.test_tgt, self.my_model, i, "eval")
            flag = self.update_results(aver_mae, aver_rmse, "tgt")

            if flag == 0:
                stop += 1
                if stop < self.early_stop:
                    print(f"Early stopp: {stop} / {self.early_stop}")
                else:
                    print(f"Early stopp: {stop} / {self.early_stop}")
                    break
            else:
                stop = 0
                continue

        self.result_print("tgt")
