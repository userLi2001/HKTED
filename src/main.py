from Dataloader import Dataloader

from Dataset import Dataset
from MyModel import MyModel
from Parameter import Prepare
from Trainer import Trainer

if __name__ == "__main__":
    args = Prepare()
    dataset = Dataset(args).get_datasets()
    dataloader = Dataloader(args, dataset).get_dataloaders()
    my_model = MyModel(args, dataset)
    Trainer(args, dataloader, my_model).trainer()
