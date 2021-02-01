from torch.utils.data.dataset import Dataset
from COCO.dataset import COCOKeypointDataset

class DatasetManager:
    def __init__(self, dataset_names, train='true'):
        self.coco_dataset = None
        self.main_dataset = None

        for dataset in dataset_names:
            if dataset == "COCOKeypoint":
                self.coco_dataset = COCOKeypointDataset(train)
        
        self.main_dataset = self.coco_dataset

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, index):
        return self.main_dataset[index]

    def collate_fn(self, batch):
        return list(batch)

    def evaluate(self):
        self.main_dataset.evaluate()