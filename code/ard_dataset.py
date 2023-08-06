from torch.utils.data import Dataset 
from utils import read_json

class ARDDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.data = read_json(path)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample["word"], sample["gloss"], sample["electra"], sample["sgns"]
    
    def __len__(self):
        return len(self.data)
