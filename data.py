import torch
import random
from PIL import Image
from collections import Counter
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

class ThumbNail(Dataset):
    def __init__(
            self,
            clickbait_path,
            non_clickbait_path):
        super().__init__()
        data = []
        label = []
        for img in clickbait_path.glob("*.jpg"):
            data.append(Image.open(img))
            label.append(0)
        for img in non_clickbait_path.glob("*.jpg"):
            data.append(Image.open(img))
            label.append(1)
        # Shuffle both label and image
        temp = list(zip(data, label))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        self.data, self.label = list(res1), list(res2)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    #resize_shape = (32, 32)
    resize_shape = (360, 640) # Place holder
    transform = T.Compose(
        [#T.RandomHorizontalFlip(),
        T.Resize(resize_shape),
        T.ToTensor(),])
    image_batch = [transform(image) for image, label in batch]
    label_batch = [label for image, label in batch]
    image_batch = torch.stack(image_batch, dim=0)
    label_batch = torch.tensor(label_batch)
    #print(image_batch.shape, label_batch.shape)
    return image_batch, label_batch

def batch_cifar(batch):
    from torchvision.transforms import transforms as T
    transform = T.ToTensor()
    image_batch = [transform(image) for image, label in batch]
    label_batch = [label for image, label in batch]
    image_batch = torch.stack(image_batch, dim=0)
    label_batch = torch.tensor(label_batch)
    #print(image_batch.shape, label_batch.shape)
    return image_batch, label_batch

def split_dataset(dataset):
    # default split ration: 0.7:0.1:0.2
    N = len(dataset)
    split = [int(0.7*N), int(0.1*N), N - int(0.7*N) - int(0.1*N)]
    return random_split(dataset, split)

def class_balance_check(dataset):
    label = [l for m, l in dataset]
    c = Counter()
    c.update(label)
    print(c)

        













