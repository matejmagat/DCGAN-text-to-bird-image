from pathlib import Path
from typing import Any, List, Tuple
import random
import numpy as np

from networkx.algorithms.operators.binary import difference

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from torchvision.utils import Image, save_image
from image_transform import *

from bert_encoder import BERTWrapper


def create_embeddings(caption_root:str, max_length=512, device="cpu"):
    bert = BERTWrapper(max_length=max_length, device=device)
    caption_root = Path(caption_root)
    embeddings_root = caption_root.parent / 'embeddings'
    if not caption_root.exists():
        raise FileNotFoundError(f"Caption root not found: {caption_root}")
    if not embeddings_root.exists():
        embeddings_root.mkdir(parents=True, exist_ok=True)

    for folder in tqdm(caption_root.iterdir()):
        if not embeddings_root.joinpath(folder.name).exists():
            embeddings_root.joinpath(folder.name).mkdir(parents=True, exist_ok=True)
        for caption_path in folder.iterdir():
            with open(caption_path, 'r') as f:
                caption = f.read()

            embedding_path = embeddings_root / folder.name
            embedding = bert(caption)

            torch.save(embedding.contiguous().clone(), embedding_path / f"{caption_path.stem}.pt")




class CUBDataset(Dataset):
    def __init__(self, image_root:str, embeddings_root:str,
                 subset="train", split:Tuple[float, float, float]=(0.7, 0.15, 0.15), seed:int=1234,
                 image_transform=lambda x: x, device="cpu"):

        super().__init__()
        if subset == "train":
            desc = "Loading train set"
        elif subset == "val":
            desc = "Loading validation set"
        elif subset == "test":
            desc = "Loading test set"
        else:
            raise ValueError(f"Unknown subset: {subset}")

        self.image_root = Path(image_root)
        self.embeddings_root = Path(embeddings_root)
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        if not self.embeddings_root.exists():
            raise FileNotFoundError(f"Embedding root not found: {self.embeddings_root}")

        self.device = device

        self.image_transform = image_transform

        self.sample_paths = []

        for folder_path in tqdm(self.image_root.iterdir(), desc=desc):
            for image_path in folder_path.iterdir():
                self.sample_paths.append(folder_path.name + "/" + image_path.stem)

        rng = np.random.default_rng(seed)
        dataset_size = len(self.sample_paths)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        train_size = int(split[0] * dataset_size)
        val_size = int(split[1] * dataset_size)

        if subset == "train":
            indices = indices[:train_size]
        elif subset == "val":
            indices = indices[train_size:train_size+val_size]
        else: # subset == "test"
            indices = indices[train_size+val_size:]

        self.sample_paths:List[str] = [self.sample_paths[i] for i in indices]
        print(f"Using {len(self.sample_paths)} samples for {subset} subset")

    def pick_random(self, sample_path: str):
        sample_class = Path(sample_path).parent.name
        while True:
            random_path = np.random.choice(self.sample_paths)
            if Path(random_path).parent.name != sample_class:
                break

        return random_path

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        random_path = self.pick_random(sample_path)

        image_path = self.image_root / f"{sample_path}.jpg"
        embeddings_path = self.embeddings_root / f"{sample_path}.pt"

        wrong_image_path = self.image_root / f"{random_path}.jpg"

        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        wrong_image = Image.open(wrong_image_path).convert('RGB')
        wrong_image = self.image_transform(wrong_image)

        embedding = torch.load(embeddings_path, map_location=torch.device(self.device))

        return image, embedding, wrong_image


def get_cub_dataloader(image_root, embeddings_root, batch_size, split=(0.7, 0.15, 0.15), seed=1234,
                       image_size=64, num_workers=0, device="cpu") -> dict[str, DataLoader[Any]]:

    train_set = CUBDataset(
        image_root=image_root,
        embeddings_root=embeddings_root,
        image_transform=get_image_transform(image_size),
        device=device,
        seed=seed,
        split=split,
        subset="train",
    )

    val_set = CUBDataset(
        image_root=image_root,
        embeddings_root=embeddings_root,
        image_transform=get_image_transform(image_size),
        device=device,
        seed=seed,
        split=split,
        subset="val",
    )

    test_set = CUBDataset(
        image_root=image_root,
        embeddings_root=embeddings_root,
        image_transform=get_image_transform(image_size),
        device=device,
        seed=seed,
        split=split,
        subset="test",
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    if len(val_set) > 0:
        val_dataloader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        val_dataloader = None

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }




if __name__ == '__main__':

    dataloaders = get_cub_dataloader(image_root='/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/images',
                                    embeddings_root='/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/embeddings',
                                    batch_size=32)

    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']

    print(len(train_dataloader) + len(test_dataloader))

    for image, embedding, wrong_image, wrong_embedding in tqdm(train_dataloader):
        print(image.shape, embedding.shape, wrong_image.shape, wrong_embedding.shape)

        break





