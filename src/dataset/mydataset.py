import os
import pandas as pd
from PIL import Image

import torch
from torch.utils import data

from src.utils.data_utils import get_vocabularies


# Build dataset from a csv file.
class LicensePlatesDataset(data.Dataset):
    # The csv file should have the following header: (img_path, label)
    def __init__(self, root_dir, csv_file, max_len, split, transform=None):
        super(LicensePlatesDataset, self).__init__()
        # Define root dir
        self.root_dir = root_dir
        # Build data frame
        df = pd.read_csv(csv_file)
        self.data =  df[df['split'] == split]
        # Define license plate max number of characters
        self.max_len = max_len
        # Define transformation
        self.transform = transform
        # Get vocabulaies
        self.char2id, self.id2char = get_vocabularies()

        # Define special characters
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Define image path
        img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
        # Open image with PIL
        image = Image.open(img_path)
        # Define plate id (plate number)
        plate_id = self.data.iloc[index, 1]

        # Initialize label as an array full of padding
        label = torch.full((self.max_len,), self.char2id[self.PADDING], dtype=torch.int8)
        # Replace i-th position of label with char id; UNKNOWN if char not in vocabulary
        for i, char in enumerate(plate_id):
            if char in self.char2id.keys():
                label[i] = self.char2id[char]
            else:
                label[i] = self.char2id[self.UNKNOWN]
        # Add EOS (end of string)
        label[len(plate_id)] = self.char2id[self.EOS]

        # Define label length
        label_len = torch.tensor(len(plate_id)+1)

        # Apply transformations (if needed)
        if self.transform:
            image = self.transform(image)

        # Return image, label, and label_len
        return image, label, label_len
