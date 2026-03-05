import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def build_balanced_dataset(csv_path, label_col='BINDING', seq_col='SEQUENCE_AA', random_state=42):
    data = pd.read_csv(csv_path)
    
    # Determine the minimum count among the labels to balance the dataset
    min_count = data[label_col].value_counts().min()

    # Sample `min_count` rows for each category
    classes = data[label_col].unique()
    balanced_data = pd.concat([
        data[data[label_col] == cls].sample(n=min_count, random_state=random_state)
        for cls in classes
    ])

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return balanced_data

def get_dataloaders(balanced_data, tokenizer, max_length, batch_size=32, test_size=0.2, random_state=42):
    sequences = balanced_data["SEQUENCE_AA"].tolist()
    # Note: Some models (like AntiBERTa) might need spaces between amino acids
    # We can handle that in the training script or here.
    
    # Mapping labels to integers
    label_map = {label: i for i, label in enumerate(sorted(balanced_data["BINDING"].unique()))}
    labels = balanced_data["BINDING"].map(label_map).tolist()

    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )

    train_dataset = SequenceDataset(train_seqs, train_labels, tokenizer, max_length)
    test_dataset = SequenceDataset(test_seqs, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_map
