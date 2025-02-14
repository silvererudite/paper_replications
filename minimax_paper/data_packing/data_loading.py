from torch.utils.data import Dataset


class DataPackDataset(Dataset):
    def __init__(self, tokenizer, source_dir):
        # stream the dataset from memory
        # load variable lengths of the dataset
        # tokenize the text
        # convert the tokens to tensors

        self.text_chunks = ...
        self.text_tensors = ...

    def __len__(self):
        return len(self.text_tensors)

    def __getitem__(self, idx):
        ...

    def pad_preprocess(self):
        ...

    def data_pack_preprocess(self):
        ...

    def tokenize(self):
        ...

# I want a way to convert to and from tokenizer
# profile torch for memory, time, token coverage
