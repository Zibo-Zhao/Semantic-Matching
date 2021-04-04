import config
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

class BERTDataset(Dataset):
    def __init__(self, q1,q2, target):
        self.q1=q1
        self.q2=q2
        self.target=target
        vocab = pd.read_csv(config.VOCAB_PATH, names=['token'])
        vocab_dict = {}
        for key, value in vocab.token.to_dict().items():
            vocab_dict[value] = key
        self.vocab=vocab_dict


    def __len__(self):
        return len(self.q1)

    def __getitem__(self, item):
        reverse = random.random()
        if reverse < 0.5:
            q1 = str(self.q2.iloc[item])
            q2 = str(self.q1.iloc[item])
        else:
            q1 = str(self.q1.iloc[item])
            q2 = str(self.q2.iloc[item])

        inputs=config.TOKENIZER.encode_plus(
            q1,
            q2,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        ids, masked_label=self.random_mask(ids, self.vocab)
        masked_label=[self.target[item]+5]+masked_label        
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]



        return{
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(int(self.target[item]), dtype=torch.long),
            'masked_label': torch.tensor(masked_label, dtype=torch.long)
        }

    def random_mask(self, ids, vocab):
        output_label = []
        for index, vocab_token in enumerate(ids):
            if index==0: continue
            if vocab_token !=vocab['[SEP]'] and vocab_token!=0:
                prob=random.random()
                if prob<0.15:
                    if prob < 0.15 * 0.8: ids[index] = vocab['[MASK]']
                    elif prob < 0.15 * 0.9: ids[index] = random.randrange(8, len(vocab))
                    output_label.append(vocab_token)
                else: output_label.append(0)
            elif vocab_token ==vocab['[SEP]']:
                output_label.append(3)
            else:
                output_label.append(0)
        return ids, output_label



