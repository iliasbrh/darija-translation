import torch
from torch.utils.data import Dataset

device = "cuda"

class LanguageTranslationDataset(Dataset):
    def __init__(self, data, T, src_tokenizer, tgt_tokenizer):
        super().__init__()
        
        self.data = data
        
        self.T = T
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        special_ids = [
            src_tokenizer.pad_token_id,
            src_tokenizer.bos_token_id,
            src_tokenizer.eos_token_id,
            src_tokenizer.cls_token_id,
            src_tokenizer.sep_token_id]
        self.ignore_ids = torch.tensor([x for x in special_ids if x is not None], dtype=torch.long)
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        # src_tokens = self.data[idx, 0].clone()
        src_tokens = self.data[idx, 0]
        tgt_tokens = self.data[idx, 1] 
        
        # various data augmentation techniques (mostly poorly working ones) 
        """
        # random swap
        valid_mask = ~torch.isin(src_tokens, self.ignore_ids)
        valid_indices = torch.where(valid_mask)[0]
        if torch.rand(1).item() < 0.10:
            if len(valid_indices) > 5: # to not mess up short sentences
                idx1, idx2 = valid_indices[torch.randperm(len(valid_indices))[:2]]
                temp = src_tokens[idx1].item()
                src_tokens[idx1] = src_tokens[idx2]
                src_tokens[idx2] = temp

        # word dropout
        probabilities = torch.rand(src_tokens.shape)
        drop_mask = (probabilities < 0.1) & (src_tokens != self.src_tokenizer.pad_token_id)
        src_tokens[drop_mask] = 1 # 1 = padding token
        """

        expected_output = torch.ones((tgt_tokens.size(0)), dtype=torch.long) # ones since the padding token for roberta pretrained is 1
        expected_output[:-1] = tgt_tokens[1:]

        item = {"src":src_tokens, "tgt":tgt_tokens, "output":expected_output}
        
        return item
        
