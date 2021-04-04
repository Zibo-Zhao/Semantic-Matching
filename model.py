import transformers
import torch.nn as nn
import config
import torch

class BERT_wmm(nn.Module):
     def __init__(self, keep_tokens):
         super(BERT_wmm,self).__init__()
         self.bert=transformers.BertModel.from_pretrained(config.BERT_PATH)
         self.fc=nn.Linear(768,768)
         self.layer_normalization=nn.LayerNorm((64, 768))
         # self.bert_drop=nn.Dropout(0.2)
         self.out=nn.Linear(768,6932)


         if keep_tokens is not None:
             self.embedding = nn.Embedding(6932, 768)
             weight = torch.load(config.BERT_EMBEDDING)
             weight = nn.Parameter(weight['weight'][keep_tokens])
             self.embedding.weight = weight
             self.bert.embeddings.word_embeddings = self.embedding
             print(weight.shape)


     def forward(self, ids, mask, token_type_ids):
         out1, _=self.bert(
             ids,
             attention_mask=mask,
             token_type_ids=token_type_ids,
             return_dict=False
         )
         # mean pooling
         # max pooling
         # concat
         # bert_output=self.bert_drop(out1)
         output=self.fc(out1)
         layer_normalized=self.layer_normalization(output)
         final_output=self.out(layer_normalized)
         return final_output
     
