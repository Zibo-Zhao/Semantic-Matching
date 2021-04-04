import transformers

MAX_LEN=64
BATCH_SIZE=32
EPOCHS=50
DEVICE="cuda:7"
BERT_PATH="./bert_wwm_model"
TOKENIZER=transformers.BertTokenizer.from_pretrained(BERT_PATH)
MODEL_PATH="./BaseLine/Best"
FINAL_MODEL_PATH="./BaseLine/final"
VOCAB_PATH="./bert_wwm_model/vocab.txt"
ORIGINAL_VOCAB_PATH="./bert_wwm_model/bert_vocab.txt"
BERT_EMBEDDING='./bert_wwm_model/Bert_Embedding.pt'
COUNT_PATH='./bert_wwm_model/counts.json'
K=2
