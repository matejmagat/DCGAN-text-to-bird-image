import torch
from transformers import BertTokenizer, BertModel


class BERTWrapper:
    def __init__(self, model_name='bert-base-uncased', device='cpu', max_length=512):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.device = device
        self.bert_model.to(self.device)
        self.bert_model.eval()

    def __call__(self, caption):
        inputs = self.tokenizer(
            caption,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        # Return the [CLS] embedding as a 1D tensor
        return outputs.last_hidden_state[:, 0, :].squeeze(0)