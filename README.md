# GenAI Detect Sentence_Level Segmentation
<div align="center">
  <img src="Images/Transformer_BiLSTM_CRF.png" alt="GenAI Detect Sentence-Level Segmentation" width="600"/>
</div>

## Sentence-Level Segmentation of AI-Generated and Human Text

### Overview
This repository contains the implementation of a hybrid Transformer-BiLSTM-CRF model designed for sentence-level segmentation to distinguish between AI-generated and human-written text. The model aims to address the limitations of document-level classification by identifying transition boundaries between human and machine text at the sentence level. Along with the hybrid models we also provided the code for normal implementations for this sentence level segmentation namely Neural Network CRF, and Transformer CRF.

### Key Features
- Combines Transformer-based models, BiLSTM, and Conditional Random Fields (CRF) for sequence labeling.
- Incorporates techniques such as Layer-wise Learning Rate Decay, Dynamic Dropout, and Xavier Initialization.
- Supports multiple transformer backbones like BERT, RoBERTa, DistilBERT, DeBERTa, and ALBERT.
- Optimized for detecting subtle transitions between human and AI-generated text.

### Datasets
- **Human ChatGPT Comparison Corpus (HC3-English)**: Human and machine-generated responses from various domains.
- **M4GT-Bench Task 3 Dataset**: Multidomain dataset containing ChatGPT and LLaMA-generated texts.
- Datasets are available in the following drive link: [Google Drive Link](<https://drive.google.com/drive/folders/1_de-VwGj5mJNruBWbiXtwSX4tqsKPnW4?usp=sharing>)

  
### Models Architecture
```python
class TransformerBiLSTMCRF(nn.Module):
    def __init__(self, transformer_model, hidden_dim, num_labels):
        super(TransformerBiLSTMCRF, self).__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.lstm = nn.LSTM(input_size=self.transformer.config.hidden_size, hidden_size=hidden_dim, num_layers=3, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.activation = nn.ReLU()
        self.crf = torchcrf.CRF(num_labels, batch_first=True)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = transformer_outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        emissions = self.fc(self.activation(lstm_out))
        if labels is not None:
          mask = attention_mask.bool()
          crf_labels = labels.clone()
          crf_labels[crf_labels == -100] = 0
          log_likelihood = self.crf(emissions, crf_labels, mask=mask, reduction='mean')
          return -log_likelihood
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions
```
```python
class TransformerCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TransformerTaggerCRF, self).__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            return torch.tensor(predictions)
```
```python

class RNNCRFTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, embedding_dim=768, num_layers=2, dropout=0.3):
        super(RNNCRFTagger, self).__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.hidden2hidden = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout))
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)
        embedded = embedded * attention_mask.unsqueeze(-1)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.hidden2hidden(lstm_out)
        logits = self.hidden2tag(lstm_out)

        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            padded_predictions = []
            for pred, mask_len in zip(predictions, attention_mask.sum(1).tolist()):
                pad_len = attention_mask.size(1) - len(pred)
                padded_pred = pred + [0] * pad_len
                padded_predictions.append(padded_pred)
            return torch.tensor(padded_predictions, device=input_ids.device)
```


### Results
- **M4GT Dataset:** ALBERT-BiLSTM-CRF achieved the best performance with an MAE below 10.
- **HC3 Dataset:** DeBERTa-BiLSTM-CRF demonstrated the lowest MAE of 6.

### Limitations
- Performance on multiple boundary detection is not optimal.
- Vulnerable to adversarial attacks involving syntactic and semantic perturbations.

### Future Work
- Extend the dataset to cover multiple human boundaries.
- Implement adversarial training for robust generalization.
- Enhance model scalability for real-world applications
