# GenAI Detect Sentence_Level Segmentation
<div align="center">
  <img src="Images/ACL_SRW.png" alt="GenAI Detect Sentence-Level Segmentation" width="600"/>
</div>

## Sentence-Level Segmentation of AI-Generated and Human Text

### Overview
This repository contains the implementation of a hybrid Transformer-NN-CRF model designed for sentence-level segmentation to distinguish between AI-generated and human-written text. The model aims to address the limitations of document-level classification by identifying transition boundaries between human and machine text at the sentence level. Along with the hybrid models we also provided the code for normal implementations for this sentence level segmentation namely Neural Network CRF, and Transformer CRF.

### Key Features
- Combines Transformer-based models, NN, and Conditional Random Fields (CRF) for sequence labeling.
- Incorporates techniques such as Layer-wise Learning Rate Decay, Dynamic Dropout, and Xavier Initialization.
- Supports multiple transformer backbones like BERT, RoBERTa, DistilBERT, DeBERTa, and ModernBERT.
- Optimized for detecting subtle transitions between human and AI-generated text.

### Datasets
- **AAAI Dataset**: Dataset from the paper *Towards Automatic Boundary Detection for Human-AI Collaborative Hybrid Essay in Education* [Paper Link (<https://arxiv.org/abs/2307.12267>)
- **M4GT-Bench Task 3 Dataset**: Multidomain dataset containing ChatGPT and LLaMA-generated texts.
- **Human ChatGPT Comparison Corpus (HC3-English)**: Human and machine-generated responses from various domains.
- Datasets are available in the following drive link: [Google Drive Link](<https://drive.google.com/drive/folders/1_de-VwGj5mJNruBWbiXtwSX4tqsKPnW4?usp=sharing>)

  
### Models Architecture
```python
class DeBERTaBiGRUCRFTagger(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=128):
        super(DeBERTaBiGRUCRFTagger, self).__init__()
        self.num_labels = num_labels
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        transformer_hidden_size = self.deberta.config.hidden_size
        self.gru = nn.GRU(transformer_hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        gru_output, _ = self.gru(sequence_output)
        logits = self.classifier(gru_output)
        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
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

class BiGRUCRFTagger(nn.Module):
    def __init__(self, input_dim, num_labels, embedding_dim=768, hidden_dim=512, num_layers=2, dropout=0.3):
        super(BiGRUCRFTagger, self).__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
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
        if input_ids.numel() == 0:
            raise ValueError("Input tensor is empty.")
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)
        gru_out, _ = self.gru(embedded)
        logits = self.hidden2tag(gru_out)
        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            return predictions
```


### Results
- **AAAI Dataset:** DeBERTa-BiGRU-CRF acheived the best performance with highest Cohen's Kappa Score 97.02
- **M4GT Dataset:** DeBERTa-BiGRU-CRF achieved the best performance with an MAE below 10 (8.47).
- **HC3 Dataset:** DeBERTa-BiLSTM-CRF demonstrated the lowest MAE of 6.

### Limitations
- Vulnerable to adversarial attacks involving syntactic and semantic perturbations.

### Future Work
- Implement adversarial training for robust generalization.
- Enhance model scalability for real-world applications
