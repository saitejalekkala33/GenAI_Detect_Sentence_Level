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

### Model Architecture
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
### Datasets
- **Human ChatGPT Comparison Corpus (HC3-English)**: Human and machine-generated responses from various domains.
- **M4GT-Bench Task 3 Dataset**: Multidomain dataset containing ChatGPT and LLaMA-generated texts.

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
