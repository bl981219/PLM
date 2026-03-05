import torch
import torch.nn as nn
from transformers import RoFormerModel, EsmModel, BertModel

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, output_length):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_length)

    def forward(self, token_embeddings, attention_mask=None):
        if attention_mask is not None:
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None

        transformer_output = self.transformer_encoder(token_embeddings, src_key_padding_mask=transformer_mask)
        pooled_output = transformer_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output

class PLMClassifier(nn.Module):
    def __init__(self, model_type='antiberta', classifier_type='fc', num_classes=3, 
                 hidden_dim=256, num_heads=8, num_layers=2, output_length=48):
        super(PLMClassifier, self).__init__()
        self.model_type = model_type
        self.classifier_type = classifier_type
        
        if model_type == 'antiberta':
            self.backbone = RoFormerModel.from_pretrained("alchemab/antiberta2")
        elif model_type == 'esm2':
            self.backbone = EsmModel.from_pretrained("facebook/esm-2-t6-8M-UR50D") 
        elif model_type == 'biobert':
            self.backbone = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Automatically infer embed_dim
        embed_dim = self.backbone.config.hidden_size

        if classifier_type == 'fc':
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif classifier_type == 'mlp':
            self.classifier = MLPClassifier(embed_dim, hidden_dim, num_classes)
        elif classifier_type == 'transformer':
            self.transformer = TransformerEncoderModel(embed_dim, num_heads, hidden_dim, num_layers, output_length)
            self.classifier = nn.Linear(output_length, num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if self.classifier_type == 'transformer':
            transformer_output = self.transformer(last_hidden_state, attention_mask)
            logits = self.classifier(transformer_output)
        else:
            pooled_output = last_hidden_state.mean(dim=1)
            logits = self.classifier(pooled_output)

        return logits

    def get_attention_attribution(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions
        # Simple attribution: mean attention across heads of the last layer
        last_layer_attention = attentions[-1]
        cls_attention = last_layer_attention[:, :, 0, :]
        attribution = cls_attention.mean(dim=1)
        return attribution
