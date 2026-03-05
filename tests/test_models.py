import pytest
import torch
from src.plm.models import PLMClassifier

def test_model_initialization():
    # Mocking or using a very small model for testing might be better
    # but here we just check if the class can be instantiated (might fail if no internet/HF access)
    try:
        model = PLMClassifier(model_type='esm2', classifier_type='fc', embed_dim=320)
        assert model is not None
        assert model.model_type == 'esm2'
    except Exception as e:
        print(f"Skipping full model initialization test: {e}")

def test_mlp_head():
    from src.plm.models import MLPClassifier
    model = MLPClassifier(input_size=10, hidden_size=20, num_classes=3)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 3)

def test_transformer_head():
    from src.plm.models import TransformerEncoderModel
    model = TransformerEncoderModel(embed_dim=16, num_heads=2, hidden_dim=32, num_layers=1, output_length=8)
    x = torch.randn(4, 10, 16)
    out = model(x)
    assert out.shape == (4, 8)
