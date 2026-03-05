import argparse
import torch
from transformers import RoFormerTokenizer, EsmTokenizer, BertTokenizer
from src.plm import PLMClassifier, build_balanced_dataset, get_dataloaders, train

def get_tokenizer(model_type):
    if model_type == 'antiberta':
        return RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    elif model_type == 'esm2':
        return EsmTokenizer.from_pretrained("facebook/esm-2-t6-8M-UR50D")
    elif model_type == 'biobert':
        return BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Antibody PLM Classifier')
    parser.add_argument('--model', type=str, default='antiberta', choices=['antiberta', 'esm2', 'biobert'])
    parser.add_argument('--classifier', type=str, default='fc', choices=['fc', 'mlp', 'transformer'])
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.ckpt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    config = {
        'model_type': args.model,
        'classifier_type': args.classifier,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'save_path': args.save_path,
        'device': args.device,
        'log_interval': 10,
        'require_improvement': 1000,
    }

    print(f"Loading {args.model} tokenizer...")
    tokenizer = get_tokenizer(args.model)
    
    print(f"Building balanced dataset from {args.dataset}...")
    balanced_data = build_balanced_dataset(args.dataset)
    
    train_loader, test_loader, label_map = get_dataloaders(
        balanced_data, tokenizer, max_length=args.max_len, batch_size=args.batch_size
    )
    
    print(f"Initializing {args.model} model with {args.classifier} head...")
    model = PLMClassifier(
        model_type=args.model, 
        classifier_type=args.classifier, 
        num_classes=len(label_map)
    )

    print("Starting training...")
    train(model, train_loader, test_loader, config)

if __name__ == '__main__':
    main()
