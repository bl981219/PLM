import time
import torch
import torch.optim as optim
from sklearn import metrics
import numpy as np
from datetime import timedelta
from src.plm.loss import CrossEntropyLoss

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def evaluate(model, data_iter, criterion, device, label_names=None):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for batch in data_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if label_names:
        report = metrics.classification_report(labels_all, predict_all, target_names=label_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def train(model, train_loader, test_loader, config):
    start_time = time.time()
    model.to(config['device'])
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    criterion = config.get('criterion', CrossEntropyLoss(config['device']))
    
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    
    model.train()
    for epoch in range(config['num_epochs']):
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}]')
        for batch in train_loader:
            input_ids = batch["input_ids"].to(config['device'])
            attention_mask = batch["attention_mask"].to(config['device'])
            labels = batch["label"].to(config['device'])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % config['log_interval'] == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, test_loader, criterion, config['device'])
                
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config['save_path'])
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config['require_improvement']:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
