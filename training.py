from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np

def loss_fn(outputs, targets,teacher_outputs, outputs_mask):
    criterion=nn.CrossEntropyLoss()
    criterion1=nn.MSELoss()
    loss = criterion(outputs[outputs_mask].view(-1, 6932), targets[outputs_mask].view(-1))\
           +criterion1(outputs[outputs_mask], teacher_outputs[outputs_mask])
    return loss

def train_fn(data_loader, model, optimizer, device, model_teacher, Dis_flag):
    model.train()
    losses=[]
    criterion  = nn.CrossEntropyLoss()
    description=tqdm(enumerate(data_loader), total=len(data_loader))
    for bi, d in description:
        ids = d['ids']
        token_type_ids = d["token_type_ids"]
        mask = d['mask']
        targets = d['masked_label']
        # targets=targets.unsqueeze(1)

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        outputs_mask=(targets!=0)


        optimizer.zero_grad()
        outputs=model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        if Dis_flag:
            outputs_teacher=model_teacher(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss=loss_fn(outputs, targets, outputs_teacher, outputs_mask)
        else:
            loss = criterion(outputs[outputs_mask].view(-1, 6932), targets[outputs_mask].view(-1))
        losses.append(loss.cpu().detach().numpy())
        description.set_description(f'loss:{np.mean(losses):.4f}')
        loss.backward()
        optimizer.step()
       

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            token_type_ids = d["token_type_ids"]
            mask = d['mask']
            targets = d['masked_label']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            targets = targets[:, 0] - 5
            prediction = outputs[:, 0, 5:7].cpu().detach().numpy()
            prediction = prediction[:, 1] / (prediction.sum(axis=1) + 1e-8)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(prediction.tolist())

    return (fin_outputs, fin_targets)
