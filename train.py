import yaml

from utils import *
from dataset import *

from sklearn.model_selection import train_test_split
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm


train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(degrees=10, translate=(0.25, 0.25), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


def train(opt, model, hyper_parameter, path_dir):
    device ='cuda' if torch.cuda.is_available() else 'cpu'

    train_data, valid_data = train_test_split(get_train_data(), test_size=0.2)

    n_epochs = hyper_parameter['n_epochs']
    patience = hyper_parameter['patience']
    batch_size = hyper_parameter['batch_size']
    learning_rate = hyper_parameter['learning_rate']
    weight_decay = hyper_parameter['weight_decay']

    train_loss_function = CutMixCrossEntropyLoss(True)
    valid_loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    best_acc = 0
    stale = 0
    counter = 0
    
    train_dataloader = DataLoader(CutMix(train_valid_dataset(train_data, tfm = train_transform), num_class=12, beta=1.0, prob=0.5, num_mix=2), batch_size = batch_size, num_workers=0)

    valid_dataloader = DataLoader(train_valid_dataset(valid_data, tfm = test_transform), batch_size = batch_size, num_workers=0)

    train_loss_epoch = []
    valid_loss_epoch = []
    valid_accs_epoch = []

    for epoch in range(0,n_epochs):
        counter += 1
        model = model.to(device)
        model.train()
        train_loss = []
    #     train_accs=[]
        for batch in tqdm(train_dataloader):
            imgs, label = batch
    #         print(imgs.shape,label.shape)
            imgs = imgs.to(device)
            label = label.to(device)

            logits = model(imgs)

            loss = train_loss_function(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #         acc = (logits.argmax(dim=-1) == label.to(device)).float().mean()

            train_loss.append(loss.item())
    #         train_accs.append(acc)

        scheduler.step()

        train_loss = sum(train_loss)/len(train_loss)

        train_loss_epoch.append(train_loss)
    #     train_accs = sum(train_accs) / len(train_accs)

    #     train_accs_epoch.append(train_accs)

        print(f"[ Train | {epoch+1}/{n_epochs} ] loss = {train_loss:.5f}")

        print('Starting validation')   

        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_dataloader):
            imgs,label = batch

            imgs = imgs.to(device)
            label = label.to(device)

            with torch.no_grad():
                logits = model(imgs)

            loss = valid_loss_function(logits, label)

            acc = (logits.argmax(dim=-1) == label.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accs = sum(valid_accs) / len(valid_accs)

        valid_loss_epoch.append(valid_loss)
        valid_accs_epoch.append(valid_accs)

        print(f"[ Valid | {epoch+1}/{n_epochs} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")
        print('--------------------------------------')

        # save models
        if valid_accs > best_acc:
            print(f"Best model found at epoch {epoch+1}, saving model")
            torch.save(model.state_dict(), f"{path_dir}/best.ckpt") 

            with open(f"{path_dir}/config.yaml", 'w') as file:
                yaml.dump(utils_function.r_yaml(opt.hyp), file)

            best_acc = valid_accs
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break 
    
    utils_function.draw_train_valid_loss(counter, valid_accs_epoch, train_loss_epoch, valid_loss_epoch, path_dir)
      
    utils_function.draw_confusion_matrix(valid_dataloader, path_dir)