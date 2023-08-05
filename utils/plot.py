import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from models.resnet import CustomizedResNet

import torch
from torchvision import models

from tqdm import tqdm


def draw_train_valid_loss(epoch, valid_accs, train_loss, valid_loss, path_dir):
    
    epochs=np.arange(0,epoch,1)
    
    if isinstance(valid_accs,np.ndarray):
        pass
    else:
        for i in range(len(valid_accs)):
            valid_accs[i]=valid_accs[i].cpu().numpy()

    plt.figure(figsize=(15, 15))
    plt.plot(epochs, train_loss,'-', label='Training Loss')
    plt.plot(epochs, valid_loss,'m--', label='Validation Loss')
#     plt.plot(epochs, train_accs_epoch[fold],'g-.', label='Training acc')
    plt.plot(epochs, valid_accs, 'r:',label='Validation acc')

    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss/accs Curves'+'\n'+"   "+f"valid_accs:{max(valid_accs):.2f}"+'\n'+f"train_loss:{min(train_loss):.2f}"+"   "+f"valid_loss:{min(valid_loss):.2f}")

    plt.legend(['train loss', 'valid loss','train acc' ,'valid acc'])
    plt.grid(True)
    plt.savefig(f'{path_dir}/train_valid_loss.png')


def draw_confusion_matrix(valid_dataloader, path_dir):

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    layers_to_freeze = []
    pretrained_resnet50 = models.resnet50(pretrained=True)
    model = CustomizedResNet(pretrained_resnet50, num_classes=12, freeze_layers=layers_to_freeze)
    best_model = model.to(device)
    best_model.load_state_dict(torch.load(f'{path_dir}/best.ckpt'))
    best_model.eval()
    label = []
    prediction = []
    with torch.no_grad():
        for imgs, lab in tqdm(valid_dataloader):
            imgs = imgs.to(device)
            label_p = best_model(imgs)
            label_predict = np.argmax(label_p.cpu().data.numpy(), axis=1)
            label += lab.cpu().squeeze().tolist()
            prediction += label_predict.squeeze().tolist()

    cm = confusion_matrix(label, prediction)

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f'{path_dir}/confusion_matrix.png')

