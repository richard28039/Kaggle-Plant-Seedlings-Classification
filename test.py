import numpy as np
import pandas as pd
from utils import *

from dataset import test_dataset

import torch 
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def test(model, hyper_parameter, path_dir):
    
    batch_size = hyper_parameter['batch_size']

    test_dataloader = DataLoader(test_dataset('dataset/test/', tfm = test_transform), batch_size=batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_model = model.to(device)
    best_model.load_state_dict(torch.load(f'{path_dir}/best.ckpt'))
    best_model.eval()
    
    img_name = []
    prediction = []
    
    with torch.no_grad():
        for imgs,img_n in tqdm(test_dataloader):
            imgs = imgs.to(device)
            label_p = best_model(imgs)
            label_predict = np.argmax(label_p.cpu().data.numpy(), axis=1)
            img_name.append(img_n)
            prediction += label_predict.squeeze().tolist()

    name_flatten = [element for sublist in img_name for element in sublist]
    name_flatten = [item.split('/')[2] for item in name_flatten]

    preds = []

    _, num_to_class = utils_function.class_to_num()

    for i in prediction:
        preds.append(num_to_class[i])

    df = pd.DataFrame({"file":name_flatten, "species":preds})
    df.to_csv(f'{path_dir}/submission.csv', index = False)