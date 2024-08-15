from model import LORAModel, Model, LORAModelMod, AdapterModel, VPTModel, VPTLORAModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn 
import pandas as pd
from torchvision import transforms
import logging
import sys
from tqdm import tqdm 
from PIL import Image
import os
from os.path import join
import argparse
import json
from utils_own import get_results

class dataset(Dataset):
    def __init__(self, df):
        self.images = df['image'].values
        self.labels = df['label'].values
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.images[idx])), self.labels[idx]

def main(args):
    csv = pd.read_csv(f'Datasets/{args["dataset"]}_val.csv')
    device = args['device']
    exp_name = join(args['folder'], args['exp_name'])
    # exp_name = 'Experiments\Experiments_colab\\baseline_all_three_seed_16'
    # exp_name = 'Experiments\Experiments_colab\lora_all_three_seed_16'
    # exp_name = 'Experiments\\baseline_adam_lr0.005_imagenet'

    

    data = dataset(csv)
    dataloader = DataLoader(data, shuffle = True, batch_size=64)

    if args['dataset'] == 'AID':
        num_classes = 30
    elif args["dataset"] == 'PatternNet':
        num_classes = 38
    elif args["dataset"] == 'EuroSat':
        num_classes = 10
    elif args['dataset'] == 'UCMerced_LandUse':
        num_classes = 21
    
    if args['model_type'] == 'base_imagenet':
        model = Model(num_classes = num_classes).to(device)
    elif args['model_type'] == 'lora':
        model = LORAModel(num_classes = num_classes).to(device)
    elif args['model_type'] == 'lora_mod':
        model = LORAModelMod(num_classes = num_classes).to(device)
    elif args['model_type'] == 'adapter':
        model = AdapterModel(num_classes = num_classes).to(device)
    elif args['model_type'] == 'vpt':
        model = VPTModel(num_classes = num_classes).to(device)
    elif args['model_type'] == 'vptlora':
        model = VPTLORAModel(num_tokens = args['num_tokens'],num_classes=num_classes).to(device)
    

    epoch = model.load_model(exp_name, latest = False)

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    preds = []
    with torch.no_grad():
        num_runs = 1
        running_precision = running_recall = running_f1 = running_acc = 0
        for i in range(num_runs):
            for idx,data in enumerate(tqdm(dataloader)):
                image, label = data
                image = image.cuda()
                label = label.cuda()
                _,predictions,_,_ = model(image,torch.zeros_like(image))
                total += len(label)
                correct += sum(predictions.softmax(dim = -1).argmax(dim = -1) == label)
                all_labels.append(label)
                preds.append(predictions)

            print('Accuracy: ', (correct/total).item()*100)

            precision, recall, f1, acc = get_results(torch.cat(preds), torch.cat(all_labels),verbose = False)
            running_precision += precision
            running_recall += recall
            running_f1 += f1
            running_acc += acc

    return {'Precision':running_precision/num_runs, 'Recall':running_recall/num_runs,'F1 Score':running_f1/num_runs,'Accuracy':running_acc/num_runs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for your script")
    parser.add_argument('--model_type', choices=['base_imagenet', 'lora', 'lora_mod', 'adapter'], help='Model type', default = 'base_imagenet')
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='Device (e.g., cuda or cpu)')
    parser.add_argument('--seed', type=int, default=16, help='Random seed')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--folder', type=str, default='Experiments_all/Experiments_AID')
    parser.add_argument('--dataset', choices=['AID', 'PatternNet','EuroSat','UCMerced_LandUse'], default = 'AID')
    
    args = vars(parser.parse_args())
    if not args['all']:
        main(args)
    else:
        all_results = {}
        for exp in os.listdir(args['folder']):
            if os.path.isdir(join(args['folder'],exp)):
                dic = json.load(open(join(args['folder'],exp,'params.json'), 'r'))
                dic['folder'] = args['folder']
                dic['dataset'] = args['dataset']
                all_results[exp] = main(dic)
        
        
        all_results_sorted = dict(sorted(all_results.items(), key=lambda x: x[1]['Accuracy'],reverse = True))
        with open(f'{args["folder"]}/validation_results.json', 'w') as f:
            json.dump(all_results_sorted, f)




    