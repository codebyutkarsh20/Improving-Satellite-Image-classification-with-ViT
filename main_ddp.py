import builtins
import torch
from torch import nn
import numpy as np
import random
from model import Model, LORAModel, LORAModelMod, AdapterModel, VPTModel, VPTLORAModel
from utils_own import horizontal_flip, horizontal_flip_target, vertical_flip, vertical_flip_target, flip, get_results, set_seed
from data import train_dataset, val_dataset, triplet_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
import json
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import random
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main(args):
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    args = vars(args)
    seed = args['seed']
    set_seed(seed)
    # exp = 'Experiments/baseline_adam_lr0.005_dino_imagenet'
    exp=f'{args["folder"]}/'+args['exp_name']
    if gpu_id == 0:
        writer = SummaryWriter(exp)
        with open(f'{exp}/params.json', 'w') as f:
            json.dump(args, f)
    num_epochs = args['num_epochs']
    device = args['device'] + f':{gpu_id}'
    # trial_4 was changing the attention weights also with attention loss
    # trial 5 was not changing the attention weights with attention loss with a loss weight of 10^3
    # trial 6 was changing only the qkv values in attention blocks with attention loss with a loss weight of 10^3
    # trial 7 was not changing the attention weights with attention loss with a loss weight of 10^4
    # lora - lora adaptation with attention loss of weight 10^4 (lora_2 is rerun)
    # lora_mod - lora adaptation modified with attention loss of weight 10^4 (lora_mod_2 is rerun)
        


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

    model = DDP(model, device_ids=[gpu_id])

    
    if args['use_triplet_loss']:
        train_data = triplet_dataset(csv_path=f'Datasets/{args["dataset"]}_triplet_train.csv')
        val_data = triplet_dataset(csv_path=f'Datasets/{args["dataset"]}_triplet_val.csv')
    else:
        train_data = train_dataset(csv_path=f'Datasets/{args["dataset"]}_train.csv')
        val_data = val_dataset(csv_path=f'Datasets/{args["dataset"]}_val.csv')


    train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = False,sampler=DistributedSampler(train_data))
    val_dataloader = DataLoader(val_data, batch_size = 32, shuffle = False)

    loss_output_fn = nn.CrossEntropyLoss()
    loss_attn_fn = nn.MSELoss()
    loss_triplet_fn = nn.TripletMarginLoss()

    optim = torch.optim.Adam(model.parameters(), lr = args['lr'])

    step_train = 0
    step_val = 0
    resume_epochs = 0
    max_val_acc = 0


    if args['resume'] and gpu_id == 0:
        resume_epochs = model.load_model(f'{exp}')
        model = model.to(device)
        

    for epoch in range(resume_epochs+1, num_epochs+1):
        print()
        print(f'EPOCH {epoch}')
        print()

        train_dataloader.sampler.set_epoch(epoch)
        

        running_train_correct = 0
        total_train_len = 0

        model.train()
        for idx, input_data in enumerate(tqdm(train_dataloader)):
            if args['use_triplet_loss']:
                anc, image_2, labels, type, pos, neg = input_data
            else:
                anc, image_2, labels, type = input_data
            anc = anc.to(device)
            image_2 = image_2.to(device)
            labels = labels.to(device)
            b = len(labels)

            if args['use_triplet_loss']:
                pos = pos.to(device)
                neg = neg.to(device)

            
            self_attn_1, output_1, self_attn_2, output_2 = model(anc, image_2)

            frames = []
            for i in range(b):
                frames.append(flip(self_attn_1[i],type[i]).unsqueeze(0))

            self_attn_1 = torch.cat(frames)
            
            loss_out = (loss_output_fn(output_1, labels) + loss_output_fn(output_2, labels)) # TODO try weighted cross entropy also

            loss_att = loss_attn_fn(self_attn_1, self_attn_2)*args['attn_loss_weight'] 

            if args['use_triplet_loss']:
                anc_features, pos_features, neg_features = model.triplet_forward(anc, pos, neg)
                loss_trip = loss_triplet_fn(anc_features, pos_features, neg_features)*args['triplet_loss_weight']

            loss = 0
            if args['add_cross_entropy']:
                loss += loss_out
            if args['add_attn_loss']:
                loss += loss_att
            if args['add_triplet_loss']:
                loss+=loss_trip

            
            optim.zero_grad()
            loss.backward()
            optim.step()

            # overall_output_1.append(output_1)
            # overall_output_2.append(output_2)   
            # overall_labels.append(labels)
            
            total_train_len += b
            running_train_correct += sum(output_1.softmax(dim = -1).argmax(dim = -1) == labels)

            if step_train%1 == 0 and gpu_id == 0:
                writer.add_scalar('Loss_overall/train', loss.item(), step_train)
                writer.add_scalar('Loss_output/train', loss_out.item(), step_train)
                writer.add_scalar('Loss_attention/train', loss_att.item(), step_train)
                if args['use_triplet_loss']:
                    writer.add_scalar('Loss_triplet/train', loss_trip.item(), step_train)


            # if step_train%10 == 0:
                # print(f'Epoch: {epoch}, Loss Overall: {loss.item()}, Loss Output: {loss_out}, Loss attention: {loss_att}')
                # precision, recall, f1, acc = get_results(torch.cat(overall_output_1), torch.cat(overall_labels),verbose=False)
                # writer.add_scalar('Output_1/precision_train', precision, step_train)
                # writer.add_scalar('Output_1/recall_train', recall, step_train)
                # writer.add_scalar('Output_1/f1_train', f1, step_train)
                # writer.add_scalar('Output_1/accuracy_train', acc, step_train)

                # precision, recall, f1, acc = get_results(torch.cat(overall_output_2), torch.cat(overall_labels),verbose=False)
                # writer.add_scalar('Output_2/precision_train', precision, step_train)
                # writer.add_scalar('Output_2/recall_train', recall, step_train)
                # writer.add_scalar('Output_2/f1_train', f1, step_train)
                # writer.add_scalar('Output_2/accuracy_train', acc, step_train)

            step_train += 1
            


        print(f'TRAIN ACCURACY: {running_train_correct/total_train_len*100}')
        
        if gpu_id == 0:
            model.save_model(epoch,f'{exp}')
               


        overall_output_1 = []
        overall_output_2 = []
        overall_labels = []   
        
        if gpu_id == 0:
            model.eval()
            for idx, input_data in enumerate(tqdm(val_dataloader)):
                with torch.no_grad():
                    if args['use_triplet_loss']:
                        anc, image_2, labels, type, pos, neg = input_data
                    else:
                        anc, image_2, labels, type = input_data
                    anc = anc.to(device)
                    image_2 = image_2.to(device)
                    labels = labels.to(device)
                    b = len(labels)

                    if args['use_triplet_loss']:
                        pos = pos.to(device)
                        neg = neg.to(device)

                    self_attn_1, output_1, self_attn_2, output_2 = model(anc, image_2)

                    frames = []
                    for i in range(b):
                        frames.append(flip(self_attn_1[i],type[i]).unsqueeze(0))

                    self_attn_1 = torch.cat(frames)
                    
                    loss_out = (loss_output_fn(output_1, labels) + loss_output_fn(output_2, labels)) # TODO try weighted cross entropy also

                    loss_att = loss_attn_fn(self_attn_1, self_attn_2)*args['attn_loss_weight'] 

                    if args['use_triplet_loss']:
                        anc_features, pos_features, neg_features = model.triplet_forward(anc, pos, neg)
                        loss_trip = loss_triplet_fn(anc_features, pos_features, neg_features)*args['triplet_loss_weight']

                    loss = 0
                    if args['add_cross_entropy']:
                        loss += loss_out
                    if args['add_attn_loss']:
                        loss += loss_att
                    if args['add_triplet_loss']:
                        loss+=loss_trip

                    if step_val%5:
                        writer.add_scalar('Loss_overall/val', loss.item(), step_val)
                        writer.add_scalar('Loss_output/val', loss_out.item(), step_val)
                        writer.add_scalar('Loss_attention/val', loss_att.item(), step_val)
                    if args['use_triplet_loss']:
                        writer.add_scalar('Loss_triplet/val', loss_trip.item(), step_val)


                    step_val+=1
                    overall_output_1.append(output_1)
                    overall_output_2.append(output_2)
                    overall_labels.append(labels)
                    
        
            
            print('Metrics')
            precision, recall, f1, acc = get_results(torch.cat(overall_output_1), torch.cat(overall_labels))
            writer.add_scalar('Output_1/precision_val', precision, epoch)
            writer.add_scalar('Output_1/recall_val', recall, epoch)
            writer.add_scalar('Output_1/f1_val', f1, epoch)
            writer.add_scalar('Output_1/accuracy_val', acc, epoch)

        

        if acc > max_val_acc and gpu_id == 0:
            max_val_acc = acc
            model.save_model(epoch, exp, latest=False)
                 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for your script")

    # Boolean parameters
    parser.add_argument('--add_cross_entropy', action='store_true', default=False, help='Add cross entropy')
    parser.add_argument('--add_attn_loss', action='store_true', default=False, help='Add attention loss')
    parser.add_argument('--use_triplet_loss', action='store_true', default=False, help='Use triplet loss')
    parser.add_argument('--add_triplet_loss', action='store_true', default=False, help='Add triplet loss')


    # Choice parameter for model_type
    parser.add_argument('--model_type', choices=['base_imagenet', 'lora', 'lora_mod', 'adapter','vpt','vptlora'], help='Model type', default = 'base_imagenet')
    parser.add_argument('--num_tokens', type=int, default=2, help='Num tokens for VPT')

    # Additional parameters
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--seed', type=int, default=16, help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (e.g., cuda or cpu)')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from a checkpoint')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--attn_loss_weight', type=int, default=10000, help='Attention loss weight')
    parser.add_argument('--triplet_loss_weight', type=int, default=10, help='Triplet loss weight')
    parser.add_argument('--dataset', choices=['AID', 'PatternNet','EuroSat','UCMerced_LandUse'], default = 'PatternNet')
    parser.add_argument('--folder', default='Experiments_PatternNet')


    args = parser.parse_args()
    main(args)

