"""
https://github.com/ahmedbesbes/character-based-cnn/blob/master/train.py

torch.utils.data.random_split(dataset, lengths)
"""
import os
os.chdir('S:/TRIALDEV/CAMARADES/Qianying/RoB_All/pytorch')

import csv
csv.field_size_limit(100000000)
import argparse
import shutil
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_helper import rob_data
#from train_validate import train, validate
#from model import DocCNN


os.chdir('S:/TRIALDEV/CAMARADES/Qianying/RoB_All')


parser = argparse.ArgumentParser('CNN for document classification')
parser.add_argument('--data_path', type=str, default='./datafile/fulldata.csv', help='path of the data (default: ./data/fulldata.csv)')
parser.add_argument('--sep', type=str, default='\t', help='delimiter to use (default: \t)')
parser.add_argument('--encoding', type=str, default='utf-8', help='encoding to use when reading/writing (default: utf-8)')    
parser.add_argument('--chunksize', type=int, default=100000, help='size of the chunks when loading the data using pandas (default: 100000)') 
parser.add_argument('--max_rows', type=int, default=None, help='the maximum number of rows to load from the dataset (default: None)')

parser.add_argument('--num_classes', type=float, default=2, help='number of classes (default: 2)')
parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of training data (default: 0.8)')
parser.add_argument('--label_column', type=str, default='RoBLabel', help='column name of the labels (default: RoBLabel)')
parser.add_argument('--text_column', type=str, default='RoBText', help='column name of the texts (default: RoBText)')    
   
parser.add_argument('--steps', nargs='+', default=['remove_urls', 'remove_non_ascii', 'remove_digits', 'remove_punctuations', 
                                                   'strip_whitespaces', 'remove_one_character', 'lower'], 
                    help='text preprocessing steps to include on the text')


    
parser.add_argument('--log_path', type=str, default='./logs/', help='path of tensorboard log file (default: ./logs/)')
parser.add_argument('--model_name', type=str, help='prefix name of saved models')
parser.add_argument('--output', type=str, default='./models/', help='path of the folder where models are saved (default: ./models/)')
parser.add_argument('--workers', type=int, default=1, help='number of workers in PyTorch DataLoader (default: 1)')
parser.add_argument('--checkpoint', type=int, choices=[0, 1], default=1, help='save the model on disk or not (default: 1)')
    

parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 128)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs (default: 10)')
parser.add_argument('--patience', type=int, default=3, help=' number of epochs to wait without improvement of the validation loss (default: 3)')
parser.add_argument('--vocab_size', type=int, help='number of total unique words in all documents')
parser.add_argument('--embed_dim', type=int, default=200, help='number of embedding dimension (default: 200)')
parser.add_argument('--num_filter', type=int, default=100, help='number of each kind of filter (default: 100)')
parser.add_argument('--filter_size', type=str, default='3,4,5', help='comma-separated filter sizes (default: 3,4,5)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes (default: 2)')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate (default: 0.5)')
parser.add_argument('--max_length', type=int, default=4000, help='maximum length (number of words) to fix for all the documents (default: 4000)')  


args = parser.parse_args()




#%%
log_path = args.log_path
if os.path.isdir(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)
if not os.path.exists(args.output):
    os.makedirs(args.output)

writer = SummaryWriter(log_path)
# writer.close()



training_params = {"batch_size": args.batch_size,
                   "shuffle": True,
                   "num_workers": args.workers}

validation_params = {"batch_size": args.batch_size,
                     "shuffle": False,
                     "num_workers": args.workers}

full_dataset = rob_data(args)   
train_size = int(args.train_ratio * len(full_dataset))
validation_size = len(full_dataset) - train_size
training_set, validation_set = torch.utils.data.random_split(full_dataset, [train_size, validation_size])
training_generator = DataLoader(training_set, **training_params)
validation_generator = DataLoader(validation_set, **validation_params)


# update args
args.vocab_size = 
args.filter_size = [int(f) for f in args.filter_size.split(',')]


model = DocCNN(args)     ###############################################
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

best_loss = 1e10
best_epoch = 0

for epoch in range(args.epochs):
    train_loss, train_acc, train_sen, train_spec, train_pres, train_f1 = train(model, training_generator, optimizer, criterion, epoch, writer)        
    val_loss, val_acc, val_sen, val_spec, val_pres, val_f1 = validate(model, validation_generator, optimizer, criterion, epoch, writer)

    print('[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \ttrain_sen: {:.4f} \ttrain_spec: {:.4f} \ttrain_pres: {:.4f} \ttrain_f1: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f} \tval_sen: {:.4f} \tval_spec: {:.4f} \tval_pres: {:.4f} \tval_f1: {:.4f}'.
          format(epoch + 1, args.epochs, 
                 train_loss, train_acc, train_sen, train_spec, train_pres, train_f1, 
                 val_loss, val_acc, val_sen, val_spec, val_pres, val_f1))
    print("=" * 32)

    # early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        if args.checkpoint == 1:
            torch.save(model, args.output + 'char_cnn_epoch_{}_{}_{}_loss_{}_acc_{}_sen_{}_spec_{}.pth'.
                       format(args.model_name, epoch, 
                              optimizer.state_dict()['param_groups'][0]['lr'], 
                              round(val_loss, 4), round(val_acc, 4), round(val_sen, 4), round(val_spec, 4)))

    if epoch - best_epoch > args.patience > 0:
        print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, val_loss, best_epoch))
        break
        
        
        

# def run(args, both_cases=False):
# run(args)
    
    

