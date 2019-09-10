"""
https://github.com/ahmedbesbes/character-based-cnn/blob/master/train.py
"""

import torch
import numpy as np
from tqdm import tqdm
import data_helper

def train(model, training_generator, optimizer, criterion, epoch, writer, print_every=25):
    model.train()
    losses = []
    accuraries = []
    sensitivities = []
    specificities = []
    precisions = []
    f1s = []
    
    num_iter_per_epoch = len(training_generator)
    progress_bar = tqdm(enumerate(training_generator), total=num_iter_per_epoch)

    for iter, batch in progress_bar:
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        training_metrics = data_helper.rob_metrics(labels.cpu().numpy(), predictions.cpu().detach().numpy(),
                                                   list_metrics=["accuracy", "sensitivity", "specificity", "precision", "f1"])
        losses.append(loss.item())
        accuraries.append(training_metrics["accuracy"])
        sensitivities.append(training_metrics["sensitivity"])
        specificities.append(training_metrics["specificity"])
        precisions.append(training_metrics["precision"])
        f1s.append(training_metrics['f1'])

        writer.add_scalar('Train/Loss', loss.item(), epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Accuracy', training_metrics['accuracy'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Sensitivity', training_metrics['sensitivity'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Specificity', training_metrics['specificity'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Precision', training_metrics['precision'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/f1', training_metrics['f1'], epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Training - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}, Sensitivity: {}, Specificity: {}, Precision: {}, f1: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries),
                np.mean(sensitivities),
                np.mean(specificities),
                np.mean(precisions),
                np.mean(f1s)
            ))

    return np.mean(losses), np.mean(accuraries), np.mean(sensitivities), np.mean(specificities), np.mean(precisions), np.mean(f1s)



def validate(model, validation_generator, criterion, epoch, writer, print_every=25):
    model.eval()
    losses = []
    accuraries = []
    sensitivities = []
    specificities = []
    precisions = []
    f1s = []
    
    num_iter_per_epoch = len(validation_generator)
    progress_bar = tqdm(enumerate(validation_generator), total=num_iter_per_epoch)

    for iter, batch in progress_bar:
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)
        validation_metrics = data_helper.rob_metrics(labels.cpu().numpy(), predictions.cpu().detach().numpy(),
                                                     list_metrics=["accuracy", "sensitivity", "specificity", "precision", "f1"])
        

        losses.append(loss.item())
        accuraries.append(validation_metrics["accuracy"])
        sensitivities.append(validation_metrics["sensitivity"])
        specificities.append(validation_metrics["specificity"])
        precisions.append(validation_metrics["precision"])
        f1s.append(validation_metrics['f1'])

        writer.add_scalar('Test/Loss', loss.item(), epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Accuracy', validation_metrics['accuracy'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Sensitivity', validation_metrics['sensitivity'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Specificity', validation_metrics['specificity'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Precision', validation_metrics['precision'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/f1', validation_metrics['f1'], epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}, Sensitivity: {}, Specificity: {}, Precision: {}, f1: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries),
                np.mean(sensitivities),
                np.mean(specificities),
                np.mean(precisions),
                np.mean(f1s)
            ))

    return np.mean(losses), np.mean(accuraries), np.mean(sensitivities), np.mean(specificities), np.mean(precisions), np.mean(f1s)
