import torch
import models
import torchmetrics
from pathlib import Path
from data import collate_fn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import plot_loss, plot_acc, plot_hyperparameter_loss, get_adam_optimizer

def train_one_epoch(model, dataloader, optimizer, device='cuda'):
    '''
    Train one epochs.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/train_loader
        optimizer:
            optimizer with model parameters loaded in.
        device:
            cuda or cpu.
    Return:
        tensor: mean loss value in one epoch with train loader
    '''
    model.train()
    loss_list = []
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        loss = model(data, target)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.mean(torch.tensor(loss_list))


@torch.no_grad()
def validation_one_epoch(model, dataloader, device='cuda'):
    '''
    Validatae one epochs.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/val_loader
        device:
            cuda or cpu
    Return:
        tensor: mean loss value in one epoch with validation loader
    '''
    model.train()
    loss_list = []
    for data, target in dataloader:
        data = data.to(device)
        target = target.cuda()
        loss = model(data, target)
        loss_list.append(loss.item())
    return torch.mean(torch.tensor(loss_list))


@torch.no_grad()
def evaluate(model, dataloader, num_class=10, device='cuda'):
    '''
    Evaluate model performance using the dataloader.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/val_loader
        num_class:
            number of classes in the dataset
        device:
            cuda or cpu
    Return:
        float, float: accuracy over test set, f1-score over test set
    '''
    acc_metrics = torchmetrics.Accuracy()
    f1_metrics = torchmetrics.F1Score(num_class, average='macro')
    model.eval()
    for data, target in dataloader:
        data = data.to(device)
        pred = model(data)
        pred = pred.softmax(dim=-1)
        acc = acc_metrics(pred.cpu(), target)
        f1 = f1_metrics(pred.cpu(), target)
    return acc_metrics.compute(), f1_metrics.compute()


def hyper_parameters_search(
        model_name,
        num_class,
        pretrained,
        feature_extractor,
        progress,
        hyperparameters,
        validation_dataset,
        validation_epoch=50,
        device='cuda'):
    '''
    Hyperparameter seach function. Select hyperparameters base on last epoch loss.
    Args:
        model_type:
            which model to use.
        num_class:
            number of classes in the dataset
        pretrained:
            pretrained backbone
        feature_extractor:
            freeze backbone as feature extractor
        progress:
            display progress for model
        hyperparameters:
            list of hyperparameters to seach from. need to have format 
            (learning weight, weight_decay, batch size)
        validation_dataset:
            dataset for validation.
        validation_epoch:
            number of epoch to search each set of hyperparamters from.
            default: 50
    Return:
        tuple: best set of hyperparameters
    '''
    param_loss = []
    for param in hyperparameters:
        lr, weight_decay, batch_size = param
        # Create dataloader with according batchsize
        dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, collate_fn=collate_fn)
        # Create model
        model = models.__dict__[model_name](
            num_class, pretrained, feature_extractor, progress)
        model.to(device)
        # Create optimizer
        optimizer = get_adam_optimizer(model, lr, weight_decay)
        # Start straining
        loss = []
        for epoch in range(validation_epoch):
            epoch_loss = train_one_epoch(model, dataloader, optimizer)
            #print(f"epoch {epoch}", f"loss: {epoch_loss}")
            loss.append(epoch_loss)
        # Plot loss graph
        plot_hyperparameter_loss(loss, [lr, weight_decay, batch_size])
        # Measure performance for hyperparameters
        metrics = evaluate(model, dataloader, num_class, device)
        acc = metrics[0]
        param_loss.append((param, acc))
    # Select hyperparameter with the highest last epoch accuracy
    param_loss.sort(key=lambda x: x[1])
    best_param = param_loss[-1][0]
    return best_param


def train_model(model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        num_class=10,
        training_epoch=50,
        save_path=Path('.'),
        device='cuda'):
    '''
    Training function. 
    Args:
        model:
            trainable model.
        val_loader:
            dataloader for validation dataset. keep track of validation loss.
        training_epoch:
            number of epoch to train the model.
        save_path:
            pathway to save the model state_dict.
    '''
    model.train()
    model.to(device)
    # Record training and validation loss
    train_loss = []
    val_loss = []
    train_acc_list = []
    val_acc_list = []
    save_path.mkdir(exist_ok=True)
    # Start training
    for epoch in range(training_epoch):
        epoch_train_loss = train_one_epoch(
            model, train_loader, optimizer, device)
        # Saving model every epochs
        torch.save(model.state_dict(), str(save_path/f'epoch-{epoch}.pth'))
        train_loss.append(epoch_train_loss)
        epoch_val_loss = validation_one_epoch(model, val_loader)
        val_loss.append(epoch_val_loss)
        # Measure model performance
        train_acc, train_f1 = evaluate(model, train_loader, num_class)
        val_acc, val_f1 = evaluate(model, val_loader, num_class)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"epoch {epoch}")
        '''
        print(
            "\t train loss: {:.4f}".format(epoch_train_loss),
            "\t train acc: {:.4f}".format(train_acc),
            "\t train f1: {:.4f}".format(train_f1),
            "\t val loss: {:.4f}".format(epoch_val_loss),
            "\t val acc: {:.4f}".format(val_acc),
            "\t val f1: {:.4f}".format(val_f1))
        '''
        print(
            "\t train loss: {:.4f}".format(epoch_train_loss),
            "\t train acc: {:.4f}".format(train_acc),
            "\t val loss: {:.4f}".format(epoch_val_loss),
            "\t val acc: {:.4f}".format(val_acc))
    plot_loss(train_loss, 'train')
    plot_loss(val_loss, 'validation')
    plot_acc(train_acc_list, 'train')
    plot_acc(val_acc_list, 'validation')
    return train_loss, val_loss



