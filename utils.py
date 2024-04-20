import torch
import itertools
from data import collate_fn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def get_hyper_parameters():
    lr = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5]
    weight_decay = [1e-3, 5e-3, 1e-4, 5e-4]
    batch_size = [4, 8, 16, 24]
    return itertools.product(lr, weight_decay, batch_size)


def get_adam_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    return optimizer


def get_dataloader(train, val, test, batch_size, collate_fn):
    train_loader = DataLoader(
        train, batch_size = batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(
        val, batch_size = batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(
        test, batch_size = batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def plot_loss(loss, mode="train"):
    '''
    Plot loss over epoch, helper function.
    Args:
        loss:
            list of loss over all epochs/
    '''
    plt.plot(loss)
    plt.title(f"{mode} loss/epoch")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def plot_acc(acc, mode="train"):
    '''
    Plot acc over epoch, helper function.
    Args:
        acc:
            list of acc over all epochs/
    '''
    plt.plot(acc)
    plt.title(f"{mode} accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


def plot_hyperparameter_loss(loss, hyperparameter):
    '''
    Plot loss over epoch, helper function.
    Args:
        loss:
            list of loss over all epochs/
    '''
    lr, weight_decay, batch_size = hyperparameter
    plt.plot(loss)
    plt.title(f"lr:{lr} weight_decay:{weight_decay} batch-size:{batch_size}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def show_examples(dataset, num_examples=5):
    dl = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True)
    counter = 0
    for counter, (image, label) in enumerate(dl):
        if (counter == num_examples):
            break
        plt.imshow(image[0].permute((1,2,0)))
        if label == 0:
            plt.title('Clickbait')
        else:
            plt.title('Non-Clickbait')
        plt.figure()



def get_saliency_map(X, y, model):
    pred = model(X)
    loss = torch.sum(pred[torch.arange(y.shape[0]), y])
    loss.backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    return saliency


def show_saliency_map(model, dataloader, num_example=5):
    for image, label in dataloader:
        if label != 0:
            continue
        if num_example == 0:
            break
        num_example -= 1
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(image[0].permute(1,2,0))
        image.requires_grad = True
        saliency = get_saliency_map(image, label, model)
        axarr[1].imshow(saliency[0], cmap=plt.cm.hot)
        plt.gcf().set_size_inches(16, 9)
        


def get_qualitative(model, dataset):
    tp, tn, fp, fn = None, None, None, None
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    for image, label in dataloader:
        pred = torch.argmax(model(image)[0]).item()
        if (label == 0 and pred == 0):
            tp = image
        elif (label == 0 and pred == 1):
            fn = image
        elif (label == 1 and pred == 0):
            fp = image
        elif (label == 1 and pred == 1):
            tn = image
        else:
            pass
    return tp, tn, fp, fn


