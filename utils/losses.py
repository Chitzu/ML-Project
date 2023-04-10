import torch


def CrossEntropyLoss(predictions, labels):
    ce = torch.nn.CrossEntropyLoss()
    loss = ce(predictions, labels)
    return loss
