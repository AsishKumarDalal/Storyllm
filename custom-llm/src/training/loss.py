import torch
from .batch import get_batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
def calc_loss_loader(split, model, device, num_batches=5):
    """
    Calculate average loss over num_batches using get_batch function

    Args:
        split: 'train' or 'validation'
        model: the model to evaluate
        device: device to run on
        num_batches: number of batches to average over
    """
    total_loss = 0.

    for i in range(num_batches):
        input_batch, target_batch = get_batch(split)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches