import torch
from .loss import calc_loss_loader
from ..tokenizer.encode_decode import text_to_token_ids,token_ids_to_text
def evaluate_model(model, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader('train', model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader('validation', model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
