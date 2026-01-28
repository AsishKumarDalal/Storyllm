
from .batch import get_batch
from .loss import *
from .evaluate import *
from ..inference.generate import *
def train_model_simple(model, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       steps_per_epoch=1000):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for step in range(steps_per_epoch):
            # Use the get_batch function instead of DataLoader
            input_batch, target_batch = get_batch('train')

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample__(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen