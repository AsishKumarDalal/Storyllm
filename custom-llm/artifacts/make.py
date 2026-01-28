import os
import matplotlib.pyplot as plt


def save_training_charts(train_losses, val_losses, tokens_seen):
    os.makedirs("artifacts", exist_ok=True)

    # Loss vs evaluation steps
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("loss_curve.png")
    plt.close()

    # Loss vs tokens seen
    plt.figure()
    plt.plot(tokens_seen, train_losses, label="Train Loss")
    plt.xlabel("Tokens Seen")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Tokens Seen")
    plt.savefig("loss_vs_tokens.png")
    plt.close()
