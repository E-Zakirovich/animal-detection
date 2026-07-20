# main.py
import torch
from dataset import DataPipeline
from model.cnn import CNN
from train import Trainer
import config


def main():
    # pick the best available device: CUDA > MPS (Apple Silicon) > CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # build data pipeline and get loaders
    pipeline = DataPipeline()
    train_loader, val_loader, test_loader = pipeline.load_dataset()

    # build model
    model = CNN()

    # train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.learning_rate
    )
    trained_model = trainer.fit(num_epochs=config.num_epochs)

    # save trained weights
    torch.save(trained_model.state_dict(), "trained_model.pth")
    print("Model saved to trained_model.pth")


if __name__ == "__main__":
    main()