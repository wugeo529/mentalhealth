import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from data.dataset import SpectrogramDataset
from model.lstm_model import LSTMClassifier
from utils.train_eval import train_model, evaluate_model

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = SpectrogramDataset(
        config["paths"]["label_file"],
        config["paths"]["spectrogram_root"],
        transform=transform,
        device=device
    )

    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    model = LSTMClassifier(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = StepLR(optimizer, step_size=config["training"]["step_size"], gamma=config["training"]["gamma"])

    train_model(model, dataloader, criterion, optimizer, scheduler, config["training"]["num_epochs"], device)
    evaluate_model(model, dataloader, device)
