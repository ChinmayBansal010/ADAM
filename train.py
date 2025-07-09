import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import logging
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

ignore_words = ['?', '!', ',', '.']

def load_intents(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def preprocess_data(intents):
    all_words, tags, xy = [], [], []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    return np.array(X_train), np.array(y_train), all_words, tags

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloader(X, y, batch_size=8):
    dataset = ChatDataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def save_model(model, input_size, hidden_size, output_size, all_words, tags, path='model.pth'):
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }, path)

def train_model(model, loader, device, criterion, optimizer, scheduler,
                epochs, input_size, hidden_size, output_size, all_words, tags, model_path):

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        scheduler.step()
        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(loader)

        if (epoch + 1) % 100 == 0 or avg_loss < 0.01:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, input_size, hidden_size, output_size, all_words, tags, model_path)

        if avg_loss < 0.001:
            logging.info("Early stopping triggered.")
            break

    logging.info(f"Training complete. Best Loss: {best_loss:.4f}")
    logging.info(f"Total training time: {time.time() - start_time:.2f} seconds")

def main():
    intents = load_intents("intents.json")
    X_train, y_train, all_words, tags = preprocess_data(intents)

    input_size = len(X_train[0])
    hidden_size = 64
    output_size = len(tags)
    batch_size = 8
    lr = 0.001
    epochs = 2000
    model_path = "model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("training_log.txt"),
            logging.StreamHandler()
        ]
    )

    train_model(model, train_loader, device, criterion, optimizer, scheduler,
                epochs, input_size, hidden_size, output_size, all_words, tags, model_path)

if __name__ == "__main__":
    main()
