import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

ignore_words = ['?', '!', ',', '.']

def load_intents(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def preprocess_data(intents):
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    return np.array(X_train), np.array(y_train), all_words, tags

class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = X_data
        self.y_data = y_data
        self.n_samples = len(X_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def create_dataloader(X, y, batch_size=8):
    dataset = ChatDataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def save_model(model, input_size, hidden_size, output_size, all_words, tags, filepath='model.pth'):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, filepath)

def train_model(model, dataloader, device, criterion, optimizer, scheduler,
                num_epochs, input_size, hidden_size, output_size, all_words, tags, model_path):
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for words, labels in dataloader:
            words = words.to(device)
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if (epoch+1) % 100 == 0 or loss.item() < 0.0001:
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%')

        if total_loss < 0.00001:
            break

    print(f'Training complete. Final loss: {total_loss:.4f}')
    save_model(model, input_size, hidden_size, output_size, all_words, tags, model_path)

def main():
    intents = load_intents('intents.json')
    X_train, y_train, all_words, tags = preprocess_data(intents)

    input_size = len(X_train[0])
    hidden_size = 64
    output_size = len(tags)
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 2000
    model_path = 'model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    train_model(model, train_loader, device, criterion, optimizer, scheduler,
                num_epochs, input_size, hidden_size, output_size, all_words, tags, model_path)

if __name__ == '__main__':
    main()
