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

def save_model(model, input_size, hidden_size, output_size, all_words, tags, filepath='data.pth'):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    
    FILE = 'model.pth'
    torch.save(data,FILE)

def train_model(model, dataloader, device, criterion, optimizer, num_epochs,
                input_size, hidden_size, output_size, all_words, tags):
    for epoch in range(num_epochs):
        for words, labels in dataloader:
            words = words.to(device)
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')
        if (loss.item() < 0.01):
            break
    print(f'Training complete. Final loss: {loss.item():.4f}')
    save_model(model, input_size, hidden_size, output_size, all_words, tags)


def main():
    intents = load_intents('intents.json')
    X_train, y_train, all_words, tags = preprocess_data(intents)

    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 2000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, device, criterion, optimizer, num_epochs,
            input_size, hidden_size, output_size, all_words, tags)

if __name__ == '__main__':
    main()
