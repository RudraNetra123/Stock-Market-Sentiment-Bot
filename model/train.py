import torch
from torch.utils.data import DataLoader
from dataset import SentimentPriceDataset
from lstm_model import LSTMRegressor

def train_model(csv_path, model_path='saved_model.pth', seq_len=2, epochs=5, lr=0.001):
    dataset = SentimentPriceDataset(csv_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = LSTMRegressor()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for seq, target in dataloader:
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

if __name__ == "__main__":
    train_model('../data/sentiment_price_history.csv')
