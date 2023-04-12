import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=runs to show tensorboard

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else "cpu"

mv_reviews_file = "processed_imdb_data.csv"
mv_reviews = pd.read_csv(mv_reviews_file)

dataset = mv_reviews["processed_review"].apply(lambda doc: [int(w) for w in doc.split(" ")]).tolist()
labels = mv_reviews["label"].astype("int32").tolist()

train_data, test_data, train_label, test_label = train_test_split(dataset, labels, test_size=0.2, random_state=1)

vocab_size = 123039 + 1
batch_size = 64
seq_len = 200
embedding_dim = 64
epoch = 10

class ImdbDataset(Dataset):
    def __init__(self, mode):
        if mode == "train":
            self.dataset = train_data
            self.labels = train_label
        else:
            self.dataset = test_data
            self.labels = test_label
    
    def __getitem__(self, index):
        doc = self.dataset[index]
        if len(doc) < seq_len:
            doc.extend([vocab_size-1]*(seq_len-len(doc)))
        doc = doc[:seq_len]
        label = self.labels[index]
        X = torch.from_numpy(np.array(doc)).to(torch.long).to(device)
        y = torch.from_numpy(np.array(label)).to(torch.float).to(device)
        return X, torch.reshape(y, [1])

    def __len__(self):
        return len(self.labels)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, rnn_layer=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.extractor = nn.LSTM(input_size=embedding_dim, 
                               hidden_size=hidden_dim,
                               num_layers=rnn_layer,
                               batch_first=True,
                               bidirectional=True)
        self.drop_layer = nn.Dropout(0.5)
        self.mlp = nn.Linear(2 * rnn_layer * hidden_dim, 1)
       
    def forward(self, raw_input):
        input_embedding = self.embedding(raw_input)
        output, (h_n, c_n) = self.extractor(input_embedding)
        h_n.permute(1, 0, 2)
        h_n = torch.reshape(h_n, [-1, self.extractor.hidden_size * (2 if self.extractor.bidirectional else 1)])
        h_n_dropout = self.drop_layer(h_n)
        return self.mlp(h_n_dropout)


def train():
    writer = SummaryWriter("runs/lstm_training_10_epoches")
    dataloader = DataLoader(ImdbDataset("training"), batch_size=batch_size, shuffle=True, num_workers=4)
    model = LSTMClassifier(vocab_size, embedding_size=128, hidden_dim=64, rnn_layer=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.BCEWithLogitsLoss()
    iter = 0
    for i in range(epoch):
        for batch_doc, label in dataloader:
            iter += 1
            output = model(batch_doc)
            optimizer.zero_grad()
            loss = loss_func(output, label.to(device))
            loss.backward()
            optimizer.step()
            print(f"training loss {loss}")
            writer.add_scalar("training_loss", loss, iter)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, "models/lstm_classifier_10_epoch.pkl")  


def test():
    dataloader = DataLoader(ImdbDataset("validation"), batch_size=batch_size)
    model = LSTMClassifier(vocab_size, embedding_size=128, hidden_dim=64, rnn_layer=1).to(device)
    model.load_state_dict(torch.load("models/lstm_classifier_10_epoch.pkl")["model"])
    pred_list = list()
    for batch_doc, _ in dataloader:
        output = model(batch_doc)
        pred = torch.sigmoid(torch.squeeze(output))
        batch_pred_list = pred.tolist()
        if type(batch_pred_list).__name__ == "list":
            pred_list.extend(batch_pred_list)
        else:
            pred_list.append(batch_pred_list)
    print(f"total test auc: {roc_auc_score(test_label, pred_list)}")
    

if __name__ == "__main__":
    train()
    test()