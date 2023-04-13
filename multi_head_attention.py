import torch
import torch.nn as nn
from imdb_text_classifier import *

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=2, mid_dim=128):
        super().__init__()
        self.head_num = head_num
        self.output_dim = output_dim
        
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)
        
        self.ffn_layer_1 = nn.Linear(output_dim, mid_dim)
        self.ffn_layer_2 = nn.Linear(mid_dim, output_dim)
        
        self.layer_norm_1 = nn.LayerNorm(output_dim)
        self.layer_norm_2 = nn.LayerNorm(output_dim)
        
    def forward(self, input_mat):
        Q = self.query_layer(input_mat)
        K = self.key_layer(input_mat)
        V = self.value_layer(input_mat)

        head_dim = self.output_dim // self.head_num
        
        Q_expand = torch.cat(torch.split(Q, head_dim, dim=2), dim=0)
        K_expand = torch.cat(torch.split(K, head_dim, dim=2), dim=0)
        V_expand = torch.cat(torch.split(V, head_dim, dim=2), dim=0)

        sim_mat = torch.matmul(Q_expand, K_expand.transpose(1,2)) / (head_dim ** 0.5)

        sim_mat = nn.functional.softmax(sim_mat, dim=2)
        sim_mat = torch.matmul(sim_mat, V_expand)

        sim_mat = torch.cat(torch.split(sim_mat, input_mat.shape[0], dim=0), dim=2)
        self_atte_res = self.layer_norm_1(input_mat + torch.nn.functional.dropout(sim_mat, p=0.1))
        
        ffn_res = self.ffn_layer_2(nn.functional.relu(self.ffn_layer_1(self_atte_res)))
        add_and_norm = self.layer_norm_2(self_atte_res + torch.nn.functional.dropout(ffn_res, p=0.1))
        
        return add_and_norm


class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, input_dim, output_dim, head_num=2, mid_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = TransformerEncoder(input_dim, output_dim, head_num, mid_dim)
        self.mlp = nn.Linear(output_dim, 1)
        
    def forward(self, raw_input):
        input_embedding = self.embedding(raw_input)
        encode_block = self.encoder(input_embedding)
        cls_part = torch.squeeze(encode_block[:, -1, :], dim=1)
        return self.mlp(cls_part)

def train():
    writer = SummaryWriter("runs/transformer_training_10_64_epoches")
    dataloader = DataLoader(ImdbDataset("training"), batch_size=batch_size, shuffle=True, num_workers=4)
    model = SelfAttentionClassifier(vocab_size, embedding_size=128, input_dim=128, output_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.BCEWithLogitsLoss()
    iter = 0
    for i in range(0, epoch):
        for batch_doc, label in dataloader:
            iter += 1
            output = model(batch_doc)
            optimizer.zero_grad()
            loss = loss_func(output, label.to(device))
            loss.backward()
            optimizer.step()
            print("training loss:", loss)
            writer.add_scalar("training_loss", loss, iter)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, "models/transformer_classifier_64.pkl")

def test():
    dataloader = DataLoader(ImdbDataset("validation"), batch_size=batch_size)
    model = SelfAttentionClassifier(vocab_size, embedding_size=128, input_dim=128, output_dim=128).to(device)
    model.load_state_dict(torch.load("models/transformer_classifier_64.pkl")["model"])
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