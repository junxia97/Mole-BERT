import argparse
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import GNN, DiscreteGNN, GNNDecoder
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from util import MaskAtom
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tensorboardX import SummaryWriter
criterion = nn.CrossEntropyLoss()
import timeit
NUM_NODE_ATTR = 119 
NUM_NODE_CHIRAL = 4
NUM_BOND_ATTR = 4

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x, e):    
        encoding_indices = self.get_code_indices(x, e) # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss
    
    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c + index_n + index_o)
        # compute L2 distance
        encoding_indices = torch.ones(x.size(0)).long().to(x.device)
        # C:
        distances = (
            torch.sum(e[index_c] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[0: 377] ** 2, dim=1) -
            2. * torch.matmul(e[index_c], self.embeddings.weight[0: 377].t())
        )
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N:
        distances = (
            torch.sum(e[index_n] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[378: 433] ** 2, dim=1) -
            2. * torch.matmul(e[index_n], self.embeddings.weight[378: 433].t())
        ) 
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O:
        distances = (
            torch.sum(e[index_o] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[434: 488] ** 2, dim=1) -
            2. * torch.matmul(e[index_o], self.embeddings.weight[434: 488].t())
        )   
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others:
        distances = (
            torch.sum(e[index_others] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[489: 511] ** 2, dim=1) -
            2. * torch.matmul(e[index_others], self.embeddings.weight[489: 511].t())
        ) 
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489

        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))     

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def train_vae(args, epoch, model_list, loader, optimizer_list, device):
    criterion = nn.CrossEntropyLoss()

    model, vq_layer, dec_pred_atoms, dec_pred_bonds, dec_pred_atoms_chiral = model_list
    optimizer_model, optimizer_model_vq, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_atoms_chiral = optimizer_list
    
    model.train()
    vq_layer.train()
    dec_pred_atoms.train()
    dec_pred_atoms_chiral.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accum = 0
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr) 
        e, e_q_loss = vq_layer(batch.x, node_rep)
        pred_node = dec_pred_atoms(e, batch.edge_index, batch.edge_attr)
        pred_node_chiral = dec_pred_atoms_chiral(e, batch.edge_index, batch.edge_attr)
        atom_loss = criterion(pred_node, batch.x[:, 0]) 
        atom_chiral_loss = criterion(pred_node_chiral, batch.x[:, 1]) 
        recon_loss = atom_loss + atom_chiral_loss

        if args.edge:
            edge_rep = e[batch.edge_index[0]] + e[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep, batch.edge_index, batch.edge_attr)
            recon_loss += criterion(pred_edge, batch.edge_attr[:,0])
    
        loss = recon_loss + e_q_loss
        optimizer_model.zero_grad()
        optimizer_model_vq.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        optimizer_dec_pred_atoms_chiral.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()
        optimizer_model.step()
        optimizer_model_vq.step()
        optimizer_dec_pred_atoms.step()
        optimizer_dec_pred_atoms_chiral.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} train_loss: {loss.item():.4f}")

    return loss_accum/step

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=5,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')

    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--num_tokens', type=int, default=512,
                        help='number of atom tokens (default: 512)') 
    parser.add_argument('--commitment_cost', type = float, default = 0.25, help = 'commitment_cost')
    parser.add_argument('--edge', type=int, default=1, help='whether to decode edges or not together with atoms') 

    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.0,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.edge))
    #set up dataset and transform function.
    dataset = MoleculeDataset("/root/Mole-BERT-plus/dataset/" + args.dataset, dataset=args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    model = DiscreteGNN(args.num_layer, args.emb_dim).to(device)
    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False
    vq_layer = VectorQuantizer(args.emb_dim, args.num_tokens, args.commitment_cost).to(device)
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    atom_chiral_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_CHIRAL, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.edge:
        NUM_BOND_ATTR = 4
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type='linear').to(device)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model, vq_layer, atom_pred_decoder, bond_pred_decoder, atom_chiral_pred_decoder] 
    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_model_vq = optim.Adam(vq_layer.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms_chiral = optim.Adam(atom_chiral_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms_chiral = optim.Adam(atom_chiral_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_dec_chiral = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms_chiral, lr_lambda=scheduler)        
        scheduler_list = [scheduler_model, scheduler_dec, scheduler_dec_chiral, None]
    else:
        scheduler_model = None
        scheduler_dec = None

    optimizer_list = [optimizer_model, optimizer_model_vq, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_atoms_chiral]
    output_file = "./checkpoints/" + args.output_model_file   
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss = train_vae(args, epoch, model_list, loader, optimizer_list, device)
        if not resume:
            if epoch == args.epochs:
                torch.save(model.state_dict(), output_file + f"vqencoder.pth")
                torch.save(vq_layer.state_dict(), output_file + f"vqquantizer.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

    # output_file = "./checkpoints/" + args.output_model_file
    # if resume:
    #     torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}_{args.start_epoch}.pth")
    # elif not args.output_model_file == "":
    #     torch.save(model.state_dict(), output_file + ".pth")

if __name__ == "__main__":
    main()
