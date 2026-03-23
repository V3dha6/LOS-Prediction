import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Set global style for white background
plt.style.use('default')

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def plot_performance_metrics(t_loss, v_loss, t_acc, v_acc):
    # Loss Window
    plt.figure(figsize=(8, 6))
    plt.plot(t_loss, label='Train Loss', color='blue')
    plt.plot(v_loss, label='Validation Loss', color='orange')
    plt.title('VAE Training vs Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show() # Pops up in a new window

    # Accuracy Window
    plt.figure(figsize=(8, 6))
    plt.plot(t_acc, label='Train Acc', color='blue')
    plt.plot(v_acc, label='Val Acc', color='orange')
    plt.title('VAE Training vs Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12) # X-axis as requested
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show() # Pops up in a new window

def plot_conf_matrix(cm):
    # Confusion Matrix Window
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show() # Pops up in a new window

def train_and_extract_features():
    df = pd.read_csv('data/processed/model_ready_data.csv', low_memory=False)
    metadata_cols = ['hadm_id', 'subject_id', 'los', 'anchor_age', 'gender', 'admission_type']
    icd_df = df.drop(columns=metadata_cols, errors='ignore')
    data = torch.tensor(icd_df.apply(pd.to_numeric, errors='coerce').fillna(0).values, dtype=torch.float32)
    data = torch.clamp(data, 0, 1)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    model = VAE(input_dim=data.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    t_loss, v_loss, t_acc, v_acc = [], [], [], []
    
    for epoch in range(100):
        model.train()
        recon, mu, logvar = model(train_data)
        loss = F.binary_cross_entropy(recon, train_data, reduction='sum') + \
               -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        t_loss.append(loss.item() / len(train_data))
        t_acc.append(((recon > 0.5).float() == train_data).float().mean().item())
        
        model.eval()
        with torch.no_grad():
            v_recon, v_mu, v_logvar = model(val_data)
            v_loss_val = F.binary_cross_entropy(v_recon, val_data, reduction='sum') + \
                         -0.5 * torch.sum(1 + v_logvar - v_mu.pow(2) - v_logvar.exp())
            v_loss.append(v_loss_val.item() / len(val_data))
            v_acc.append(((v_recon > 0.5).float() == val_data).float().mean().item())

    # Generate Confusion Matrix for top 10 features
    conf_mtx = confusion_matrix(val_data.numpy()[:, :10].flatten(), 
                                (v_recon > 0.5).float().numpy()[:, :10].flatten())

    # Trigger Pop-ups
    plot_performance_metrics(t_loss, v_loss, t_acc, v_acc)
    plot_conf_matrix(conf_mtx)

    # Save latents
    model.eval()
    with torch.no_grad():
        _, mu, _ = model.forward(data)
        latent_df = pd.DataFrame(mu.numpy(), columns=[f'latent_{i}' for i in range(32)])
        pd.concat([df[['hadm_id', 'los']], latent_df], axis=1).to_csv('data/processed/latent_features.csv', index=False)

if __name__ == "__main__":
    train_and_extract_features()