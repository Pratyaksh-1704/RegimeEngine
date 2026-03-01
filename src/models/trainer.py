import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from .encoder import TCNEncoder, contrastive_loss

logger = logging.getLogger(__name__)

class TCNTrainer:
    """Trainer for the self-supervised TCN."""
    
    def __init__(self, model: TCNEncoder, lr=1e-3, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def train(self, dataloader: DataLoader, epochs: int = 50, margin: float = 1.0) -> list:
        self.model.train()
        logger.info(f"Starting TCN training for {epochs} epochs on {self.device}")
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for w1, w2 in dataloader:
                w1 = w1.to(self.device)
                w2 = w2.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass for both windows
                z1 = self.model(w1)
                z2 = self.model(w2)
                
                # Compute objective
                loss = contrastive_loss(z1, z2, margin=margin)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
                
        logger.info("Training complete.")
        return loss_history
        
    def extract_features(self, dataloader: DataLoader):
        """Extracts latent embeddings z_t for the dataset."""
        self.model.eval()
        embeddings = []
        logger.info("Extracting latent embeddings...")
        
        with torch.no_grad():
            for w in dataloader:
                w = w.to(self.device)
                z = self.model(w)
                embeddings.append(z.cpu())
                
        return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test
    from src.data.dataset import WindowedDataset
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(np.random.randn(100, 2))
    dataset = WindowedDataset(df, window_size=21)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    
    model = TCNEncoder(input_size=2, num_channels=[16, 32, 64], latent_dim=8)
    trainer = TCNTrainer(model)
    trainer.train(loader, epochs=5)
