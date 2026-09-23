from .early_stopping import EarlyStopping
from .criterion import Criterion
from .metrics import ClassificationMetrics
from .model import Model

def train(
    model: Model, 
    ea: EarlyStopping, 
    max_epochs: int, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: Criterion,
    device: torch.device = torch.device("cpu")
):
    """
    Standard training loop utilizing the EarlyStopping mechanism.
    """
    model.to(device)
    
    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for x, e, y in train_loader:
            x, e, y = x.to(device), e.to(device), y.to(device)
            
            optimizer.zero_grad()
            e_pred, y_pred = model(x)
            loss = criterion(e, e_pred, y, y_pred)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, e, y in val_loader:
                x, e, y = x.to(device), e.to(device), y.to(device)
                e_pred, y_pred = model(x)
                loss = criterion(e, e_pred, y, y_pred)
                val_loss += loss.item() * x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- Early Stopping Check ---
        # Note: ea() internally increments its own epoch counter
        ea(value=val_loss, model=model)
        
        if ea.early_stop:
            print("Training terminated by EarlyStopping.")
            break
