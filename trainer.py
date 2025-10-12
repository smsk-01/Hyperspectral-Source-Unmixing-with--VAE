import numpy as np
import torch
import torch.optim as optim
from loss import loss_function

def train_vae(vae, train_loader, epochs=50, lr=1e-3, device='mps',
              patience=10, min_delta=0.0):
    """
    Train on the FULL dataset (no test set, no eval pass).
    Early stopping & LR scheduling are driven by TRAIN loss only.
    """
    optimizer = optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)

    # LR scheduler on train loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    vae.to(device)
    train_losses = []

    best_train_loss = float('inf')
    patience_counter = 0
    best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

    for epoch in range(epochs):
        vae.train()
        epoch_losses = []

        for element in train_loader:
            if isinstance(element, (tuple, list)):
                element = element[0]

            pixel = element[:, :vae.nb_channels].to(device).float()

            # Forward
            output, alpha, abundances = vae(pixel)

            # Loss (same weights as your previous TRAIN section)
            loss, recon, kl = loss_function(
                pixel, output, alpha, 
            )

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # Step LR scheduler on train loss
        scheduler.step(train_loss)

        # Early stopping on train loss
        improved = (best_train_loss - train_loss) > min_delta
        if improved:
            best_train_loss = train_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1:02d} | "
              f"Train: {train_loss:.6f} | "
              f"Recon(last): {recon.item():.6f} | "
              f"KL(last): {kl.item():.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best (by train loss)
    vae.load_state_dict(best_state)
    return train_losses
