import torch 
import torch.nn as nn
from unet import UNET
from dataset import train_loader, val_loader, device
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def train(model, num_epochs, train_loader, val_loader, lr=1e-5, weight_decay=1e-6, patience=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_losses = []
    val_losses = []
    best_model_state = None
    counter = 0
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.003)

    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = []
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            avg_train_loss.append(loss.item())
        avg_train_loss = np.mean(avg_train_loss)
        model.eval()
        avg_val_loss = []
        for j, (x, y) in enumerate(tqdm(val_loader)):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
                loss = criterion(output, y)
                avg_val_loss.append(loss.item())
        avg_val_loss = np.mean(avg_val_loss)
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        tqdm.write(f"EPOCH: [{epoch+1}/{num_epochs}]\nTRAIN LOSS: [{avg_train_loss}]\nVALIDATION LOSS: [{avg_val_loss}]")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    loss_df = pd.DataFrame({'train loss':train_losses, 'val loss':val_losses})
    plt.plot(loss_df.index+1, loss_df['train loss'], label='train loss')
    plt.plot(loss_df.index+1, loss_df['val loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'model.pth')
    return model

def test():
    train_instance = train(
        model=UNET(),
        num_epochs=3,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        weight_decay=0.0001,
        patience=6
    )
    return train_instance
if __name__ == "__main__":
    test()

    

