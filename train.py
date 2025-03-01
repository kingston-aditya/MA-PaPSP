import torch
import numpy as np
import torch.nn as nn
from models import pipeline1, pipeline2, pipeline3
from configs.config import get_config, get_weights_file_name
from tqdm import tqdm

def train(model, config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # load dataset
    train_dataloader, = 

    # get the pipeline ready
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    loss_fn = nn.MSELoss().to(device)
    
    for epoch in range(config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            # get the data
            X = batch["input"].to(device)
            Y = batch["output"].to(device)
            RX = batch["ret_input"].to(device)
            RY = batch["ret_input"].to(device)
            
            # make prediction
            out = model.forward(X, Y, RX, RY)

            # get loss function
            loss = loss_fn(out, Y)
            batch_iterator.set_postfix({f"loss": f"{loss.item(): 6.3f}"})
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()
        
        # save the model
        model_filename = get_weights_file_name(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() 
        }, model_filename)

if __name__ == "__main__":
    config = get_config()

    # get model
    if config["pipeline"] == 1:
        model = pipeline1.pipeline1(config["embed_size"], config["SA_number"], config["CA_number"])
    elif config["pipeline"] == 2:
        model = pipeline2.pipeline2(config["embed_size"], config["SA_number"], config["CA_number"])
    elif config["pipeline"] == 3:
        model = pipeline3.pipeline3(config["embed_size"], config["SA_number"], config["CA_number"])
    else:
        print("NOT a NUMBER")

    # get the training pipeline correct
    train(model, config)
    


