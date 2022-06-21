import torch
from tqdm import tqdm


def train(model, dataloader, num_epochs, optimizer, save_to):
    # Logging
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features = model(inputs)
                    optimizer.zero_grad()
                    loss = model.get_loss(inputs, output)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                running_loss += loss.detach().item() / batch_size
                batch_num += 1
                tepoch.set_postfix(loss=loss.detach().item() / batch_size)

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, save_to + ".pt")
