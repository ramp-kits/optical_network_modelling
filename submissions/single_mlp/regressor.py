import torch
from torch.nn.functional import relu


class Regressor(torch.nn.Module):
    """A simple PyTorch MLP model using only the input power"""

    def __init__(self):
        super(Regressor, self).__init__()
        # Definition of the modules of the model
        # Two fully connected layers
        self.fc0 = torch.nn.Linear(32, 128)
        self.fc1 = torch.nn.Linear(128, 32)

    def forward(self, p_in):
        # Compute the output of the model using a tanh activation function
        p_out = self.fc1(torch.tanh(self.fc0(p_in)))
        # Return positive values when evaluating the model
        return p_out if self.training else relu(p_out)

    def fit(self, X, y):
        # Turn on training mode
        self.train()
        # Get data and create train data loaders
        data_as_list = [
            [torch.tensor(p_in).float(), torch.tensor(p_out).float()]
            for (_, p_in, _), p_out in zip(X, y)]
        train_loader = torch.utils.data.DataLoader(data_as_list,
                                                   batch_size=128)
        # Instantiate criterion and optimizer
        crit = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.parameters())

        # Training loop
        for e in range(100):
            for p_in, p_out in train_loader:
                opt.zero_grad()
                preds = self(p_in)
                # Since the evaluation is only done for on-channels it
                # helps the optimization to only backpropagate through them.
                on_chan = p_in != 0
                on_preds = torch.mul(on_chan, preds)
                on_p_out = torch.mul(on_chan, p_out)
                loss = crit(on_preds, on_p_out)
                loss.backward()
                opt.step()

    def predict(self, X):
        # Turn on evaluation mode
        self.eval()
        # No ground truth when predicting
        p_in = torch.stack([torch.tensor(p_in).float() for _, p_in, _ in X])
        preds = self(p_in).detach().numpy()
        return preds
