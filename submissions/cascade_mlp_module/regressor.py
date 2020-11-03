import torch
from torch.nn.functional import relu
from torch.nn.utils.rnn import pad_sequence


class Regressor():
    """A PyTorch MLP model consisting of an MLP for each module type.
    The model is learnt only on single module.
    The model takes as input the input power and the meta data of the
    corresponding cascade. To predict the output power the model
    simply cascades the different MLPs matching the input module cascade."""

    def __init__(self):
        super().__init__()
        # Since the model need meta data present in the data
        # we will only instantiate the model when calling the fit function
        self.Model = PyTorchModel  # PyTorch model class
        self.model = None  # PyTorch model instance
        self.mod_id = None  # Module IDs

    def fit(self, X, y):
        # Retrieve some information about the modules from the data
        all_mods = set(
            [(("type", mod[0]), ("nb_feat", len(mod[1]))) for seq, _, _ in X
             for mod in seq])
        mod_info = [dict(m) for m in all_mods]
        self.mod_id = {mod["type"]: i for i, mod in enumerate(mod_info)}

        # Instantiate the PyTorch model
        self.model = self.Model(mod_info)

        # Turn on training mode
        self.model.train()
        # Get data and create train data loaders
        data_list = [{"mod_id_seq": torch.tensor(
            [self.mod_id[mod] for mod, _ in mod_seq]),
            "mod_feat_seq_list": [torch.tensor(feat).float() for
                                  _, feat in mod_seq],
            "input_power": torch.tensor(p_in).float(),
            "output_power": torch.tensor(p_out).float()} for
            (mod_seq, p_in, campaign_id), p_out in zip(X, y)]

        train_loader = torch.utils.data.DataLoader(data_list, batch_size=128,
                                                   collate_fn=collate_fn)
        # Instantiate criterion and optimizer
        crit = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        # Training loop
        for e in range(100):
            for data in train_loader:
                (mod_id_seq, mod_feat_seq, p_in), p_out = data
                opt.zero_grad()
                preds = self.model(mod_id_seq, mod_feat_seq, p_in)
                # Since the evaluation is only done for on-channels it
                # helps the optimization to only backpropagate through them.
                on_chan = p_in != 0
                on_preds = torch.mul(on_chan, preds)
                on_p_out = torch.mul(on_chan, p_out)
                loss = crit(on_preds, on_p_out)
                # Since we are only looking at single modules
                # loss may contain no backpropagatable elements
                if loss.requires_grad:
                    loss.backward()
                opt.step()

    def predict(self, X):
        # Turn on evaluation mode
        self.model.eval()
        # No ground truth when predicting, format input arguments
        # Input powers
        p_in = torch.stack([torch.tensor(p_in).float() for _, p_in, _ in X])
        # Module features
        mod_feat_seq = [[torch.tensor(feat).float() for _, feat in mod_seq]
                        for mod_seq, _, _ in X]
        # Module IDs
        mod_id_seq = [torch.tensor([self.mod_id[mod] for mod, _ in mod_seq])
                      for mod_seq, _, _ in X]
        mod_id_seq = pad_sequence(mod_id_seq, batch_first=True,
                                  padding_value=-1)
        # Model prediction
        preds = self.model(mod_id_seq, mod_feat_seq, p_in).detach().numpy()
        return preds


class PyTorchModel(torch.nn.Module):
    def __init__(self, mod_info):
        super(PyTorchModel, self).__init__()
        self.mod_info = mod_info
        # Construct as many MLPs as modules present in the data
        self.MLPs = torch.nn.ModuleList(
            [MLP(m["nb_feat"]) for m in self.mod_info])

    def forward(self, mod_id_seq, mod_feat_seq, p_in):
        seq_len = torch.tensor(list(map(len, mod_feat_seq)))
        p_out = p_in
        if self.training:
            # Training done on single modules
            # returning p_in for cascade (no backpropagation)
            for i, m in enumerate(self.MLPs):
                msk = torch.mul(mod_id_seq[:, 0] == i, seq_len == 1)
                if msk.any():
                    feats = torch.stack(
                        [f[0] for i, f in enumerate(mod_feat_seq) if msk[i]])
                    p_out[msk] = m(torch.cat([p_out[msk], feats], dim=-1))
            return p_out

        else:
            # Concatenate MLP to evaluate cascades
            max_nb_mod = max(seq_len)
            for n in range(max_nb_mod):
                for i, m in enumerate(self.MLPs):
                    msk = torch.mul(mod_id_seq[:, n] == i, seq_len > n)
                    if msk.any():
                        feats = torch.stack(
                            [f[n] for i, f in enumerate(mod_feat_seq) if
                             msk[i]])
                        p_out[msk] = m(torch.cat([p_out[msk], feats], dim=-1))
            return relu(p_out)


class MLP(torch.nn.Module):
    """A simple two layer MLP taking as input the
    input powers and the features of the module"""

    def __init__(self, feat_size):
        super(MLP, self).__init__()
        # Definition of the modules of the model
        # Two fully connected layers
        self.fc0 = torch.nn.Linear(32 + feat_size, 128)
        self.fc1 = torch.nn.Linear(128, 32)

    def forward(self, x):
        # Compute the output of the model using a tanh activation function
        p_out = self.fc1(torch.tanh(self.fc0(x)))
        return p_out


def collate_fn(batch):
    # Power output
    p_out = torch.stack([sample["output_power"] for sample in batch])
    # Power input
    p_in = torch.stack([sample["input_power"] for sample in batch])
    # Module id
    l_id_seq = [sample["mod_id_seq"] for sample in batch]
    mod_id_seq = pad_sequence(l_id_seq, batch_first=True, padding_value=-1)
    # Module features
    mod_feat_seq = [sample["mod_feat_seq_list"] for sample in batch]

    return (mod_id_seq, mod_feat_seq, p_in), p_out
