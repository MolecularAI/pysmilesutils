import math, os, time, sys

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    BatchSampler,
    SubsetRandomSampler,
)

from typing import Any, List, Optional


class SMILESCollater:
    def __call__(self, data):
        return self.__collate__(data)

    def __collate__(self, data):
        if type(data[0]) == tuple:
            data_new = tuple(map(list, zip(*data)))
        else:
            data_new = data

        return data_new


class SMILESDataset(Dataset):
    def __init__(self, reactants, products, tokenizer, augmenter=None):
        self.reactants = reactants
        self.products = products
        self.tokenizer = tokenizer

        if augmenter is None:
            self.augmenter = lambda x: x
        else:
            self.augmenter = augmenter

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, idx):
        return self.reactants[idx], self.products[idx]


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        custom_encoder=None,
        custom_decoder=None,
    ):
        super(TransformerModel, self).__init__()

        self.emb_encoder = nn.Embedding(n_tokens, d_model)
        self.emb_decoder = nn.Embedding(n_tokens, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.fc_out = nn.Linear(d_model, n_tokens)

        self.src_mask = None
        self.tgt_mask = None

        self.reset_parameters()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def reset_parameters(self):
        self.emb_encoder.reset_parameters()
        self.emb_decoder.reset_parameters()
        self.transformer._reset_parameters()
        nn.init.xavier_normal_(
            self.fc_out.weight
        )  

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != src.shape[0]:
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            self.src_mask = mask

        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.shape[0]:
            device = tgt.device
            mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
            self.tgt_mask = mask

        src = self.emb_encoder(src)
        src = self.pe(src)

        tgt = self.emb_decoder(tgt)
        tgt = self.pe(tgt)

        out = self.transformer(
            src, tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask,
        )
        out = self.fc_out(out)

        return out


# TODO I think this wrapper is much more general than just for transformers
class TransformerWrapper:
    def __init__(
        self,
        n_tokens,
        length,
        d_model=128,
        n_head=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_feedforward=256,
        dropout=0.0,
    ):

        self.model = TransformerModel(
            n_tokens,
            # length,
            d_model=d_model,
            n_head=n_head,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            d_feedforward=d_feedforward,
            dropout=dropout,
        )

    def predict(self, reactants):
        # This function from reactants mols/smiles to product mol/smiles would need the vectorizers .transform() and .reverse_transform()
        raise NotImplemented

    def sample_single(self, reactant, length, start_idx, stop_idx):
        length = length
        sample_vector = [0] * (length)
        sample_vector[0] = start_idx
        sample_vector_t = torch.tensor([sample_vector], device=self.device).long()
        for i in range(sample_vector_t.shape[1] - 1):
            out = self.model.forward(reactant, sample_vector_t)
            next_char_idx = torch.argmax(out[i])
            sample_vector_t[0, i + 1] = next_char_idx
            if next_char_idx == stop_idx:
                break
        return sample_vector_t

    def save(self, filename):
        raise NotImplemented

    def set_optimizer(self, learning_rate=0.001):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_device(self, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, reactant, product_in):
        return self.model.forward(reactant, product_in)

    def fit(self, train_dataset, val_set, epochs, batch_size=128, callbacks=[]):
        """trainset is a Dataset, providing reactant and product arrays
        
        val_set are pre_vectorized and tensorized (X_val_t, y_val_t)
        epochs is self_explanatory
        batch_size is needed for DataLoader
        callbacks, will be called at end of each epoch and sent a reference to the model
        
        """
        self.epochs = epochs
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        X_val_t = val_set[0]
        y_val_t = val_set[1]
        self.losses = []
        self.val_losses = []
        time0 = time.time()
        for e in range(self.epochs):
            self.model.train()  # Ensure the network is in "train" mode with e.g. dropouts active
            running_loss = 0
            time_e0 = time.time()
            for reactant_t, product_t in train_loader:
                # Push numpy to CUDA tensors
                reactant_t = torch.tensor(reactant_t, device=self.device).long()
                product_t = torch.tensor(product_t, device=self.device).long()

                # Training pass
                self.optimizer.zero_grad()  # Initialize the gradients, which will be recorded during the forward pass
                output = self.model.forward(
                    reactant_t, product_t[:, :-1]
                )  # Forward pass of the mini-batch # (seq, batch, tokens)

                output = output.transpose(0, 1)  # => Batch, seq, tokens
                output_t = output.transpose(1, 2)  # => Batch, tokens, seq
                # loss Expects, batch, classes, length
                loss = nn.CrossEntropyLoss()(
                    output_t, product_t[:, 1:]
                )  # .sum(dim=1)  # (batch)

                loss.backward()  # calculate the backward pass
                # Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()  # Optimize the weights

                running_loss += loss.item()
            else:
                time_pr_epoch = time.time() - time_e0
                samples_pr_sec = len(train_dataset) / time_pr_epoch

                with torch.no_grad():
                    self.model.eval()  # Evaluation mode
                    # TODO calculate the val_set in mini_batches to avoid memory issues
                    pred_val = self.model.forward(X_val_t[0:512], y_val_t[0:512, :-1])
                    pred_val = pred_val.transpose(0, 1)
                    pred_val = pred_val.transpose(1, 2)
                    val_loss = nn.CrossEntropyLoss()(
                        pred_val, y_val_t[0:512, 1:]
                    ).item()
                    self.model.train()

                train_loss = running_loss / len(train_loader)
                self.losses.append(train_loss)
                self.val_losses.append(val_loss)

                print(
                    "| epoch {:3d} | "
                    "samples/sec {:04.1f} | "
                    "loss {:3.3f} | val_loss {:3.3f} |".format(
                        e, samples_pr_sec, train_loss, val_loss
                    )
                )

                for callback in callbacks:
                    callback(self)


if __name__ == "__main__":
    main()

