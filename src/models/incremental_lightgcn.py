"""
IncrementalLightGCN — extends RecBole's LightGCN with incremental graph updates.

Instead of retraining from scratch when new interactions arrive, this class:
  1. Adds new edges to the interaction graph (supports new users/items)
  2. Recomputes the normalized adjacency matrix
  3. Runs a small number of gradient steps warm-started from existing embeddings

New users and items are handled by expanding the embedding matrices on the fly.
New embeddings are initialized with the mean of all existing embeddings so they
start with a reasonable representation rather than random noise.

Usage:
    # Load a pre-trained RecBole LightGCN checkpoint
    model = IncrementalLightGCN.from_checkpoint("saved/LightGCN-xxx.pth", config, dataset)

    # Simulate new interactions arriving (new user/item IDs handled automatically)
    model.add_interactions(new_user_ids, new_item_ids)
    model.incremental_update(new_user_ids, new_item_ids, n_epochs=5)

    # Recommend for a user (same API as RecBole LightGCN)
    scores = model.full_sort_predict(interaction)
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from recbole.model.general_recommender import LightGCN


class IncrementalLightGCN(LightGCN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # Track all interactions as a mutable COO structure
        self._coo_rows = list(self.interaction_matrix.row)
        self._coo_cols = list(self.interaction_matrix.col)

    #  Load from RecBole checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config, dataset) -> "IncrementalLightGCN":

        # create instance of IncrementalLightGCN
        model = cls(config, dataset)
        # read a file from checkpoint and set the weights
        ckpt = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")
        return model

    # Embedding expansion for new users/items

    def expand_embeddings(self, new_n_users: int, new_n_items: int):
        """
        Expand user and/or item embedding matrices to accommodate new entities.
        New rows are initialized with the mean of all existing embeddings so
        new users/items start with an average representation rather than random noise.

        Called automatically by add_interactions when new IDs are detected.
        Can also be called directly for no_update strategy to register new
        entities without touching the graph.

        Example:
            model has 6022 users. A new user gets assigned ID 6022.
            expand_embeddings(6023, ...) adds one row to user_embedding
            initialized as mean of rows 0..6021.
        """
        if new_n_users > self.n_users:
            old_weight = self.user_embedding.weight.data
            # mean of all existing user embeddings → reasonable starting point
            mean_emb = old_weight.mean(dim=0, keepdim=True)
            n_new = new_n_users - self.n_users
            new_rows = mean_emb.expand(n_new, -1).clone()
            new_weight = torch.cat([old_weight, new_rows], dim=0)
            self.user_embedding = nn.Embedding(new_n_users, self.latent_dim)
            self.user_embedding.weight = nn.Parameter(new_weight)
            self.user_embedding.to(self.device)
            self.n_users = new_n_users
            print(f"  Expanded user embeddings: {new_n_users - n_new} → {new_n_users} (+{n_new} new users)")

        if new_n_items > self.n_items:
            old_weight = self.item_embedding.weight.data
            mean_emb = old_weight.mean(dim=0, keepdim=True)
            n_new = new_n_items - self.n_items
            new_rows = mean_emb.expand(n_new, -1).clone()
            new_weight = torch.cat([old_weight, new_rows], dim=0)
            self.item_embedding = nn.Embedding(new_n_items, self.latent_dim)
            self.item_embedding.weight = nn.Parameter(new_weight)
            self.item_embedding.to(self.device)
            self.n_items = new_n_items
            print(f"  Expanded item embeddings: {new_n_items - n_new} → {new_n_items} (+{n_new} new items)")

        # Rebuild interaction matrix and adjacency matrix with new shape.
        # Edges stay the same — only the matrix dimensions grow.
        data = np.ones(len(self._coo_rows), dtype=np.float32)
        self.interaction_matrix = sp.coo_matrix(
            (data, (np.array(self._coo_rows), np.array(self._coo_cols))),
            shape=(self.n_users, self.n_items),
        )
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Invalidate cache after any expansion
        self.restore_user_e = None
        self.restore_item_e = None

    # Graph update 

    def add_interactions(self, user_ids: np.ndarray, item_ids: np.ndarray):
        """
        Add new user-item interactions to the graph and recompute the
        normalized adjacency matrix. Existing embeddings are preserved.

        New users/items (IDs outside current matrix size) are handled
        automatically — embeddings are expanded with mean initialization.
        """
        # Expand embeddings if any new user/item IDs are outside current range
        max_user = int(user_ids.max()) + 1 if len(user_ids) > 0 else self.n_users
        max_item = int(item_ids.max()) + 1 if len(item_ids) > 0 else self.n_items
        if max_user > self.n_users or max_item > self.n_items:
            self.expand_embeddings(
                max(max_user, self.n_users),
                max(max_item, self.n_items),
            )

        self._coo_rows.extend(user_ids.tolist())
        self._coo_cols.extend(item_ids.tolist())

        # Rebuild interaction matrix with new edges and updated shape
        data = np.ones(len(self._coo_rows), dtype=np.float32)
        self.interaction_matrix = sp.coo_matrix(
            (data, (np.array(self._coo_rows), np.array(self._coo_cols))),
            shape=(self.n_users, self.n_items),
        )

        # Recompute normalized adjacency matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Invalidate cached embeddings so forward() recomputes them
        self.restore_user_e = None
        self.restore_item_e = None

    # ── Incremental training ──────────────────────────────────────────────────

    def incremental_update(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_epochs: int = 5,
        learning_rate: float = 0.001,
        n_neg_samples: int = 1,
    ):
        """
        Run a small number of BPR gradient steps on new interactions only.
        Warm-starts from existing embeddings — much cheaper than full retraining.
        !!!!!!!
        n_neg_samples for later
        !!!!!!!
        """

        # before updating the model mode is set to training mode. this is not neccesary for lightgcn
        # but recbole checks
        self.train()

        # As in paper implementation
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        user_tensor = torch.LongTensor(user_ids).to(self.device)
        pos_tensor  = torch.LongTensor(item_ids).to(self.device)

        for epoch in range(n_epochs):
            # Sample random negatives (items not in this batch's positives).
            # The idea is that most likely the randomly picked id is an item that the user didn't interact with
            neg_ids = np.random.randint(0, self.n_items, size=len(item_ids))
            neg_tensor = torch.LongTensor(neg_ids).to(self.device)

            # resets all stored gradients to zero before computing fresh ones
            # A gradient is just a number that answers:
            # "if I increase this weight slightly, does the loss go up or down, and by how much?"
            optimizer.zero_grad()

            # calculate the embeddings
            user_all_emb, item_all_emb = self.forward()

            u_emb   = user_all_emb[user_tensor]
            pos_emb = item_all_emb[pos_tensor]
            neg_emb = item_all_emb[neg_tensor]

            # calculate the scores
            pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
            neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)

            # measure the mistakes
            bpr_loss = self.mf_loss(pos_scores, neg_scores)
            reg_loss = self.reg_loss(
                self.user_embedding(user_tensor),
                self.item_embedding(pos_tensor),
                self.item_embedding(neg_tensor),
                require_pow=self.require_pow,
            )
            loss = bpr_loss + self.reg_weight * reg_loss

            # trace back which weight caused the error and fix it
            loss.backward()
            optimizer.step()

        # set the model mode to evaluate
        self.eval()
        # Invalidate cache so next recommendation uses updated embeddings
        self.restore_user_e = None
        self.restore_item_e = None
