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

import copy

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
        """
        new_n_users: Total number of user rows of the model's embedding table post expansion
        new_n_items: Total number of item rows of the model's embedding table post expansion
        """
    def expand_embeddings(self, new_n_users: int, new_n_items: int):
        """
        Expand user and/or item embedding matrices to accommodate new entities.
        New rows are initialized with the mean of all existing embeddings so
        new users/items start with an average representation rather than random noise.
        """
        if new_n_users > self.n_users:
            old_weight = self.user_embedding.weight.data
            # mean of all existing user embeddings down column for each dimension keep the shape
            # as table with one row
            mean_emb = old_weight.mean(dim=0, keepdim=True)
            n_new = new_n_users - self.n_users
            # exand the table only for new rows and no new column and save it in new place
            new_rows = mean_emb.expand(n_new, -1).clone()
            new_weight = torch.cat([old_weight, new_rows], dim=0)

            # create new table with the right size becuase pytoch's nn.embedding size is fixed
            self.user_embedding = nn.Embedding(new_n_users, self.latent_dim)
            # set the expanded embedding and mark as trainable
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


        data = np.ones(len(self._coo_rows), dtype=np.float32)
        # build a matrix one row per user and one column per item and each entry is one if there
        # is an interaction between that item and user
        self.interaction_matrix = sp.coo_matrix(
            (data, (np.array(self._coo_rows), np.array(self._coo_cols))),
            shape=(self.n_users, self.n_items),
        )
        # calculate the adj matrix using the recbole implementation
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Invalidate cache after any expansion
        self.restore_user_e = None
        self.restore_item_e = None

    # Graph update 
    """
    user_ids: New users being added
    item_ids: New items being added
    """
    def add_interactions(self, user_ids: np.ndarray, item_ids: np.ndarray):
        """
        Add new user-item interactions to the graph and recompute the
        normalized adjacency matrix. Existing embeddings are preserved.
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

    # Negative sampling
    """
    users_np: Users we want a negative sample for 
    csr     : interaction matrix
    """
    def _sample_negatives(self, users_np: np.ndarray, csr) -> np.ndarray:
        """
        Draw one negative item per given user, guaranteed not to be an item
        that user has already interacted with — rejection sampling, matching
        RecBole's own Sampler (recbole/sampler/sampler.py) and the LightGCN
        paper's negative sampling: draw uniformly, then re-sample only the
        draws that collided with a true positive, repeating until clean.
        """
        n = len(users_np)
        neg_ids = np.random.randint(0, self.n_items, size=n)
        check_list = np.arange(n)
        while len(check_list) > 0:
            collide = [
                idx for idx in check_list
                if neg_ids[idx] in csr.indices[csr.indptr[users_np[idx]]: csr.indptr[users_np[idx] + 1]]
            ]
            if not collide:
                break
            collide = np.array(collide)
            neg_ids[collide] = np.random.randint(0, self.n_items, size=len(collide))
            check_list = collide
        return neg_ids

    # Incremental training

    def incremental_update(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_epochs: int = 50,
        learning_rate: float = 0.001,
        val_fraction: float = 0.1,
        patience: int = 5,
    ):
        """
        Run BPR gradient steps on new interactions only, warm-started from
        existing embeddings — much cheaper than full retraining. One
        rejection-sampled negative per positive (see _sample_negatives),
        matching RecBole's default sample_num=1 and the LightGCN paper's
        standard BPR formulation.

        A fraction of the batch (val_fraction) is held out as a validation
        slice, evaluated on BPR loss after every epoch. Training stops early
        once validation loss hasn't improved for `patience` consecutive
        epochs, and the best-validation-loss weights seen during this update
        are restored at the end — so additional epochs (up to n_epochs) are
        only spent while they still buy real improvement, instead of always
        running a fixed epoch count regardless of whether it's still helping.
        """

        # before updating the model mode is set to training mode. this is not neccesary for lightgcn
        # but recbole checks
        self.train()

        # As in paper implementation
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # CSR view of the interaction graph, used for negative-sampling
        # rejection checks below. Built once per update (not per epoch) —
        # already reflects this batch's new edges, since add_interactions()
        # is always called before incremental_update() by the caller.
        csr = self.interaction_matrix.tocsr()

        # Hold out a validation slice, fixed for the whole update, so
        # validation loss is comparable across epochs.
        n = len(user_ids)
        perm = np.random.permutation(n)
        n_val = max(1, int(n * val_fraction)) if n > 1 else 0
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_users = torch.LongTensor(user_ids[train_idx]).to(self.device)
        train_pos   = torch.LongTensor(item_ids[train_idx]).to(self.device)

        val_users = torch.LongTensor(user_ids[val_idx]).to(self.device)
        val_pos   = torch.LongTensor(item_ids[val_idx]).to(self.device)
        # Fixed validation negatives (not resampled per epoch) so validation
        # loss is comparable across epochs. Rejection-sampled so a "negative"
        # can never actually be an item this user already interacted with.
        val_neg_ids = self._sample_negatives(user_ids[val_idx], csr)
        val_neg = torch.LongTensor(val_neg_ids).to(self.device)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.state_dict())
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            # Rejection-sampled negatives — guaranteed not to be items this
            # user already interacted with (see _sample_negatives).
            neg_ids = self._sample_negatives(user_ids[train_idx], csr)
            neg_tensor = torch.LongTensor(neg_ids).to(self.device)

            # resets all stored gradients to zero before computing fresh ones
            # A gradient is just a number that answers:
            # "if I increase this weight slightly, does the loss go up or down, and by how much?"
            optimizer.zero_grad()

            # calculate the embeddings
            user_all_emb, item_all_emb = self.forward()

            u_emb   = user_all_emb[train_users]
            pos_emb = item_all_emb[train_pos]
            neg_emb = item_all_emb[neg_tensor]

            # calculate the scores
            pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
            neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)

            # measure the mistakes
            bpr_loss = self.mf_loss(pos_scores, neg_scores)
            reg_loss = self.reg_loss(
                self.user_embedding(train_users),
                self.item_embedding(train_pos),
                self.item_embedding(neg_tensor),
                require_pow=self.require_pow,
            )
            loss = bpr_loss + self.reg_weight * reg_loss

            # trace back which weight caused the error and fix it
            loss.backward()
            optimizer.step()

            # Validation check — did this epoch actually improve things?
            if len(val_idx) > 0: # this safe gaurd should never be used
                with torch.no_grad():
                    user_all_emb, item_all_emb = self.forward()
                    val_pos_scores = torch.mul(user_all_emb[val_users], item_all_emb[val_pos]).sum(dim=1)
                    val_neg_scores = torch.mul(user_all_emb[val_users], item_all_emb[val_neg]).sum(dim=1)
                    val_loss = self.mf_loss(val_pos_scores, val_neg_scores).item()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

        print(f"  Incremental update: ran {epoch + 1}/{n_epochs} epochs"
              + (f" (best val_loss={best_val_loss:.4f})" if len(val_idx) > 0 else ""))

        # Restore the best-validation-loss weights seen during this update
        # (not necessarily the weights from the very last epoch)
        if len(val_idx) > 0:
            self.load_state_dict(best_state)

        # set the model mode to evaluate
        self.eval()
        # Invalidate cache so next recommendation uses updated embeddings
        self.restore_user_e = None
        self.restore_item_e = None

        return epoch + 1, (best_val_loss if len(val_idx) > 0 else None)
