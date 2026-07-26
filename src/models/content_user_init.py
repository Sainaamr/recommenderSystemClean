"""
Content-aware cold-start initializer for new users.

When a new user's first interactions arrive during streaming, uses the
items they interacted with to find geographically and categorically
similar trained users, then returns a weighted average of their frozen
LightGCN embeddings as the new user's cold-start embedding.

Location is the primary signal (alpha=0.7), categories are secondary.
Geographic sections use lat/lon binning at 0.5-degree resolution (~50km).

Standalone — does not modify any existing model or experiment file.
"""

import numpy as np
import pandas as pd

"""
lat: latitude
lon: longitude  
precision: 50km resolution
Buckets a precise location into a coarser gride cell
the "grid mapper" approach
"""
def _geo_bin(lat: float, lon: float, precision: float = 0.5):
    return (round(lat / precision) * precision, round(lon / precision) * precision)

"""
clean up the raw category of an item and turn it into a clean set
"""
def _parse_categories(cat_str) -> set:
    if not cat_str or pd.isna(cat_str):
        return set()
    cats = [c.strip() for c in str(cat_str).replace("\\/", "/").split(",")]
    return {c for c in cats if c}


class ContentUserInitializer:
    """
    Two-step index:
      build()         — call once after historical training
      get_embedding() — call at runtime for each new user
    """

    def __init__(self, alpha: float = 0.7, top_k: int = 20,
                 geo_precision: float = 0.5):
        self.alpha         = alpha # ensures locations is more significant than catalogue
        self.top_k         = top_k # how many of the most similar trained users to average together
        self.geo_precision = geo_precision # grid-cell size
        self._item_geo      = {}   # iid → geo_bin tuple (trained items only)
        self._item_cats     = {}   # iid → set of category strings (trained items only)
        self._item_geo_by_token  = {} # raw item token → geo_bin
        self._item_cats_by_token = {}  # raw item token →  category set
        self._user_emb      = None # (n_trained, d) frozen embeddings
        self._user_geo_freq = {}   # uid → {geo_bin: count} how often user interacted with that geo_bin
        self._user_cat_freq = {}   # uid → {category: count}
        self._trained_uids  = []   # sorted list of uids that have geo profiles
        self._excluded_user_items = {} # raw user token → list of historical iids

    # Index construction
        """
        item_meta_path        : path to yelp.item
        historical_inter_path : path to yelp-historical.inter (ratings >= 3 kept)
        user2id / item2id     : token → internal-id mappings from RecBole
        user_emb              : (n_users, d) numpy array of frozen embeddings
        """
    def build(self, item_meta_path: str, historical_inter_path: str,
              user2id: dict, item2id: dict, user_emb: np.ndarray):

        self._user_emb = user_emb.astype(np.float32)

        df_hist = pd.read_csv(historical_inter_path, sep="\t")

        trained_pairs = []             # (uid, iid) to count geo/cat for
        needed_excluded_tokens = set() # excluded items actually referenced by an excluded user

        # for every historical row if it was excluded due to interaction limit
        # item will be added to excluded_user_items either by id or raw token
        for _, row in df_hist.iterrows():
            token = str(row["user_id:token"])
            uid = user2id.get(token)
            item_token = str(row["item_id:token"])
            iid = item2id.get(item_token)

            if uid is None:
                item_key = iid if iid is not None else item_token
                if iid is None:
                    needed_excluded_tokens.add(item_token)
                self._excluded_user_items.setdefault(token, [])
                if item_key not in self._excluded_user_items[token]:
                    self._excluded_user_items[token].append(item_key)
                continue

            if iid is None or uid >= user_emb.shape[0]:
                continue

            # storing of user items that the model is trained on
            trained_pairs.append((uid, iid))

        # Item geo + category: trained items, plus only the
        # excluded items referenced by an excluded user's dropped interaction
        df_item = pd.read_csv(item_meta_path, sep="\t", low_memory=False)
        for _, row in df_item.iterrows():
            token = str(row["item_id:token"])
            iid = item2id.get(token)
            if iid is None and token not in needed_excluded_tokens:
                continue
            try:
                geo = _geo_bin(float(row["latitude:float"]),
                               float(row["longitude:float"]),
                               self.geo_precision)
            except (ValueError, TypeError):
                geo = None
            cats = _parse_categories(row.get("categories:token_seq", ""))
            if iid is None:
                self._item_geo_by_token[token]  = geo
                self._item_cats_by_token[token] = cats
            else:
                self._item_geo[iid]  = geo
                self._item_cats[iid] = cats

        print(f"  ContentUserInit: indexed {len(self._item_geo)} trained items, "
              f"{len(self._item_geo_by_token)} recovered excluded items")
        # record how frequent a user interacted with a catalogue and geo locaiton
        user_geo_freq: dict = {}
        user_cat_freq: dict = {}
        for uid, iid in trained_pairs:
            geo = self._item_geo.get(iid)
            if geo:
                geo_counts = user_geo_freq.setdefault(uid, {})
                geo_counts[geo] = geo_counts.get(geo, 0) + 1
            for cat in self._item_cats.get(iid, set()):
                cat_counts = user_cat_freq.setdefault(uid, {})
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        self._user_geo_freq = user_geo_freq
        self._user_cat_freq = user_cat_freq

        # for debugging and reproducibility reasons
        self._trained_uids  = sorted(self._user_geo_freq.keys())

        print(f"  ContentUserInit: recovered historical items for "
              f"{len(self._excluded_user_items)} users excluded from training")
        print(f"  ContentUserInit: built profiles for {len(self._trained_uids)} trained users")

    def get_excluded_history(self, token: str) -> list:
        """
        Historical items for a user excluded from training due to lack of enough interactions
        , looked up by their raw token (before they were assigned a uid). Each entry is
        either an int (iid, for items that survived into the trained
        vocabulary) or a str (raw item token, for items that were also
        excluded)
        """
        return self._excluded_user_items.get(token, [])

    # Runtime cold-start embedding

    def get_embedding(self, iids: list) -> np.ndarray:
        """
        iids : list of internal item IDs (int) and/or raw item tokens (str)
        from the new user's first interactions
        int: is a trained item, looked up via _item_geo/_item_cats;
        str: is an item that was itself excluded from training (no iid exists for it), looked up via
        _item_geo_by_token/_item_cats_by_token
        Returns a (d,) numpy array
        the cold-start embedding Falls back to global mean if no neighbors found.
        """
        if not iids or not self._trained_uids:
            return self._user_emb.mean(axis=0)

        # New user's geo and category signal from their first interactions
        new_geos: dict = {}
        new_cats: dict = {}
        for item_key in iids:
            if isinstance(item_key, str):

                geo  = self._item_geo_by_token.get(item_key)
                cats = self._item_cats_by_token.get(item_key, set())
            else:
                geo  = self._item_geo.get(item_key)
                cats = self._item_cats.get(item_key, set())
            if geo:
                new_geos[geo] = new_geos.get(geo, 0) + 1
            for cat in cats:
                new_cats[cat] = new_cats.get(cat, 0) + 1

        if not new_geos and not new_cats:
            return self._user_emb.mean(axis=0)
        # normalizing denominators for location and category signals
        total_geo = max(sum(new_geos.values()), 1)
        total_cat = max(sum(new_cats.values()), 1)

        # Score each trained user: location dominates, categories refine
        scores = []
        for uid in self._trained_uids:
            geo_freq = self._user_geo_freq.get(uid, {})
            cat_freq = self._user_cat_freq.get(uid, {})
            # go over every geo bin a new user interacted with
            # for each geo bin take the min of frequency of interaction of new user and
            # this trained user and sum it all and normalize it
            geo_overlap = sum(
                min(new_geos[g], geo_freq.get(g, 0)) for g in new_geos
            ) / total_geo

            cat_overlap = sum(
                min(new_cats[c], cat_freq.get(c, 0)) for c in new_cats
            ) / total_cat
            # calculate the similarity score as weighted blend of two overlap fractions
            # since location is more important than catalogy
            score = self.alpha * geo_overlap + (1.0 - self.alpha) * cat_overlap
            if score > 0:
                scores.append((score, uid))

        if not scores:
            return self._user_emb.mean(axis=0)
        # sort based on the highest score and retrieve the top trained users
        scores.sort(reverse=True)
        top = scores[:self.top_k]
        # normalizing denominator for weight
        total_w = sum(s for s, _ in top)
        # initiate the embedding for new user
        d   = self._user_emb.shape[1]
        emb = np.zeros(d, dtype=np.float32)

        # weight acts as factor that makes user
        # with higher similarity to the new user more significant
        for w, uid in top:
            emb += (w / total_w) * self._user_emb[uid]

        return emb
