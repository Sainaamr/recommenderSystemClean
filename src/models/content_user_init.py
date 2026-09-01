"""
Content-aware cold-start initializer for new users.

When a new user's first interactions arrive during streaming, uses the
items they interacted with to find geographically and categorically
similar trained users, then returns a weighted average of their frozen
LightGCN embeddings as the new user's cold-start embedding.

Two schemas are supported, both reduced to the same shape: each item yields a
PRIMARY set and a SECONDARY set of content tokens, and users are compared by
histogram intersection over each, combined as alpha*primary + (1-alpha)*secondary.

  yelp   — primary: geohash cell (4 chars, ~39km x 19.5km); secondary: categories.
           Location dominates because a user in Phoenix cannot visit a Las Vegas
           business at all.

  ml-1m  — primary: genre; secondary: release decade. The roles are swapped
           relative to yelp because year barely discriminates: 63% of ml-1m films
           are 1990 or later and 59% fall in the 1990s alone, so a decade profile
           is near-identical for every user, while the 18 genres (1.65 per film)
           separate users cleanly.

Standalone — does not modify any existing model or experiment file.
"""

import pickle
import numpy as np
import pandas as pd
import pygeohash as pgh

"""
lat: latitude
lon: longitude
precision: geohash string length (4 chars ~= 39km x 19.5km)
"""
def _geo_bin(lat: float, lon: float, precision: int = 4):
    return pgh.encode(lat, lon, precision=precision)

"""
clean up the raw category of an item and turn it into a clean set
"""
def _tok(v) -> str:
    """
    Canonical token string for an ID cell.

    DataFrame.iterrows() upcasts a whole row to a single dtype, so on a file whose
    rating/timestamp columns are float64 an int64 user_id comes back as 1.0 and
    str() yields "1.0", which never matches RecBole's "1". Yelp is unaffected
    (its tokens are strings, so rows stay object dtype), which is why this only
    shows up on numeric-ID datasets like ml-1m.
    """
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _parse_genres(genre_str) -> set:
    """ml-1m genre:token_seq is space-separated, e.g. "Animation Children's Comedy"."""
    if not genre_str or pd.isna(genre_str):
        return set()
    return {g.strip() for g in str(genre_str).split() if g.strip()}


def _decade(year) -> set:
    """Release decade as a one-element set, so it slots into the same machinery."""
    try:
        return {str(int(year) // 10 * 10)}
    except (ValueError, TypeError):
        return set()


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
                 geo_precision: int = 4, schema: str = "yelp"):
        assert schema in ("yelp", "ml-1m"), f"unknown content schema {schema!r}"
        self.schema = schema
        self.alpha         = alpha # ensures locations is more significant than catalogue
        self.top_k         = top_k # how many of the most similar trained users to average together
        self.geo_precision = geo_precision # geohash string length
        self._item_geo      = {}   # iid → geohash string (trained items only)
        self._item_cats     = {}   # iid → set of category strings (trained items only)
        self._item_geo_by_token  = {} # raw item token → geo_bin
        self._item_cats_by_token = {}  # raw item token →  category set
        self._user_emb      = None # (n_trained, d) frozen embeddings
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
            token = _tok(row["user_id:token"])
            uid = user2id.get(token)
            item_token = _tok(row["item_id:token"])
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
            token = _tok(row["item_id:token"])
            iid = item2id.get(token)
            if iid is None and token not in needed_excluded_tokens:
                continue
            if self.schema == "ml-1m":
                geo  = _parse_genres(row.get("genre:token_seq", ""))
                cats = _decade(row.get("release_year:token"))
            else:
                try:
                    geo = {_geo_bin(float(row["latitude:float"]),
                                    float(row["longitude:float"]),
                                    self.geo_precision)}
                except (ValueError, TypeError):
                    geo = set()
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
            for geo in self._item_geo.get(iid, set()):
                geo_counts = user_geo_freq.setdefault(uid, {})
                geo_counts[geo] = geo_counts.get(geo, 0) + 1
            for cat in self._item_cats.get(iid, set()):
                cat_counts = user_cat_freq.setdefault(uid, {})
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # for debugging and reproducibility reasons
        self._trained_uids  = sorted(user_geo_freq.keys())
        self._trained_uids_arr = np.array(self._trained_uids)

        # Dense (n_trained_users, n_distinct_bins/cats) matrices, so
        # get_embedding() can score every trained user with one vectorized
        # numpy operation instead of a Python loop + dict lookups per user
        self._geo_bin_list  = sorted({g for freq in user_geo_freq.values() for g in freq})
        self._geo_bin_index = {g: i for i, g in enumerate(self._geo_bin_list)}
        self._cat_list      = sorted({c for freq in user_cat_freq.values() for c in freq})
        self._cat_index     = {c: i for i, c in enumerate(self._cat_list)}

        uid_to_row = {uid: i for i, uid in enumerate(self._trained_uids)}
        self._geo_matrix = np.zeros((len(self._trained_uids), len(self._geo_bin_list)), dtype=np.float32)
        self._cat_matrix = np.zeros((len(self._trained_uids), len(self._cat_list)), dtype=np.float32)
        for uid, freq in user_geo_freq.items():
            row = uid_to_row[uid]
            for g, count in freq.items():
                self._geo_matrix[row, self._geo_bin_index[g]] = count
        for uid, freq in user_cat_freq.items():
            row = uid_to_row[uid]
            for c, count in freq.items():
                self._cat_matrix[row, self._cat_index[c]] = count

        print(f"  ContentUserInit: recovered historical items for "
              f"{len(self._excluded_user_items)} users excluded from training")
        print(f"  ContentUserInit: built profiles for {len(self._trained_uids)} trained users")

    def save(self, path: str):
        """
        Persists everything build() computed to a single pickle file, the
        same role LightGCN's .pth checkpoint plays: a later run can load()
        this instead of re-parsing yelp.item / the historical interactions
        file and rebuilding the geo/category matrices from scratch.
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"  ContentUserInit: saved index to {path}")

    @classmethod
    def load(cls, path: str) -> "ContentUserInitializer":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        print(f"  ContentUserInit: loaded cached index from {path}")
        return obj

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

                geo  = self._item_geo_by_token.get(item_key, set())
                cats = self._item_cats_by_token.get(item_key, set())
            else:
                geo  = self._item_geo.get(item_key, set())
                cats = self._item_cats.get(item_key, set())
            for g in geo:
                new_geos[g] = new_geos.get(g, 0) + 1
            for cat in cats:
                new_cats[cat] = new_cats.get(cat, 0) + 1

        if not new_geos and not new_cats:
            return self._user_emb.mean(axis=0)
        # normalizing denominators for location and category signals
        total_geo = max(sum(new_geos.values()), 1)
        total_cat = max(sum(new_cats.values()), 1)

        # Score every trained user at once, via the dense matrices built in
        # build(). Slice down to only the columns the new user actually has
        # (typically a handful, out of up to ~1000+ total) before doing the
        # min/sum overlap — most columns would contribute 0 anyway (the new
        # user has no count there), so there's no point computing over them.
        geo_cols = [self._geo_bin_index[g] for g in new_geos if g in self._geo_bin_index]
        if geo_cols:
            sub = self._geo_matrix[:, geo_cols]
            vec = np.array([new_geos[self._geo_bin_list[c]] for c in geo_cols], dtype=np.float32)
            geo_overlap_all = np.minimum(vec[None, :], sub).sum(axis=1) / total_geo
        else:
            geo_overlap_all = np.zeros(len(self._trained_uids), dtype=np.float32)

        cat_cols = [self._cat_index[c] for c in new_cats if c in self._cat_index]
        if cat_cols:
            sub = self._cat_matrix[:, cat_cols]
            vec = np.array([new_cats[self._cat_list[c]] for c in cat_cols], dtype=np.float32)
            cat_overlap_all = np.minimum(vec[None, :], sub).sum(axis=1) / total_cat
        else:
            cat_overlap_all = np.zeros(len(self._trained_uids), dtype=np.float32)

        # calculate the similarity score as weighted blend of two overlap fractions
        # since location is more important than catalogy
        scores_all = self.alpha * geo_overlap_all + (1.0 - self.alpha) * cat_overlap_all

        mask = scores_all > 0
        if not mask.any():
            return self._user_emb.mean(axis=0)
        cand_scores = scores_all[mask]
        cand_uids = self._trained_uids_arr[mask]

        # Replicate the equivalent Python sorted(reverse=True) on (score, uid)
        # tuples exactly: ties break by DESCENDING uid, not ascending —
        # np.argsort alone (stable) would keep ties in ascending-uid order
        # and silently pick a different top-k set whenever ties sit right at
        # the cutoff, which happens often with this integer-ratio scoring.
        order = np.lexsort((-cand_uids, -cand_scores))[:self.top_k]
        top_scores = cand_scores[order]
        top_uids = cand_uids[order]

        # normalizing denominator for weight
        total_w = top_scores.sum()
        # weight acts as factor that makes user
        # with higher similarity to the new user more significant
        weights = (top_scores / total_w).astype(np.float32)
        return (weights[:, None] * self._user_emb[top_uids]).sum(axis=0)
