"""
Use local embeddings to find top right candidates for each left entry
"""
from dataclasses import dataclass
from llama_cpp import Llama
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from typing import List, Tuple
from uuid import uuid4
import numpy as np
import os
import pandas as pd

import vinge.config as config


@dataclass
class Embeddings:
    method: str
    emb: np.ndarray  # (n_samples, n_dim)
    ids: np.ndarray  # (n_samples)
    names: np.ndarray  # (nsamples,)


def _get_embedding_llama(verbose: bool = False) -> Llama:
    """
    Get the language model

    Parameters
    ----------
    verbose: bool
        Whether to instantiate the model in verbose mode

    Returns
    -------
    Llama
        The instantiated language model
    """
    model = Llama(
        model_path=os.path.join(config.VINGE_DIR, config.MISTRAL_WEIGHTS_FILE),
        embedding=True,
        verbose=verbose,
        n_gpu_layers=-1
    )
    return model


def _get_mistral_embeddings(names: pd.DataFrame) -> Embeddings:
    """
    Embed the names

    Parameters
    ----------
    names: pd.DataFrame
        [ id | name ] columns

    Returns
    -------
    Embeddings
        A representation of the embeddings
    """
    model = _get_embedding_llama()
    embeddings = np.zeros((names.shape[0], config.MISTRAL_DIM), dtype=np.float64)
    i = 0
    for name in tqdm(names["name"].values, desc="Computing Mistral embeddings"):
        query = f"Abbreviated company name. {name.upper()}"
        embeddings[i, :] = np.mean(np.array(model.embed(query)), axis=0)
        i += 1
    return Embeddings(
        method="mistral",
        emb=embeddings,
        ids=names["id"].values,
        names=names["name"].values)


def _get_ngram_embedder(names: List[str]) -> CountVectorizer:
    """
    Get a calibrated count vectorizer

    Parameters
    ----------
    names: List[str]
        The full set of left and right company names
    """
    model = Pipeline([
        ("count", CountVectorizer(ngram_range=(3, 6), analyzer="char_wb", max_features=16384, lowercase=True)),
        ("tfidf", TfidfTransformer()),
    ])
    model.fit(names)
    return model


def get_mistral_embeddings(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[Embeddings]:
    """
    Get mistral embeddings

    Parameters
    ----------
    left: pd.DataFrame
        [ id | name ]
    right: pd.DataFrame
        [ id | name ]

    Returns
    -------
    Tuple[Embeddings]
        The left and right embeddings
    """
    l_emb = _get_mistral_embeddings(left)
    r_emb = _get_mistral_embeddings(right)
    return l_emb, r_emb


def get_ngram_embeddings(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[Embeddings]:
    """
    Get character ngram embeddings

    Parameters
    ----------
    left: pd.DataFrame
        [ id | name ]
    right: pd.DataFrame
        [ id | name ]

    Returns
    -------
    Tuple[Embeddings]
        The left and right embeddings
    """
    names = np.concatenate([left["name"].values, right["name"].values])
    model = _get_ngram_embedder(names)

    l_emb = Embeddings(
        method="ngram",
        emb=model.transform(left["name"].values),
        ids=left["id"].values,
        names=left["name"].values
    )

    r_emb = Embeddings(
        method="ngram",
        emb=model.transform(right["name"].values),
        ids=right["id"].values,
        names=right["name"].values
    )

    return l_emb, r_emb


def block(left: pd.DataFrame, right: pd.DataFrame, num_candidates: int, method: str) -> pd.DataFrame:
    """
    Find the top right candidates for each left candidate

    Parameters
    ----------
    left: pd.DataFrame
        [ id | name ]
    right: pd.DataFrame
        [ id | name ]
    num_candidates: int
        Number of right candidates per left entry
    method: str
        [ ngram | mistral ]

    Returns
    -------
    pd.DataFrame
        uuid, l_id, r_id, l_name, r_name, block_method, block_similarity, block_rank
    """
    embedder = {
        "ngram": get_ngram_embeddings,
        "mistral": get_mistral_embeddings
    }[method]

    l_emb, r_emb = embedder(left, right)

    sim = cosine_similarity(l_emb.emb, r_emb.emb)

    l_idx = np.tile(np.arange(l_emb.ids.shape[0])[:, None], (num_candidates,)).ravel()
    l_ids = l_emb.ids[l_idx]
    l_names = l_emb.names[l_idx]

    r_idx = np.argsort(sim, axis=1)[:, -num_candidates:][:, ::-1].ravel()
    r_ids = r_emb.ids[r_idx]
    r_names = r_emb.names[r_idx]

    n_rows, n_cols = sim.shape
    sim = sim.ravel()
    sim = sim[l_idx * n_cols + r_idx]

    res = pd.concat([
        pd.Series([str(uuid4()) for _ in range(sim.shape[0])]).to_frame("uuid"),
        pd.Series(l_ids).to_frame("l_id"),
        pd.Series(r_ids).to_frame("r_id"),
        pd.Series(l_names).to_frame("l_name"),
        pd.Series(r_names).to_frame("r_name"),
        pd.Series([l_emb.method] * sim.shape[0]).to_frame("block_method"),
        pd.Series(sim).to_frame("block_similarity"),
    ], axis=1)

    res["block_rank"] = res.groupby(["l_id"]).block_similarity.rank(ascending=False).astype(int)

    res = res[res.block_rank <= num_candidates]

    return res
