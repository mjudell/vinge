from dataclasses import dataclass
from llama_cpp import Llama
from tqdm.auto import tqdm
import numpy as np
import os
import pandas as pd

import vinge.config as config


@dataclass
class Embeddings:
    raw: np.ndarray  # (n_samples, n_dim)
    idx: np.ndarray  # (n_samples)


def get_embedding_llama(verbose: bool = False) -> Llama:
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


def get_embeddings(names: pd.DataFrame) -> Embeddings:
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
    model = get_embedding_llama()
    embeddings = np.zeros((names.shape[0], config.MISTRAL_DIM), dtype=np.float64)
    i = 0
    for name in tqdm(names["name"].values, desc="Calculating name embeddings"):
        query = f"Abbreviated company name: {name.upper()}"
        embeddings[i, :] = np.mean(np.array(model.embed(query)), axis=0)
        i += 1
    return Embeddings(raw=embeddings, idx=names["id"].values)
