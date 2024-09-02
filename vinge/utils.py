import os
import pandas as pd
import requests

import vinge.config as config


def fetch_file(uri: str, pth: str) -> None:
    """
    Download a file to local disk

    Parameters
    ----------
    uri: str
        Path to the file
    pth: str
        Local path where file stored
    """
    with requests.get(uri, stream=True) as r:
        r.raise_for_status()
        with open(pth, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return None


def get_openai_key() -> str:
    """
    Get the user's openai API key
    """
    with open(os.path.join(config.VINGE_DIR, "openai.key"), "r") as f:
        key = f.read().strip()
    return key


def set_openai_key(key: str) -> None:
    """
    Set the user's openai API key
    """
    with open(os.path.join(config.VINGE_DIR, "openai.key"), "w") as f:
        print(key, file=f)
    return None


def read_namelist(pth: str) -> pd.DataFrame:
    """
    Read a list of company names

    Parameters
    ----------
    pth: str
        Path on disk of name list
    """
    df = pd.read_csv(pth, header=0, dtype=object)
    assert df.columns[0] == "id" and df.columns[1] == "name"
    return df
