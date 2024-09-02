from functools import lru_cache
from typing import Any, Dict, List
import json
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


@lru_cache(maxsize=3)
def get_openai_key() -> str:
    with open(os.path.join(config.VINGE_DIR, "openai.key"), "r") as f:
        key = f.read().strip()
    return key


def set_openai_key(key: str) -> None:
    with open(os.path.join(config.VINGE_DIR, "openai.key"), "w") as f:
        print(key, file=f)
    return None


def ensure_job_log() -> None:
    if not os.path.exists(config.VINGE_JOBS):
        with open(config.VINGE_JOBS, "w") as f:
            json.dump([], f)
    return None


def fetch_job_log() -> List[Dict[str, Any]]:
    with open(config.VINGE_JOBS, "r") as f:
        res = json.load(f)
    return res


def fetch_job(job_name: str) -> Dict[str, Any]:
    jobs = fetch_job_log()
    return [j for j in jobs if j["name"] == job_name][0]


def write_job_log(jobs: List[Dict[str, Any]]) -> None:
    with open(config.VINGE_JOBS, "w") as f:
        json.dump(jobs, f)
    return None


def create_job(job_name: str, output_basedir: str) -> None:
    ensure_job_log()

    jobs = fetch_job_log()
    if job_name in [j["name"] for j in jobs]:
        raise ValueError(f"Job {job_name} already exists")

    jobs.append({"name": job_name, "output_basedir": output_basedir})

    write_job_log(jobs)

    return None


def delete_job(job_name: str) -> None:
    jobs = fetch_job_log()
    jobs = [j for j in jobs if j["name"] != job_name]
    write_job_log(jobs)
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
