from openai import OpenAI
from prettytable import PrettyTable
import argparse
import json
import os
import pandas as pd
import requests
import shutil
import sys

import vinge.blocking as blocking
import vinge.config as config
import vinge.score as score
import vinge.utils as utils


def main() -> int:
    parser = argparse.ArgumentParser(description="Link financial datasets on noisy names")
    parser.add_argument("task", help="[ configure | init | submit | status | fetch ]", type=str)
    parser.add_argument("--id", help="Unique identifier for this query", type=str)
    parser.add_argument("--ngram-candidates", help="Number of candidates derived from ngram embeddings", type=int)
    parser.add_argument("--mistral-candidates", help="Number of candidates derived from Mistral embeddings", type=int)
    parser.add_argument("--left", help="Left table path", type=str)
    parser.add_argument("--right", help="Right table path", type=str)
    parser.add_argument("--output-basedir", help="Base directory for results (append id)", type=str)

    args = parser.parse_args()

    if args.task == "configure":
        return run_configuration(args)
    elif args.task == "init":
        return run_init(args)
    elif args.task == "submit":
        return run_submit(args)
    elif args.task == "status":
        return run_status(args)

    parser.print_help()

    return 1


def run_configuration(args) -> int:
    """
    Set up initial configuration
    """
    if not os.path.exists(config.VINGE_DIR):
        os.makedirs(config.VINGE_DIR)

    wts = os.path.join(config.VINGE_DIR, config.MISTRAL_WEIGHTS_FILE)
    if not os.path.exists(wts):
        utils.fetch_file(config.MISTRAL_WEIGHTS_URI, wts)

    key = input("Enter OpenAI API key:")
    utils.set_openai_key(key)

    return 0


def run_init(args) -> int:
    """
    Initialize the matching batch job
    """
    # create job log record
    utils.create_job(args.id, args.output_basedir)

    # create job directory
    tgt = os.path.join(args.output_basedir, args.id)
    if os.path.exists(tgt):
        shutil.rmtree(tgt)

    os.makedirs(tgt)

    # add left/right to job directory
    shutil.copyfile(args.left, os.path.join(tgt, "left.csv"))
    shutil.copyfile(args.right, os.path.join(tgt, "right.csv"))

    # add blocking to job directory
    left = utils.read_namelist(args.left)
    right = utils.read_namelist(args.right)
    ngram = blocking.block(left, right, args.ngram_candidates, "ngram")
    mistral = blocking.block(left, right, args.mistral_candidates, "mistral")
    blocked = pd.concat([ngram, mistral], axis=0)
    blocked.to_csv(os.path.join(tgt, "blocked.csv"), header=True, index=False)

    # add batch request to job directory
    request = score.create_openai_batch_request(blocked)
    with open(os.path.join(tgt, "request.jsonl"), "w") as f:
        print(request, file=f)

    return 0


def run_submit(args) -> int:
    """
    Submit an existing job
    """
    client = OpenAI(api_key=utils.get_openai_key())

    job = utils.fetch_job(args.id)
    job_dir = os.path.join(job["output_basedir"], job["name"])

    batch_input_file = client.files.create(
        file=open(os.path.join(job_dir, "request.jsonl"), "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    with open(os.path.join(job_dir, "batch.json"), "w") as f:
        print(batch.to_json(), file=f)

    client.close()

    return 0


def run_status(args) -> int:
    """
    Get status of all pending jobs
    """
    client = OpenAI(api_key=utils.get_openai_key())

    records = []
    table = PrettyTable()
    table.field_names = ["Job", "Status", "Progress (%)"]

    for job in utils.fetch_job_log():
        job_dir = os.path.join(job["output_basedir"], job["name"])
        job_pth = os.path.join(job_dir, "batch.json")

        with open(job_pth, "r") as f:
            batch = json.load(f)

        batch = client.batches.retrieve(batch["id"]).to_dict()

        with open(job_pth, "w") as f:
            json.dump(batch, f)

        name = job["name"]
        status = batch["status"]
        if batch["request_counts"]["total"] > 0:
            progress = round(100 * batch["request_counts"]["completed"] / batch["request_counts"]["total"])
        else:
            progress = None

        table.add_row([name, status, progress])

    client.close()

    print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

