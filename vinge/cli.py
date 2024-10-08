from openai import OpenAI
from prettytable import PrettyTable
import argparse
import json
import nltk
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
    parser.add_argument("task", help="[ configure | init | submit | status | fetch | link ]", type=str)
    parser.add_argument("--job", help="Unique identifier for this job", type=str)
    parser.add_argument("--ngram-candidates", help="Number of candidates derived from ngram embeddings", type=int)
    parser.add_argument("--mistral-candidates", help="Number of candidates derived from Mistral embeddings", type=int)
    parser.add_argument("--left", help="Left table path", type=str)
    parser.add_argument("--right", help="Right table path", type=str)
    parser.add_argument("--output-basedir", help="Base directory for results (append job)", type=str)

    args = parser.parse_args()

    if args.task == "configure":
        return run_configuration(args)
    elif args.task == "init":
        return run_init(args)
    elif args.task == "submit":
        return run_submit(args)
    elif args.task == "status":
        return run_status(args)
    elif args.task == "fetch":
        return run_fetch(args)
    elif args.task == "link":
        return run_link(args)

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
    utils.create_job(args.job, args.output_basedir)

    # create job directory
    tgt = os.path.join(args.output_basedir, args.job)
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

    job = utils.fetch_job(args.job)
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


def run_fetch(args) -> int:
    """
    Fetch results from a completed job
    """
    job = utils.fetch_job(args.job)
    job_dir = os.path.join(job["output_basedir"], job["name"])
    job_pth = os.path.join(job_dir, "batch.json")

    with open(job_pth, "r") as f:
        batch = json.load(f)

    client = OpenAI(api_key=utils.get_openai_key())

    resp = client.files.content(batch["output_file_id"])

    client.close()

    with open(os.path.join(job_dir, "results.jsonl"), "w") as f:
        print(resp.text, file=f)

    results = []
    for line in resp.text.splitlines():
        r = json.loads(line)
        result = {
            "uuid": r["custom_id"],
            "gpt_prob": score.parse_response(r["response"]["body"]["choices"][0]["message"]["content"])
        }
        results.append(result)

    results = pd.DataFrame(results)

    blocked = pd.read_csv(os.path.join(job_dir, "blocked.csv"), header=0)

    results = pd.merge(blocked, results, on=["uuid"], how="left", validate="one_to_one")

    assert results.gpt_prob.notnull().all()

    results.to_csv(os.path.join(job_dir, "results.csv"), header=True, index=False)

    utils.delete_job(args.job)

    return 0


def run_link(args) -> int:
    """
    Pick the best links from the results
    """
    results = pd.read_csv(os.path.join(args.output_basedir, "results.csv"), header=0)

    gpt_prob = results.groupby(["l_id", "r_id"]) \
                      .gpt_prob.mean() \
                      .to_frame("gpt_prob") \
                      .reset_index(drop=False)

    max_prob = gpt_prob.groupby("l_id") \
                       .gpt_prob.max() \
                       .to_frame("max_gpt_prob") \
                       .reset_index(drop=False)

    df = pd.merge(results, max_prob, on=["l_id"], how="left", validate="many_to_one")

    df = df[df.gpt_prob == df.max_gpt_prob]

    df = df.sort_values(by=["l_id", "r_id", "block_method"])
    df = df.groupby(["l_id", "r_id"]).first().reset_index(drop=False)

    link_counts = df.groupby("l_id").r_id.nunique()

    if link_counts.max() > 1:
        errors = df[df.l_id.isin(link_counts[link_counts>1].index)]
        errors = errors[["l_id", "r_id", "l_name", "r_name", "max_gpt_prob"]].rename(columns={"max_gpt_prob":"gpt_prob"})
        errors = errors.sort_values(by=["l_id", "r_id"], ascending=True)
        utils.print_table(errors)
        print("Detected duplicate best links")
        return 1

    df = df[["l_id", "r_id", "l_name", "r_name", "max_gpt_prob"]].rename(columns={"max_gpt_prob":"gpt_prob"})

    df["edit_distance"] = df.apply(lambda row: nltk.edit_distance(row.l_name, row.r_name), axis=1)

    df.to_csv(os.path.join(args.output_basedir, "links.csv"), header=True, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

