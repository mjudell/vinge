# vinge

Link financial datasets using noisy company names.

# HOWTO

Linux only.

```bash
# (optional) enable gpu support on debian
bash vinge/gpu.sh

# install
git clone https://github.com/mjudell/vinge.git
pip install ./vinge

# configure (pull language model binaries, configure api keys, etc)
vinge configure

# example
vinge link \
    --ngram-candidates 3 \
    --mistral-candidates 7 \
    --left vinge/examples/ishares.csv \
    --right vinge/examples/13f.csv \
    --ouput vinge/examples/ishares_13f_links.csv
```

# Architecture

For each entity in the left list find the top N in the right list using Mistral embeddings and cosine similarity. Then use ChatGPT to select the best match. I also use character n-gram embeddings in case the Mistral embeddings don't turn up good candidates.

```mermaid
graph LR;
    A[Left List] --> B[Mistral Left Embedding]
    C[Right List] --> D[Mistral Right Embedding]
    B --> E[Top N by Cosine Similariy]
    D --> E
    E --> F[ChatGPT Closest Match]
```
