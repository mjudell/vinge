# vinge

Link financial datasets using noisy company names.

# Architecture

For each entity in the left list find the top N in the right list using Mistral embeddings and cosine similarity. Then use ChatGPT to select the best match.

```mermaid
A[Left List] --> B[Mistral Left Embedding]
B[Right List] --> C[Mistral Right Embedding]
B --> D[Top N by Cosine Similariy]
C --> D
D --> E[ChatGPT Closest Match]
```

