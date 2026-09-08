# Building a Custom LLM with nanoGPT

Class 4, Fall 26 · From Zero to AI Agents

Train Karpathy's actual **nanoGPT transformer** from scratch and inspect its learned
**word-token embeddings**. The classroom adaptation uses whole words and punctuation,
a small sentence corpus, and an explanatory notebook. nanoGPT itself supports different
tokenizers; changing the model name alone would not turn character tokens into words.

[Open in Colab](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb)
· [Assignment Google Doc](https://docs.google.com/document/d/1MQ3YQl2ywWZF7W5_l_91FiIp7pTYPO_3viI2JVapRcc/edit)
· [Assignment text](ASSIGNMENT.md)
· [3D embedding viewer](embedding-viewer.html)

## Start here

1. Open the notebook in Colab and save your own copy. The default CPU runtime is enough.
2. Choose corpus, training steps and learning rate in section 1. Write your reasons and prediction.
3. Try 10 steps for setup, then start with 3,000 steps and a learning rate of 0.001.
4. Run All. Inspect the data, IDs, vectors, gradient, first weight update, probabilities, attention and samples.
5. Download the results ZIP and the executed notebook separately after the final cell.
6. Download embedding-viewer.html and open it locally. Use **Open your checkpoint** to load checkpoint.json from your extracted results ZIP.
7. Explain the actual evidence in your own README and submit your public repository URL through the [course portal](https://submissions-portal-eight.vercel.app).

Locally, install the dependencies in requirements.txt, then open custom_llm.ipynb
with that Python environment. You can also run custom_llm.py directly after editing
its three settings. Colab generally already includes PyTorch. The notebook downloads
the pinned nanoGPT source if needed and verifies its hash; it downloads no model weights.

## What students should understand

| Idea | Evidence |
|---|---|
| Corpus and data | Actual sentences, deduplication and held-out documents |
| Tokens and IDs | Words/punctuation mapped to arbitrary integer IDs |
| Vectors and embeddings | One word's 64 numbers before/after, and the complete table |
| Neural networks | Weighted sums, GELU, attention blocks, residuals and parameters |
| Learning | Next-token loss, a real gradient and a parameter update |
| Context and prediction | Trained causal attention and next-token probabilities |
| Inference | Samples at three temperatures without weight updates |

The story is **examples → predictions → loss → gradients → updates → changed predictions**.
Character embeddings were already real embeddings in the original lab; the difference
here is that the units represent words rather than letters. A coordinate is not a
named concept, and a small language model is not a general chat assistant.

## Corpus and tokenizer

The default is a **synthetic classroom corpus**, generated visibly in the notebook.
It repeats sentence contexts around business, finance, food, transport, software,
health and education words. No category labels or coordinates are given to the model
or viewer. This deliberately controlled dataset makes distributional learning easy
to inspect; the resulting similarities are not evidence of broad semantic knowledge.

- Normalize case and punctuation spacing, deduplicate, then split documents 90/10.
- Build the vocabulary only from training documents.
- Reserve UNK for unknown words, BOS for document start and EOS for document end.
- Report the unknown-token rate on held-out text.
- Validation shares sentence templates with training. It tests new combinations
  within those templates, not generalization to unseen domains or writing styles.

Your own UTF-8 corpus needs at least 100 distinct documents, one per line, with at
most 47 word/punctuation tokens per document and at most 509 distinct training token
types. Longer documents and larger vocabularies are rejected explicitly. Use text
you are allowed to share: the results ZIP includes the corpus.

## The actual nanoGPT model

[nanogpt_model.py](nanogpt_model.py) is an unchanged copy of Karpathy's
[model.py at commit 3adf61e](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py).
Its [MIT license](NANOGPT_LICENSE) is included.

The classroom configuration uses 2 blocks, 4 heads, 64-dimensional token/position
embeddings, a 48-token context, LayerNorm, GELU, residual connections and tied
input/output embeddings. PyTorch handles autograd; batched AdamW replaces the old
handwritten scalar training loop. Word tokenization and the teaching/evaluation/export
helpers are classroom additions, not claims about nanoGPT's default tokenizer.

The upstream repository now labels nanoGPT deprecated in favor of nanochat.
We deliberately pin nanoGPT here because this assignment is about its compact,
inspectable GPT implementation, not adopting a production training stack.

## Viewer

Open the single offline HTML file. It bundles the reference model's **actual recorded
initial and final token lookup embeddings**. Drag to rotate, scroll or use buttons to
zoom, and select a word from the menu or click a dot. Scroll the vector panel for all
64 coordinates. Selected words and their three closest neighbors are labeled.

Both states share a PCA center, basis and scale. The default projection retains
40.9% of pooled variance, so proximity in 3D can distort the full space. Neighbor
rankings use cosine similarity across all 64 coordinates. Movement lines connect
endpoints, not intermediate training trajectories. These are token lookup embeddings,
not position embeddings or context-dependent representations after attention.

Load your own checkpoint.json to see your actual run, including its saved initial
table. Files stay on your device. Legacy character checkpoints remain supported;
when no initial table is present, before/after comparison is disabled.

## Measured reference run

The complete notebook was executed in order with 3,000 steps and learning rate 0.001.
It learned 136 word/punctuation/special-token vectors. A separate 10-step setup run
also completed. These are fixed panels of 20 documents per split, not full-corpus loss.

| Step | Training panel loss | Validation panel loss |
|---|---:|---:|
| 0 | 4.9238 | 4.9247 |
| 1500 | 0.6929 | 0.7113 |
| 3000 | 0.6956 | 0.7057 |

Final examples include “our school has a question about the new educator and lesson .”
and “the consumer compared the merchandise after checking the price .”
The measured nearest neighbors of customer are client, buyer and subscriber.
Their similar designed contexts explain this result; it does not prove general understanding.

Inspect [the executed notebook](examples/custom_llm.executed.ipynb),
[complete reference evidence](examples/reference/), and
[results ZIP](examples/reference.zip). Use your own outputs in your submission.

## Results and longer training

Every run saves config.json, corpus.txt, split.json, tokenization.json, inspection.json,
history.json, training.csv, training_summary.json, training_curves.svg, the sample
timeline, temperature_comparison.json, checkpoint.json and model.pt.

- **checkpoint.json:** token labels and initial/final embedding tables for the viewer.
- **model.pt:** all network weights and model settings for inference.
- Neither includes the complete optimizer/random state for exact training resume.
- To train longer, set TRAINING_STEPS to 5000 or 10000 and Run All from the top.
  Compare validation loss and samples. More steps can overfit and are not required.
- Interrupted training can be followed by the remaining save cells. Other failures
  require correcting the cause; do not assume a complete ZIP was saved.

Use [STUDENT_README.md](STUDENT_README.md) to organize your explanation.
The [historical microgpt lab](legacy/README.md) is preserved separately; its results
must not be presented as results of this word-token model.

For maintainers: run python3 build_embedding_viewer.py to refresh the bundled
reference vectors, then node test_embedding_viewer.cjs to verify PCA, similarities,
checkpoint consistency and import validation.
