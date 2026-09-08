# Class 4 Assignment: Building a Custom LLM

Choose data, train a tiny language model, and explain what its numbers and outputs show.

## Overview

Use the ready-made notebook to understand the basics of a language model: corpus, tokens, vectors, embeddings, neural networks, and learning. It closely follows Andrej Karpathy's microgpt, using plain Python and a tiny character-level transformer. Choose your corpus, training steps, and learning rate, then submit one public GitHub repository URL with your executed notebook and evidence. This miniature model generates short strings, not general chat answers.

## What You Are Submitting

- Your executed custom_llm.ipynb, saved after the final run with settings, inspections, samples, and plots visible. Submit your own final-run version, not the unexecuted starter.
- Text samples from the untrained, halfway, and final model, using the same generation settings.
- A training/validation loss plot and the full table of measured values from the fixed evaluation panels.
- An inspection of one token's ID and embedding vector before and after training, next-token probabilities, and one real gradient and parameter update.
- A short explanation of the learning process, a temperature comparison, one limitation, and a proposed next experiment.

## Open the Notebook

- [Custom LLM sample project on GitHub](https://github.com/pepealonso95/custom-llm)
- Local Jupyter or VS Code: open custom_llm.ipynb with a Python 3 kernel. The model uses only the standard library; a notebook environment supplies the display and editing tools.
- [Open custom_llm.ipynb in Google Colab. The default CPU runtime is enough.](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb)
- Read [Karpathy's microgpt explanation](https://karpathy.github.io/2026/02/12/microgpt/) alongside the notebook. The model keeps its one layer, four attention heads, 16-number embeddings, short context, scalar backpropagation, and Adam updates. Classroom additions make data, evaluation, and learning visible.

## Your Three Choices

- Corpus: use the supplied names dataset, or your own UTF-8 file with at least 100 distinct short documents, one per line. Each document must be 1–15 characters, such as a name, place name, or product label. Explain where the data came from and what patterns it could teach.
- Training steps: a positive whole number of weight updates. Try 10 for setup, then 1,000 as a starting training budget. A step is not an entire pass through the corpus. Runtime and sample quality depend on your machine, data, and choices.
- Learning rate: the initial size of the optimizer's updates. Start with 0.01, as in microgpt. The notebook decreases it during training. Explain why excessively large or small updates could be a problem.
- Edit those three settings in section 1 and write your prediction before training. You can keep the defaults, provided you explain your choices.
- The 16-position context window keeps this model small enough to inspect. Longer lines are rejected rather than silently truncated. Use data you have permission to share, without confidential or personal records.
- Other settings are optional experiments. Change one at a time and explain it. Complete and understand the baseline notebook first.

## Evaluate the Model Fairly

- Keep the same split, evaluation panels, seed, and baseline generation settings before and after training. Duplicate lines are removed before a 90/10 document split; validation documents never supply weight updates.
- The loss plot uses fixed panels of at most 20 training and 20 validation documents. Report every measured value and both panel sizes. These are small estimates, not full-corpus measurements.
- Show all saved samples, including empty or garbled strings. Use the same starting token and sampling seed for the temperature comparison. Different corpora and vocabularies do not produce directly comparable loss scores.
- Falling training loss alone does not demonstrate generalization. Compare held-out loss and samples too. Plausible names do not establish truth or human understanding. Report lack of improvement honestly.

## Save Your Results

Each Run All creates a new llm_runs/ folder and a ZIP. In Colab, download the ZIP before ending the session. Also download the executed notebook separately after the run; the results ZIP does not contain the currently open notebook.

- The folder contains config.json, corpus.txt, split.json, tokenization.json, inspection.json, history.json, training.csv, training_summary.json, checkpoint.json, training_curves.svg, the samples/ timeline, and temperature_comparison.json.
- If you interrupt training, continue through the inspection, plot, and download cells. Report the completed steps and interruption. A failed run saves available evidence; correct the settings and rerun from the top for a fresh experiment.

## README Requirements

The README is the grading entry point. A reader should be able to follow your experiment, inspect the evidence, and understand your explanation without rerunning the notebook.

- A brief overview, the source of your corpus, and instructions to open and run your notebook.
- Your three choices and reasons, plus the number of unique documents, vocabulary size, and the train/validation split.
- What you expected before training, followed by what you actually observed in the same run.
- Actual completed steps, elapsed time, hardware, and the model's parameter count. Identify interrupted or failed runs clearly.
- Explain corpus, tokens, IDs, vectors, embeddings, neural-network weights, loss, and learning using actual notebook examples. Trace one character from text to its ID and 16-number vector, then explain one saved gradient and weight update.
- Explain how attention uses earlier context, how probabilities become generated characters, and how temperature changes sampling without updating weights. Finish with one observed limitation and one proposed next experiment.

## Suggested Workflow

- Open the sample notebook, save your own copy, and read the explanatory cells as you go.
- Choose your corpus, training steps, and learning rate. Write your reasons and prediction. A 10-step run checks setup; use a meaningful training budget for the final experiment.
- Select Run All. Let the data inspection, untrained evaluation, training, final inspection, and evidence-saving cells finish in order.
- Compare token IDs and embedding vectors, inspect the first weight update, and read the sample timeline alongside both loss curves. Compare the three temperatures without retraining.
- Save the results ZIP and the executed notebook separately, write your explanation in the README, and publish your notebook and selected evidence on GitHub.

## Starter Prompt for Your AI Assistant

Help me work through custom_llm.ipynb for Class 4. Before training, ask me for my corpus, training steps, and learning rate; explain these choices briefly and wait for my answer. Help me write a prediction. Keep the evaluation settings fixed and run the notebook in order. At each inspection, help me understand the actual data, token IDs, embedding vectors, probabilities, loss, gradients, and weight changes. Ask me to explain them in my own words. Help me save the executed notebook and evidence and write an honest README from my outputs. Do not invent results or substitute a pretrained model.

## Evidence Required in the README

- Show the untrained, halfway, and final text samples, linking the full saved files. Explain at least one visible change or lack of change.
- Embed training_curves.svg and include the full loss table from history.json. State that these are fixed training and validation panels, each with at most 20 documents.
- Link tokenization.json and inspection.json. Include one character-to-ID-to-vector example, the vector before/after, the first parameter's value/gradient/update, and one next-token probability comparison. Explain what each means.
- Link your executed notebook, config.json, training.csv, training_summary.json, and temperature_comparison.json. Explain which data and settings stayed fixed, what changed in training, and what changed only at inference.

## Submission Checklist

- After the final run, save your notebook with all outputs. Upload or push that .ipynb file, your README, and selected results to your public GitHub repository. In Colab, use File → Download → Download .ipynb. Do not clear the outputs. Open the notebook on GitHub and confirm that the final inspections, losses, samples, and plot are visible.
- Open your repository signed out and verify that the notebook, plot, sample files, and evidence links are accessible.
- Keep the complete results ZIP locally. The checkpoint stores learned weights for inspection, not the optimizer state needed for exact training resume. Include the corpus or a reproducible source link when sharing is permitted.
- [Submit your GitHub repository](https://submissions-portal-eight.vercel.app)

## Learning Focus

The central purpose is understanding how data becomes predictions and how a neural network learns. Your explanation, choices, and visible evidence should support one another. A bigger network, longer training run, or more convincing sample does not replace an explanation. Report limitations rather than claiming the model understands language from a few plausible strings. Losses across different corpora are not a class ranking.

## Definition of Done

- The notebook completes with your chosen corpus and settings, and your README records the actual training budget.
- You compare the untrained, halfway, and final model using fixed evaluation settings and show all measured losses and saved samples.
- You can point to an actual token ID, embedding vector, next-token probability, gradient, and weight update, and explain how they connect.
- You can explain the role of the corpus, the neural network, held-out data, attention/context, and temperature, plus one limitation.

## Single Deliverable

- One public GitHub repository URL through the course submission portal. Its README contains your explanation and evidence, with your executed notebook available to inspect.

## Scope

- Use the supplied microgpt-style model. Writing the network or autograd engine yourself is optional. The core runs on a CPU in plain Python without an API key or pretrained weights.
- No website, backend, deployment, GPU purchase, or large-model training is required. Keep the first experiment small enough to inspect.
- [Karpathy's microgpt source](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) is the core reference. The larger [GPT video project](https://github.com/karpathy/ng-video-lecture) is an optional route to PyTorch and Shakespeare after you understand the short-document model.
- A proposed next experiment is enough; a second training run is optional.


