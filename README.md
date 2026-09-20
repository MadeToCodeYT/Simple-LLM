# Simple LLM Made From Scratch

Currently putting this project on hold.

## Status: Stopped here

There were two main goals of this project. One, to understand how an LLM works end-to-end, from tokenization to backpropagation and cross-entropy (done). Build and tested a full pipeline: BPE tokenizer, embeddings, positional encoding, self-attention, feed-forward layers, and a backward pass. The second goal was to generate a simple response to a user's input (haven't been able to reach this yet).

## Known limitation: Training Speed

Currently, everything has been written in pure Python with everything being calculated from scratch (such as matrix multiplication, linear algebra, etc.) With the current parameters I have set, a full training run will take an estimated 6 years on a 3.1GHz CPU without a useable GPU (Intel UHD Graphics - Not supported by PyTorch's GPU backends).