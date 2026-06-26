# Task: Implement EmbHub — A Cross-Lingual Embedding Hub Layer

## Overview & Motivation

I'm researching a method to improve cross-lingual transfer in multilingual LLMs. The problem: current approaches like code-switching (training on text mixing multiple languages) improve cross-lingual transfer but have two drawbacks: (1) they require manually created mixed-language data, and (2) they cause the model to generate mixed-language output.

My idea: add a set of learnable "hub embeddings" as a new layer inserted after the embedding layer. Every token's embedding retrieves from these hub embeddings via similarity-weighted combination. We call this mechanism EmbHub (Embedding Hub) — a central embedding hub that all languages route through.

The intuition:

- In code-switching, cross-lingual transfer happens because tokens from different languages directly attend to each other, forcing the model to align their representations.

- In EmbHub, instead of tokens attending to each other across languages, ALL tokens attend to the same hub embeddings. If the English word "dog" and the Vietnamese word "chó" learn to retrieve similar combinations from the hub, they get pulled into a shared subspace — achieving a similar alignment effect to code-switching.

- This is analogous to shared experts in Mixture-of-Experts models: the hub embeddings act as a shared knowledge bank that all languages route through.

- Unlike code-switching, the coupling strength is continuously tunable via a scalar alpha. Setting alpha to 0 at inference removes the hub entirely, so the model never produces mixed-language output.

This idea is related to soft prompt tuning and prefix tuning, but differs in that: (1) the purpose is cross-lingual alignment during training, not task-specific adaptation, and (2) the contribution is similarity-based — each token retrieves different combinations from the hub based on its semantics — not a fixed prefix prepended to all inputs.

## Architecture Details

### EmbHub Layer

The EmbHub layer is a standalone module inserted after the original embedding layer. It receives token embeddings, computes similarity-based retrieval from the hub embeddings, and outputs modified embeddings of the same shape. The rest of the model stays completely unchanged.

Computation:

1. Compute similarity weights between each token embedding and all hub embeddings: `weights = softmax(token_emb @ hub_embeddings.T / sqrt(d))` where d is the embedding dimension.

2. Compute weighted combination of hub embeddings: `hub_contribution = weights @ hub_embeddings`

3. Add to original embedding with a fixed scaling factor: `output = token_emb + alpha * hub_contribution`

Parameters:
- `hub_embeddings`: learnable, shape `(num_embeddings, embedding_dim)`
- `alpha`: fixed hyperparameter (not learnable), configurable, default 0.05
- `num_embeddings`: configurable, default 1000

### Placement

The EmbHub layer must be applied to pure token embeddings BEFORE any positional encoding. For modern RoPE-based models (LLaMA, Qwen, Mistral), this is not an issue since RoPE is applied inside attention layers, not at the embedding layer. For older models with absolute positional embeddings, the EmbHub layer must be inserted before positional embeddings are added.

### Initialization

Hub embeddings should be initialized using the same initialization method that the base model uses for its original embedding layer.

### Model-Agnostic Design

- The EmbHub layer is a standalone nn.Module that can be inserted into ANY HuggingFace causal LM model (LLaMA, Qwen, Mistral, etc.), not tied to a specific architecture.
- Provide a wrapper/utility that takes any HuggingFace AutoModelForCausalLM and inserts the EmbHub layer after its embedding layer.
- Support two modes:
  - Load from pretrained: loads pretrained weights normally, inserts EmbHub layer with freshly initialized hub embeddings.
  - From scratch: initializes everything from config, inserts EmbHub layer.
- Support a flag to freeze all original model parameters (train only hub embedding params).
- Include a method to disable EmbHub at inference (set alpha=0) to verify the model works as a normal LM without it.

### Testing

Include a simple test that:
1. Loads a small pretrained HF model, inserts EmbHub layer.
2. Runs a forward pass with dummy input.
3. Verifies output shape is unchanged.
4. Verifies disabling EmbHub produces same output as original model.