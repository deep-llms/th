import math

import torch
import torch.nn as nn


class EmbHub(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        alpha: float = 0.05,
        reference_embedding: nn.Embedding = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.alpha = alpha
        self.hub_embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._scale = 1.0 / math.sqrt(embedding_dim)
        self._init_weights(reference_embedding)

    def _init_weights(self, reference_embedding: nn.Embedding = None) -> None:
        if reference_embedding is not None:
            std = reference_embedding.weight.std().item()
            mean = reference_embedding.weight.mean().item()
            self.hub_embeddings.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.hub_embeddings.data)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return token_embeddings
        weights = torch.softmax(token_embeddings @ self.hub_embeddings.t() * self._scale, dim=-1)
        hub_contribution = weights @ self.hub_embeddings
        return token_embeddings + self.alpha * hub_contribution


class EmbeddingWithHub(nn.Module):
    """Wraps an existing nn.Embedding and applies EmbHub after lookup."""

    def __init__(
        self,
        original_embedding: nn.Embedding,
        num_hub_embeddings: int = 1000,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.original_embedding = original_embedding
        self.hub = EmbHub(
            embedding_dim=original_embedding.embedding_dim,
            num_embeddings=num_hub_embeddings,
            alpha=alpha,
            reference_embedding=original_embedding,
        )

    @property
    def weight(self):
        return self.original_embedding.weight

    @property
    def num_embeddings(self):
        return self.original_embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.original_embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.original_embedding(input_ids)
        return self.hub(token_embeddings)
