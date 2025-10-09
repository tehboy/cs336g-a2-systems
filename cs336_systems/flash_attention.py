"""Implementation of a FlashAttention2 kernel."""

from torch import autograd


class FlashAttentionFunc(autograd.Function):
    @staticmethod
    def forward(ctx):
        # Note: You will need to update the signature of this method
        raise NotImplementedError("TODO: Implement FlashAttentionFunc.forward")

    @staticmethod
    def backward(ctx):
        # Note: You will need to update the signature of this method
        raise NotImplementedError(
            "TODO: Implement FlashAttentionFunc.backward"
        )
