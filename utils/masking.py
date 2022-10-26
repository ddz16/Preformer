import torch


class TriangularCausalMask():
    """
       type: bool
       size: (B, 1, L, L)
       Below the diagonal including the diagonal are all False, above the diagonal are all True
       all matrices in the batch are the same
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LogSparseMask():
    def __init__(self, B, H, L_q, L_k, device="cpu"):
        mask_shape = [B, H, L_q, L_k]
        with torch.no_grad():
            log_p = []
            for i in range(14):
                log_p.append(2 ** i)
            self._mask = torch.ones(mask_shape, dtype=torch.bool).to(device)
            for i in range(L_q):
                self._mask[..., i, i] = False
                for p in log_p:
                    if i - p >= 0:
                        self._mask[..., i, i - p] = False
                for p in log_p:
                    if i + p < L_k:
                        self._mask[..., i, i + p] = False

    @property
    def mask(self):
        return self._mask