import torch

def _qr_orthonormal(shape, device, dtype):
    """Return a random orthonormal matrix with given shape via QR."""
    a = torch.empty(shape, device=device, dtype=dtype).normal_(0, 1)
    q, r = torch.linalg.qr(a, mode='reduced')  # q: [m, n] with orthonormal columns
    # Make Q deterministic sign-wise
    d = torch.sign(torch.diag(r))
    q = q @ torch.diag(d)
    return q

def _block_embed_matrix(out_dim, in_dim, device, dtype, noise=1e-3):
    """
    Build E in R^{out_dim x in_dim}: top in_dim rows ~ orthonormal, bottom rows ~ small orth noise.
    """
    # Orthonormal columns for the first in_dim rows
    q = _qr_orthonormal((in_dim, in_dim), device, dtype)  # square orthonormal
    if out_dim == in_dim:
        E = q
    else:
        extra = torch.empty(out_dim - in_dim, in_dim, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(extra)
        extra.mul_(noise)
        E = torch.cat([q, extra], dim=0)
    return E

def _svd_align_AB_from_small(W_small: torch.Tensor, out_dim: int, rank: int, device, dtype):
    """
    Thin SVD on W_small: W_small ≈ U_r Σ_r V_r^T.
    Build A, B so that A ≈ [U_r; 0]*Σ^{1/2}, B ≈ Σ^{1/2}*V_r^T (truncated/padded to desired sizes).
    """
    with torch.no_grad():
        # W_small: [d_out_s, d_in_s]
        d_out_s, d_in_s = W_small.shape
        r = min(rank, d_out_s, d_in_s)
        # torch.svd is deprecated; use torch.linalg.svd
        U, S, Vh = torch.linalg.svd(W_small.to(device=device, dtype=dtype), full_matrices=False)
        U_r = U[:, :r]         # [d_out_s, r]
        V_r = Vh[:r, :].T      # [d_in_s, r]
        S_r = S[:r]            # [r]

        # A: [out_dim, r], B: [r, d_in_s]
        # Put sqrt(S) half to both sides for balanced scaling
        S_sqrt = S_r.clamp_min(1e-12).sqrt()         # [r]
        A = torch.zeros(out_dim, r, device=device, dtype=dtype)
        B = torch.zeros(r, d_in_s, device=device, dtype=dtype)

        # Embed U_r to top of A (rest small noise); multiply columns by sqrt(S)
        A_top = U_r * S_sqrt.unsqueeze(0)            # [d_out_s, r]
        if out_dim >= d_out_s:
            A[:d_out_s, :] = A_top
            if out_dim > d_out_s:
                noise = torch.empty(out_dim - d_out_s, r, device=device, dtype=dtype)
                torch.nn.init.orthogonal_(noise)
                noise.mul_(1e-3)
                A[d_out_s:, :] = noise
        else:
            A[:, :] = A_top[:out_dim, :]

        # B = (sqrt(S) * V_r^T)
        B[:, :d_in_s] = (S_sqrt.unsqueeze(1) * V_r.T)

        return nn.Parameter(A), nn.Parameter(B)

def init_AB_pair(out_dim, in_dim_small, rank, device, dtype,
                  method="orthogonal_embed",
                  the_same_with_weight=False,
                  init_method_function=None,
                  W_small_block: torch.Tensor=None):
    """
    New methods:
      - 'orthogonal_embed' (default): E_out = [Q; small_orth_noise], E_in = [Q; small_orth_noise]
      - 'svd_align': use thin-SVD of W_small_block for principal-direction alignment (still static)
      - 'orthogonal' and 'normal' keep your original behavior for backward-compat
    """
    A = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
    B = nn.Parameter(torch.empty(rank, in_dim_small, device=device, dtype=dtype))

    if method == "orthogonal_embed":
        # A: [out_dim, rank] ~ block-embedded orthonormal; B: [rank, in_dim_small] ~ orthonormal rows
        # 先做方形正交，再“嵌入+微扰”
        A.data.copy_(_block_embed_matrix(out_dim, rank, device, dtype, noise=1e-3))
        # 对 B：让每一行互相正交（即 B^T 列正交），用 QR 生成 rank×rank，再右侧拼零/小噪声
        B_square = _qr_orthonormal((in_dim_small, in_dim_small), device, dtype)  # [in_dim_small, in_dim_small]
        # 取其前 rank 行的转置等价于取前 rank 列，再转置
        if rank <= in_dim_small:
            B.data.copy_(B_square[:, :rank].T)
        else:
            # rank 大于 in_dim_small 的情况很少见，做零填充并加微扰
            tmp = B_square.T  # [in_dim_small, in_dim_small]
            pad = torch.zeros(rank - in_dim_small, in_dim_small, device=device, dtype=dtype)
            B.data.copy_(torch.cat([tmp, pad], dim=0))
            B.data.add_(1e-3 * torch.empty_like(B).normal_(0, 1))

    elif method == "svd_align":
        assert W_small_block is not None, "svd_align requires W_small_block (the small weight of this layer)"
        A_, B_ = _svd_align_AB_from_small(W_small_block, out_dim, rank, device, dtype)
        A = A_; B = B_

    elif method == "orthogonal":
        torch.nn.init.orthogonal_(A)
        torch.nn.init.orthogonal_(B)

    elif method == "normal":
        if the_same_with_weight:
            assert init_method_function is not None, "init_method_function must be provided when the_same_with_weight is True"
            init_method_function(A); init_method_function(B)
        else:
            torch.nn.init.normal_(A, mean=0.0, std=(1.0 / max(1, in_dim_small) ** 0.5))
            torch.nn.init.normal_(B, mean=0.0, std=(1.0 / max(1, rank) ** 0.5))
    else:
        raise ValueError(f"Unknown init method: {method}")

    return A, B
