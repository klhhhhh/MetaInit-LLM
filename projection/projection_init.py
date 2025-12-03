import torch
import torch.nn as nn

@torch.no_grad()
def svd_lora_init_from_small(
    W_s: torch.Tensor,
    d_b_out: int,
    d_b_in: int,
    r: int = None,
    theta_deg: float = 8.0,   # Small angle, recommended 5°~10°
    device=None,
    dtype=None,
    return_Wb0: bool = True,
):
    """
    Use "small-angle orthogonal mixing (without increasing rank)" to upscale 
    the small model weights W_s to the large model space, constructing 
    LoRA-style A_out, B_out, A_in, B_in.
    
    Args:
        W_s: (d_s_out, d_s_in) Small model weights
        d_b_out: Large model output dimension
        d_b_in : Large model input dimension
        r: Truncated rank (default is min(d_s_out, d_s_in))
        theta_deg: Small angle (in degrees); only affects the first r_mix columns
        device, dtype: Optional; defaults to follow W_s
        return_Wb0: Whether to return the deterministically initialized W_b^(0)
    Returns:
        A_out: (d_b_out, r)
        B_out: (r, d_s_out)
        A_in : (d_b_in , r)
        B_in : (r, d_s_in)
        W_b0 (optional): (d_b_out, d_b_in)
    """
    if device is None: device = W_s.device
    if dtype  is None: dtype  = W_s.dtype

    d_s_out, d_s_in = W_s.shape
    if r is None:
        r = min(d_s_out, d_s_in)

    # 1) Thin SVD (truncated to rank=r)
    # W_s ≈ U_s Σ_s V_s^T
    # torch.linalg.svd returns U, S, Vh, where Vh = V^T
    U, S, Vh = torch.linalg.svd(W_s.to(device=device, dtype=dtype), full_matrices=False)
    U_s = U[:, :r]                             # (d_s_out, r)
    S_r = S[:r]                                # (r,)
    V_s = Vh[:r, :].T                          # (d_s_in, r)

    # 2) Embed into the upper block of the large space: U_e, V_e
    # U_e = [U_s; 0], shape (d_b_out, r)
    # V_e = [V_s; 0], shape (d_b_in , r)
    def pad_upper(U_small, big_rows):
        d_small, rr = U_small.shape
        out = torch.zeros(big_rows, rr, device=device, dtype=dtype)
        out[:d_small, :] = U_small
        return out

    U_e = pad_upper(U_s, d_b_out)  # (d_b_out, r)
    V_e = pad_upper(V_s, d_b_in)   # (d_b_in , r)

    # 3) Orthogonal complement Q_perp for the lower block: use standard basis 
    # (identity matrix columns in the lower part), naturally orthogonal to U_e/V_e
    # The number of mixable columns r_mix is limited by the complement space dimension
    r_mix_out = min(r, max(0, d_b_out - d_s_out))
    r_mix_in  = min(r, max(0, d_b_in  - d_s_in ))

    def make_Q_perp(big_rows, small_rows, rr_mix, rr_total):
        """
        Generate Q_perp_full ∈ R^{big_rows × rr_total}:
        - The first rr_mix columns place identity vectors in the lower block, 
          other columns are zero
        - Naturally orthogonal to the upper block (U_e upper is non-zero, lower is zero)
        """
        Q_full = torch.zeros(big_rows, rr_total, device=device, dtype=dtype)
        if rr_mix > 0:
            # Place I_{rr_mix} in the lower block
            Q_full[small_rows: small_rows + rr_mix, :rr_mix] = torch.eye(rr_mix, device=device, dtype=dtype)
        return Q_full  # (big_rows, rr_total)
    
    def make_Q_perp_DCT(big_rows, small_rows, rr_mix, rr_total):
        """
        Deterministic & smooth orthonormal lower-block basis via DCT (+QR).
        Returns Q_full ∈ R^{big_rows × rr_total} whose first rr_mix columns
        live in the lower block (rows [small_rows:]) and are orthonormal.
        """
        Q_full = torch.zeros(big_rows, rr_total, device=device, dtype=dtype)
        low_rows = big_rows - small_rows
        if rr_mix > 0 and low_rows > 0:
            # DCT-II kernel on the lower block
            n = torch.arange(low_rows, device=device, dtype=dtype).unsqueeze(1)    # (low_rows, 1)
            k = torch.arange(rr_mix,  device=device, dtype=dtype).unsqueeze(0)    # (1, rr_mix)
            C = torch.cos(3.141592653589793 * (n + 0.5) * k / float(low_rows))    # (low_rows, rr_mix)

            # Orthonormalize deterministically to remove numerical correlations
            Q, _ = torch.linalg.qr(C, mode='reduced')                              # (low_rows, rr_mix)

            # Fix column signs for determinism (first nonzero entry positive)
            for j in range(Q.shape[1]):
                col = Q[:, j]
                idx = (col.abs() > 1e-12).nonzero(as_tuple=True)[0]
                if idx.numel() > 0 and col[idx[0]] < 0:
                    Q[:, j] = -col

            Q_full[small_rows:, :rr_mix] = Q
        return Q_full


    Q_perp_out = make_Q_perp(d_b_out, d_s_out, r_mix_out, r)  # (d_b_out, r)
    Q_perp_in  = make_Q_perp(d_b_in , d_s_in , r_mix_in , r)  # (d_b_in , r)

    # 4) Small-angle mixing with two separate angle vectors to keep orthonormality
    #    - theta_out: non-zero only for the first r_mix_out columns
    #    - theta_in : non-zero only for the first r_mix_in  columns
    theta_out = torch.zeros(r, device=device, dtype=dtype)
    theta_in  = torch.zeros(r, device=device, dtype=dtype)
    if r_mix_out > 0 or r_mix_in > 0:
        th = float(theta_deg) * 3.141592653589793 / 180.0
        if r_mix_out > 0:
            theta_out[:r_mix_out] = th
        if r_mix_in > 0:
            theta_in[:r_mix_in] = th

    cos_out, sin_out = torch.cos(theta_out), torch.sin(theta_out)  # (r,)
    cos_in,  sin_in  = torch.cos(theta_in),  torch.sin(theta_in)   # (r,)

    # Column-wise blend: Ue * cos + Qperp * sin
    def blend(Ue, Qperp, cos_t, sin_t):
        return Ue * cos_t.unsqueeze(0) + Qperp * sin_t.unsqueeze(0)

    U_tilde = blend(U_e, Q_perp_out, cos_out, sin_out)  # (d_b_out, r)
    V_tilde = blend(V_e, Q_perp_in , cos_in , sin_in )  # (d_b_in , r)


    # 5) Construct sqrt(Σ) (only scale columns, no need to explicitly construct diagonal matrix)
    sqrtS = torch.sqrt(S_r + 1e-12)    # (r,)

    # A_out = U_tilde * sqrtΣ,     B_out = sqrtΣ * U_s^T
    A_out = U_tilde * sqrtS.unsqueeze(0)           # (d_b_out, r)
    B_out = (sqrtS.unsqueeze(1) * U_s.T)           # (r, d_s_out)

    # A_in  = V_tilde * sqrtΣ,     B_in  = sqrtΣ * V_s^T
    A_in  = V_tilde * sqrtS.unsqueeze(0)           # (d_b_in,  r)
    B_in  = (sqrtS.unsqueeze(1) * V_s.T)           # (r, d_s_in)

    if not return_Wb0:
        return A_out, B_out, A_in, B_in

    # 6) Optional: Deterministic upscaled initialization weights W_b^(0) = U_tilde Σ V_tilde^T
    # Similarly avoid explicit diagonal: U_tilde * S then right-multiply V_tilde^T
    Wb0 = (U_tilde * S_r.unsqueeze(0)) @ V_tilde.T  # (d_b_out, d_b_in)
    return A_out, B_out, A_in, B_in, Wb0


# ===== Example Usage =====
if __name__ == "__main__":
    torch.manual_seed(0)
    # Small model weights (example)
    d_s_out, d_s_in = 8, 6
    W_small = torch.randn(d_s_out, d_s_in)

    # Target large model dimensions
    d_b_out, d_b_in = 12, 10

    # Truncated rank r (usually min(d_s_out, d_s_in) or smaller)
    r = min(d_s_out, d_s_in)

    Aout, Bout, Ain, Bin, Wb0 = svd_lora_init_from_small(
        W_small, d_b_out, d_b_in, r=r, theta_deg=8.0, return_Wb0=True
    )

    print("A_out:", Aout.shape, "B_out:", Bout.shape)
    print("A_in :", Ain.shape,  "B_in :",  Bin.shape)
    print("W_b0 :", Wb0.shape)
