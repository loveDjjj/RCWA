import warnings

import torch


pi = 3.141592652589793


def _as_batch_vector(value, batch_size, dtype, device):
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() == 0:
        return tensor.reshape(1).expand(batch_size)
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)
    if tensor.dim() == 1 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(f"Expected scalar or batch vector of length {batch_size}, got shape {tuple(tensor.shape)}")


def _diag_embed(vector):
    return torch.diag_embed(vector)


def _batch_eye(batch_size, n, dtype, device):
    return torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, n, n)


def _batch_zeros(batch_size, rows, cols, dtype, device):
    return torch.zeros((batch_size, rows, cols), dtype=dtype, device=device)


def _vstack(blocks):
    return torch.cat(blocks, dim=-2)


def _hstack(blocks):
    return torch.cat(blocks, dim=-1)


def _positive_kz(kz):
    return torch.where(torch.imag(kz) < 0, torch.conj(kz), kz)


class rcwa:
    """
    Batched PyTorch RCWA.

    This class mirrors the common torcwa.rcwa API, but treats the leading
    dimension as an explicit batch dimension. It is intended for differentiable
    batched forward calculations such as neural-network-generated structures.
    """

    def __init__(
        self,
        freq,
        order,
        L,
        *,
        dtype=torch.complex64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        stable_eig_grad=True,
        avoid_Pinv_instability=False,
        max_Pinv_instability=0.005,
        linalg_batch_mode="auto",
        linalg_batch_threshold=512,
    ):
        if dtype not in (torch.complex64, torch.complex128):
            warnings.warn("Invalid simulation data type. Set as torch.complex64.", UserWarning)
            dtype = torch.complex64

        self._dtype = dtype
        self._real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self._device = device
        self.stable_eig_grad = bool(stable_eig_grad)
        self.avoid_Pinv_instability = bool(avoid_Pinv_instability)
        self.max_Pinv_instability = max_Pinv_instability if avoid_Pinv_instability else None
        if linalg_batch_mode not in ["auto", "batched", "loop"]:
            warnings.warn("Invalid linalg_batch_mode. Set as auto.", UserWarning)
            linalg_batch_mode = "auto"
        self.linalg_batch_mode = linalg_batch_mode
        self.linalg_batch_threshold = int(linalg_batch_threshold)
        self.Pinv_instability = [] if avoid_Pinv_instability else None
        self.Qinv_instability = [] if avoid_Pinv_instability else None

        freq_tensor = torch.as_tensor(freq, dtype=dtype, device=device)
        if freq_tensor.dim() == 0:
            freq_tensor = freq_tensor.reshape(1)
        if freq_tensor.dim() != 1:
            raise ValueError("freq must be scalar or a 1D batch tensor")
        self.freq = freq_tensor
        self.batch_size = freq_tensor.shape[0]
        self.omega = 2 * pi * self.freq
        self.L = L

        self.order = order
        self.order_x = torch.arange(-order[0], order[0] + 1, dtype=torch.int64, device=device)
        self.order_y = torch.arange(-order[1], order[1] + 1, dtype=torch.int64, device=device)
        self.order_N = len(self.order_x) * len(self.order_y)
        order_x_grid, order_y_grid = torch.meshgrid(self.order_x, self.order_y, indexing="ij")
        self._conv_ox = order_x_grid.to(torch.int64).reshape([-1])
        self._conv_oy = order_y_grid.to(torch.int64).reshape([-1])
        ind = torch.arange(self.order_N, device=device)
        self._conv_indx, self._conv_indy = torch.meshgrid(ind.to(torch.int64), ind.to(torch.int64), indexing="ij")
        self.Gx_norm = 1 / (L[0] * self.freq)
        self.Gy_norm = 1 / (L[1] * self.freq)

        self.eps_in = _as_batch_vector(1.0, self.batch_size, dtype, device)
        self.mu_in = _as_batch_vector(1.0, self.batch_size, dtype, device)
        self.eps_out = _as_batch_vector(1.0, self.batch_size, dtype, device)
        self.mu_out = _as_batch_vector(1.0, self.batch_size, dtype, device)

        self.layer_N = 0
        self.thickness = []
        self.eps_conv = []
        self.mu_conv = []
        self.mu_is_identity = []
        self.P = []
        self.Q = []
        self.kz_norm = []
        self.E_eigvec = []
        self.H_eigvec = []
        self.Cf = []
        self.Cb = []
        self.layer_S11 = []
        self.layer_S21 = []
        self.layer_S12 = []
        self.layer_S22 = []

    def _use_looped_linalg(self, matrix):
        if self.linalg_batch_mode == "batched":
            return False
        if self.linalg_batch_mode == "loop":
            return matrix.dim() == 3 and matrix.shape[0] > 1
        return (
            matrix.dim() == 3
            and matrix.shape[0] > 1
            and matrix.is_cuda
            and matrix.shape[-1] >= self.linalg_batch_threshold
        )

    def _solve(self, A, B):
        if self._use_looped_linalg(A):
            return torch.stack([torch.linalg.solve(A[i], B[i]) for i in range(A.shape[0])], dim=0)
        return torch.linalg.solve(A, B)

    def _right_solve(self, B, A):
        return self._solve(A.transpose(-2, -1), B.transpose(-2, -1)).transpose(-2, -1)

    def _eig(self, A):
        if self._use_looped_linalg(A):
            eigvals = []
            eigvecs = []
            for i in range(A.shape[0]):
                eigval, eigvec = torch.linalg.eig(A[i])
                eigvals.append(eigval)
                eigvecs.append(eigvec)
            return torch.stack(eigvals, dim=0), torch.stack(eigvecs, dim=0)
        return torch.linalg.eig(A)

    def add_input_layer(self, eps=1.0, mu=1.0):
        self.eps_in = _as_batch_vector(eps, self.batch_size, self._dtype, self._device)
        self.mu_in = _as_batch_vector(mu, self.batch_size, self._dtype, self._device)
        self.Sin = []

    def add_output_layer(self, eps=1.0, mu=1.0):
        self.eps_out = _as_batch_vector(eps, self.batch_size, self._dtype, self._device)
        self.mu_out = _as_batch_vector(mu, self.batch_size, self._dtype, self._device)
        self.Sout = []

    def set_incident_angle(self, inc_ang, azi_ang, angle_layer="input"):
        self.inc_ang = _as_batch_vector(inc_ang, self.batch_size, self._dtype, self._device)
        self.azi_ang = _as_batch_vector(azi_ang, self.batch_size, self._dtype, self._device)
        if angle_layer in ["i", "in", "input"]:
            self.angle_layer = "input"
        elif angle_layer in ["o", "out", "output"]:
            self.angle_layer = "output"
        else:
            warnings.warn("Invalid angle layer. Set as input layer.", UserWarning)
            self.angle_layer = "input"
        self._kvectors()

    def add_layer(self, thickness, eps=1.0, mu=1.0):
        eps_t = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        mu_t = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        eps_homogeneous = self._is_homogeneous(eps_t)
        mu_homogeneous = self._is_homogeneous(mu_t)

        self.eps_conv.append(self._homogeneous_conv(eps_t) if eps_homogeneous else self._material_conv(eps_t))
        self.mu_conv.append(self._homogeneous_conv(mu_t) if mu_homogeneous else self._material_conv(mu_t))
        self.mu_is_identity.append(self._is_identity_material(mu_t))

        self.layer_N += 1
        self.thickness.append(torch.as_tensor(thickness, dtype=self._dtype, device=self._device))

        if eps_homogeneous and mu_homogeneous:
            eps_vec = _as_batch_vector(eps_t, self.batch_size, self._dtype, self._device)
            mu_vec = _as_batch_vector(mu_t, self.batch_size, self._dtype, self._device)
            self._eigen_decomposition_homogeneous(eps_vec, mu_vec)
        else:
            self._eigen_decomposition()
        self._solve_layer_smatrix()

    def add_layer_conv(self, thickness, eps_conv, mu=1.0):
        eps_conv_t = torch.as_tensor(eps_conv, dtype=self._dtype, device=self._device)
        if eps_conv_t.dim() != 3 or eps_conv_t.shape != (self.batch_size, self.order_N, self.order_N):
            raise ValueError(
                f"eps_conv must have shape [{self.batch_size}, {self.order_N}, {self.order_N}], "
                f"got {tuple(eps_conv_t.shape)}"
            )
        mu_t = torch.as_tensor(mu, dtype=self._dtype, device=self._device)

        self.eps_conv.append(eps_conv_t)
        self.mu_conv.append(self._homogeneous_conv(mu_t) if self._is_homogeneous(mu_t) else self._material_conv(mu_t))
        self.mu_is_identity.append(self._is_identity_material(mu_t))

        self.layer_N += 1
        self.thickness.append(torch.as_tensor(thickness, dtype=self._dtype, device=self._device))
        self._eigen_decomposition()
        self._solve_layer_smatrix()

    def solve_global_smatrix(self):
        if self.layer_N > 0:
            S11 = self.layer_S11[0]
            S21 = self.layer_S21[0]
            S12 = self.layer_S12[0]
            S22 = self.layer_S22[0]
            C = [[self.Cf[0]], [self.Cb[0]]]
        else:
            n2 = 2 * self.order_N
            S11 = _batch_eye(self.batch_size, n2, self._dtype, self._device)
            S21 = _batch_zeros(self.batch_size, n2, n2, self._dtype, self._device)
            S12 = _batch_zeros(self.batch_size, n2, n2, self._dtype, self._device)
            S22 = _batch_eye(self.batch_size, n2, self._dtype, self._device)
            C = [[], []]

        for i in range(self.layer_N - 1):
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[S11, S21, S12, S22],
                Sn=[self.layer_S11[i + 1], self.layer_S21[i + 1], self.layer_S12[i + 1], self.layer_S22[i + 1]],
                Cm=C,
                Cn=[[self.Cf[i + 1]], [self.Cb[i + 1]]],
            )

        if hasattr(self, "Sin"):
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                Sn=[S11, S21, S12, S22],
                Cm=[[], []],
                Cn=C,
            )

        if hasattr(self, "Sout"):
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[S11, S21, S12, S22],
                Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]],
                Cm=C,
                Cn=[[], []],
            )

        self.S = [S11, S21, S12, S22]
        self.C = C

    def S_parameters(
        self,
        orders,
        *,
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
        power_norm=True,
        evanscent=1e-3,
    ):
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device).reshape([-1, 2])
        ref_order = torch.as_tensor(ref_order, dtype=torch.int64, device=self._device).reshape([1, 2])
        direction = "forward" if direction in ["f", "forward"] else "backward"
        port = "transmission" if port in ["t", "transmission"] else "reflection"

        if polarization not in ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]:
            warnings.warn("Invalid polarization. Set as xx.", UserWarning)
            polarization = "xx"

        order_indices = self._matching_indices(orders)
        ref_order_index = self._matching_indices(ref_order)

        if polarization in ["xx", "yx", "xy", "yy"]:
            return self._S_parameters_xy(
                order_indices, ref_order_index, direction, port, polarization, power_norm, evanscent
            )
        return self._S_parameters_ps(
            order_indices, ref_order_index, direction, port, polarization, power_norm, evanscent
        )

    def _S_parameters_xy(self, order_indices, ref_order_index, direction, port, polarization, power_norm, evanscent):
        if polarization in ["yx", "yy"]:
            order_indices = order_indices + self.order_N
        if polarization in ["xy", "yy"]:
            ref_order_index = ref_order_index + self.order_N

        if direction == "forward" and port == "transmission":
            idx = 0
        elif direction == "forward" and port == "reflection":
            idx = 1
        elif direction == "backward" and port == "reflection":
            idx = 2
        else:
            idx = 3

        S = self.S[idx][:, order_indices, ref_order_index.reshape(()).item()]
        if power_norm:
            normalization = self._xy_power_normalization(
                order_indices, ref_order_index, direction, port, polarization, evanscent
            )
            S = S * normalization
        S = torch.where(torch.isnan(S) | torch.isinf(S), torch.zeros_like(S), S)
        return S.reshape(self.batch_size) if S.shape[-1] == 1 else S

    def _S_parameters_ps(self, order_indices, ref_order_index, direction, port, polarization, power_norm, evanscent):
        if direction == "forward" and port == "transmission":
            idx = 0
            order_sign, ref_sign = 1, 1
            order_k0_norm2 = self.eps_out * self.mu_out
            ref_k0_norm2 = self.eps_in * self.mu_in
        elif direction == "forward" and port == "reflection":
            idx = 1
            order_sign, ref_sign = -1, 1
            order_k0_norm2 = self.eps_in * self.mu_in
            ref_k0_norm2 = self.eps_in * self.mu_in
        elif direction == "backward" and port == "reflection":
            idx = 2
            order_sign, ref_sign = 1, -1
            order_k0_norm2 = self.eps_out * self.mu_out
            ref_k0_norm2 = self.eps_out * self.mu_out
        else:
            idx = 3
            order_sign, ref_sign = -1, -1
            order_k0_norm2 = self.eps_in * self.mu_in
            ref_k0_norm2 = self.eps_out * self.mu_out

        order_kx = self.Kx_norm_dn[:, order_indices]
        order_ky = self.Ky_norm_dn[:, order_indices]
        order_kt = torch.sqrt(order_kx**2 + order_ky**2)
        order_kz_complex = torch.sqrt(order_k0_norm2[:, None] - order_kx**2 - order_ky**2)
        order_kz = order_sign * torch.abs(torch.real(order_kz_complex))
        order_is_evanescent = self._is_evanescent(order_kz_complex, evanscent)
        order_inc = torch.atan2(torch.real(order_kt), order_kz)
        order_azi = torch.atan2(torch.real(order_ky), torch.real(order_kx))

        ref_idx = ref_order_index.reshape(()).item()
        ref_kx = self.Kx_norm_dn[:, ref_idx]
        ref_ky = self.Ky_norm_dn[:, ref_idx]
        ref_kt = torch.sqrt(ref_kx**2 + ref_ky**2)
        ref_kz_complex = torch.sqrt(ref_k0_norm2 - ref_kx**2 - ref_ky**2)
        ref_kz = ref_sign * torch.abs(torch.real(ref_kz_complex))
        ref_is_evanescent = self._is_evanescent(ref_kz_complex, evanscent)
        ref_inc = torch.atan2(torch.real(ref_kt), ref_kz)
        ref_azi = torch.atan2(torch.real(ref_ky), torch.real(ref_kx))

        xx = self.S[idx][:, order_indices, ref_idx]
        xy = self.S[idx][:, order_indices, ref_idx + self.order_N]
        yx = self.S[idx][:, order_indices + self.order_N, ref_idx]
        yy = self.S[idx][:, order_indices + self.order_N, ref_idx + self.order_N]
        xx = torch.where(order_is_evanescent, torch.zeros_like(xx), xx)
        xy = torch.where(order_is_evanescent, torch.zeros_like(xy), xy)
        yx = torch.where(order_is_evanescent, torch.zeros_like(yx), yx)
        yy = torch.where(order_is_evanescent, torch.zeros_like(yy), yy)

        if polarization == "pp":
            S = (
                torch.cos(order_azi) / torch.cos(order_inc) * torch.cos(ref_inc)[:, None] * torch.cos(ref_azi)[:, None] * xx
                + torch.sin(order_azi) / torch.cos(order_inc) * torch.cos(ref_inc)[:, None] * torch.cos(ref_azi)[:, None] * yx
                + torch.cos(order_azi) / torch.cos(order_inc) * torch.cos(ref_inc)[:, None] * torch.sin(ref_azi)[:, None] * xy
                + torch.sin(order_azi) / torch.cos(order_inc) * torch.cos(ref_inc)[:, None] * torch.sin(ref_azi)[:, None] * yy
            )
        elif polarization == "ps":
            S = (
                torch.cos(order_azi) / torch.cos(order_inc) * (-torch.sin(ref_azi)[:, None]) * xx
                + torch.sin(order_azi) / torch.cos(order_inc) * (-torch.sin(ref_azi)[:, None]) * yx
                + torch.cos(order_azi) / torch.cos(order_inc) * torch.cos(ref_azi)[:, None] * xy
                + torch.sin(order_azi) / torch.cos(order_inc) * torch.cos(ref_azi)[:, None] * yy
            )
        elif polarization == "sp":
            S = (
                -torch.sin(order_azi) * torch.cos(ref_inc)[:, None] * torch.cos(ref_azi)[:, None] * xx
                + torch.cos(order_azi) * torch.cos(ref_inc)[:, None] * torch.cos(ref_azi)[:, None] * yx
                - torch.sin(order_azi) * torch.cos(ref_inc)[:, None] * torch.sin(ref_azi)[:, None] * xy
                + torch.cos(order_azi) * torch.cos(ref_inc)[:, None] * torch.sin(ref_azi)[:, None] * yy
            )
        else:
            S = (
                -torch.sin(order_azi) * (-torch.sin(ref_azi)[:, None]) * xx
                + torch.cos(order_azi) * (-torch.sin(ref_azi)[:, None]) * yx
                - torch.sin(order_azi) * torch.cos(ref_azi)[:, None] * xy
                + torch.cos(order_azi) * torch.cos(ref_azi)[:, None] * yy
            )

        S = torch.where(ref_is_evanescent[:, None], torch.zeros_like(S), S)
        if power_norm:
            normalization = self._ps_power_normalization(order_indices, ref_idx, direction, port, evanscent)
            S = S * normalization
        S = torch.where(torch.isnan(S) | torch.isinf(S), torch.zeros_like(S), S)
        return S.reshape(self.batch_size) if S.shape[-1] == 1 else S

    def _is_homogeneous(self, tensor):
        return tensor.dim() == 0 or (tensor.dim() == 1 and tensor.shape[0] in [1, self.batch_size])

    def _is_identity_material(self, tensor):
        if not self._is_homogeneous(tensor):
            return False
        vec = _as_batch_vector(tensor, self.batch_size, self._dtype, self._device)
        return bool(torch.all(vec == torch.ones_like(vec)).detach().cpu().item())

    def _homogeneous_conv(self, value):
        vec = _as_batch_vector(value, self.batch_size, self._dtype, self._device)
        return vec[:, None, None] * _batch_eye(self.batch_size, self.order_N, self._dtype, self._device)

    def _matching_indices(self, orders):
        ox = (orders[:, 0] + self.order[0]).to(torch.int64)
        oy = (orders[:, 1] + self.order[1]).to(torch.int64)
        return ox * len(self.order_y) + oy

    def _kvectors(self):
        if self.angle_layer == "input":
            k0 = torch.real(torch.sqrt(self.eps_in * self.mu_in))
        else:
            k0 = torch.real(torch.sqrt(self.eps_out * self.mu_out))
        self.kx0_norm = k0 * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
        self.ky0_norm = k0 * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        kx = self.kx0_norm[:, None] + self.order_x.to(self._dtype)[None, :] * self.Gx_norm[:, None]
        ky = self.ky0_norm[:, None] + self.order_y.to(self._dtype)[None, :] * self.Gy_norm[:, None]
        kx_grid = kx[:, :, None].expand(self.batch_size, len(self.order_x), len(self.order_y))
        ky_grid = ky[:, None, :].expand(self.batch_size, len(self.order_x), len(self.order_y))
        self.Kx_norm_dn = kx_grid.reshape(self.batch_size, -1)
        self.Ky_norm_dn = ky_grid.reshape(self.batch_size, -1)
        self.Kx_norm = _diag_embed(self.Kx_norm_dn)
        self.Ky_norm = _diag_embed(self.Ky_norm_dn)

        kz = _positive_kz(torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
        self.Vf = self._V_matrix(kz)

        if hasattr(self, "Sin"):
            kz_in = _positive_kz(torch.sqrt(self.eps_in[:, None] * self.mu_in[:, None] - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
            self.Vi = self._V_matrix(kz_in)
            Vtmp1 = self.Vf + self.Vi
            Vtmp2 = self.Vf - self.Vi
            n2 = self.Vi.shape[-1]
            sol = self._solve(Vtmp1, torch.cat([self.Vi, Vtmp2, self.Vf], dim=-1))
            X_vi, X_diff, X_vf = torch.split(sol, n2, dim=-1)
            self.Sin = [
                2 * X_vi,
                -X_diff,
                X_diff,
                2 * X_vf,
            ]

        if hasattr(self, "Sout"):
            kz_out = _positive_kz(torch.sqrt(self.eps_out[:, None] * self.mu_out[:, None] - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
            self.Vo = self._V_matrix(kz_out)
            Vtmp1 = self.Vf + self.Vo
            Vtmp2 = self.Vf - self.Vo
            n2 = self.Vo.shape[-1]
            sol = self._solve(Vtmp1, torch.cat([self.Vf, Vtmp2, self.Vo], dim=-1))
            X_vf, X_diff, X_vo = torch.split(sol, n2, dim=-1)
            self.Sout = [
                2 * X_vf,
                X_diff,
                -X_diff,
                2 * X_vo,
            ]

    def _V_matrix(self, kz):
        tmp1 = _vstack(
            [
                _diag_embed(-self.Ky_norm_dn * self.Kx_norm_dn / kz),
                _diag_embed(kz + self.Kx_norm_dn**2 / kz),
            ]
        )
        tmp2 = _vstack(
            [
                _diag_embed(-kz - self.Ky_norm_dn**2 / kz),
                _diag_embed(self.Kx_norm_dn * self.Ky_norm_dn / kz),
            ]
        )
        return _hstack([tmp1, tmp2])

    def _material_conv(self, material):
        return self.convolution_matrix(material)

    def convolution_matrix(self, material):
        if material.dim() == 2:
            material = material.unsqueeze(0).expand(self.batch_size, -1, -1)
        if material.dim() != 3 or material.shape[0] != self.batch_size:
            raise ValueError("Patterned material must have shape [nx, ny] or [batch, nx, ny]")

        material_N = material.shape[-2] * material.shape[-1]
        material_fft = torch.fft.fft2(material, dim=(-2, -1)) / material_N
        real = torch.real(material_fft)
        imag = torch.imag(material_fft)
        conv_real = real[:, self._conv_ox[self._conv_indx] - self._conv_ox[self._conv_indy], self._conv_oy[self._conv_indx] - self._conv_oy[self._conv_indy]]
        conv_imag = imag[:, self._conv_ox[self._conv_indx] - self._conv_ox[self._conv_indy], self._conv_oy[self._conv_indx] - self._conv_oy[self._conv_indy]]
        return torch.complex(conv_real, conv_imag)

    def _eigen_decomposition_homogeneous(self, eps, mu):
        zero_mu = torch.zeros_like(self.mu_conv[-1])
        zero_eps = torch.zeros_like(self.eps_conv[-1])
        P_base = _hstack([_vstack([zero_mu, -self.mu_conv[-1]]), _vstack([self.mu_conv[-1], zero_mu])])
        Q_base = _hstack([_vstack([zero_eps, self.eps_conv[-1]]), _vstack([-self.eps_conv[-1], zero_eps])])
        K_stack = _vstack([self.Kx_norm, self.Ky_norm])
        self.P.append(P_base + (1 / eps)[:, None, None] * torch.matmul(K_stack, _hstack([self.Ky_norm, -self.Kx_norm])))
        self.Q.append(Q_base + (1 / mu)[:, None, None] * torch.matmul(K_stack, _hstack([-self.Ky_norm, self.Kx_norm])))

        n2 = self.P[-1].shape[-1]
        self.E_eigvec.append(_batch_eye(self.batch_size, n2, self._dtype, self._device))
        kz = _positive_kz(torch.sqrt(eps[:, None] * mu[:, None] - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
        self.kz_norm.append(torch.cat([kz, kz], dim=-1))

    def _eigen_decomposition(self):
        zero_mu = torch.zeros_like(self.mu_conv[-1])
        zero_eps = torch.zeros_like(self.eps_conv[-1])
        K_stack = _vstack([self.Kx_norm, self.Ky_norm])
        P_tmp = self._right_solve(K_stack, self.eps_conv[-1])
        Q_tmp = K_stack if self.mu_is_identity[-1] else self._right_solve(K_stack, self.mu_conv[-1])
        self.P.append(
            _hstack([_vstack([zero_mu, -self.mu_conv[-1]]), _vstack([self.mu_conv[-1], zero_mu])])
            + torch.matmul(P_tmp, _hstack([self.Ky_norm, -self.Kx_norm]))
        )
        self.Q.append(
            _hstack([_vstack([zero_eps, self.eps_conv[-1]]), _vstack([-self.eps_conv[-1], zero_eps])])
            + torch.matmul(Q_tmp, _hstack([-self.Ky_norm, self.Kx_norm]))
        )
        eigval, eigvec = self._eig(torch.matmul(self.P[-1], self.Q[-1]))
        kz = torch.sqrt(eigval)
        self.kz_norm.append(torch.where(torch.imag(kz) < 0, -kz, kz))
        self.E_eigvec.append(eigvec)

    def _solve_layer_smatrix(self):
        kz_vec = self.kz_norm[-1]
        thickness = self.thickness[-1]
        if thickness.dim() == 0:
            thickness_factor = thickness
        else:
            thickness_factor = thickness.reshape(self.batch_size, 1)
        phase_vec = torch.exp(1.0j * self.omega[:, None] * kz_vec * thickness_factor)
        E = self.E_eigvec[-1]
        E_kz = E * kz_vec[:, None, :]
        self.H_eigvec.append(self._solve(self.P[-1], E_kz))
        H = self.H_eigvec[-1]
        Vf_inv_H = self._solve(self.Vf, H)
        EH_plus = E + Vf_inv_H
        EH_minus = E - Vf_inv_H
        EH_minus_phase = EH_minus * phase_vec[:, None, :]
        E_phase = E * phase_vec[:, None, :]
        Ctmp1 = _vstack([EH_plus, EH_minus_phase])
        Ctmp2 = _vstack([EH_minus_phase, EH_plus])
        Ctmp = _hstack([Ctmp1, Ctmp2])

        n2 = 2 * self.order_N
        rhs_f = _vstack([
            2 * _batch_eye(self.batch_size, n2, self._dtype, self._device),
            _batch_zeros(self.batch_size, n2, n2, self._dtype, self._device),
        ])
        rhs_b = _vstack([
            _batch_zeros(self.batch_size, n2, n2, self._dtype, self._device),
            2 * _batch_eye(self.batch_size, n2, self._dtype, self._device),
        ])
        Csol = self._solve(Ctmp, torch.cat([rhs_f, rhs_b], dim=-1))
        self.Cf.append(Csol[:, :, :n2])
        self.Cb.append(Csol[:, :, n2:])

        eye = _batch_eye(self.batch_size, n2, self._dtype, self._device)
        self.layer_S11.append(torch.matmul(E_phase, self.Cf[-1][:, :n2, :]) + torch.matmul(E, self.Cf[-1][:, n2:, :]))
        self.layer_S21.append(
            torch.matmul(E, self.Cf[-1][:, :n2, :])
            + torch.matmul(E_phase, self.Cf[-1][:, n2:, :])
            - eye
        )
        self.layer_S12.append(
            torch.matmul(E_phase, self.Cb[-1][:, :n2, :])
            + torch.matmul(E, self.Cb[-1][:, n2:, :])
            - eye
        )
        self.layer_S22.append(torch.matmul(E, self.Cb[-1][:, :n2, :]) + torch.matmul(E_phase, self.Cb[-1][:, n2:, :]))

    def _RS_prod(self, Sm, Sn, Cm, Cn):
        n2 = 2 * self.order_N
        eye = _batch_eye(self.batch_size, n2, self._dtype, self._device)
        tmp1 = self._solve(eye - torch.matmul(Sm[2], Sn[1]), eye)
        tmp2 = self._solve(eye - torch.matmul(Sn[1], Sm[2]), eye)
        S11 = torch.matmul(Sn[0], torch.matmul(tmp1, Sm[0]))
        S21 = Sm[1] + torch.matmul(Sm[3], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0])))
        S12 = Sn[2] + torch.matmul(Sn[0], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3])))
        S22 = torch.matmul(Sm[3], torch.matmul(tmp2, Sn[3]))

        C = [[], []]
        for m in range(len(Cm[0])):
            C[0].append(Cm[0][m] + torch.matmul(Cm[1][m], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0]))))
            C[1].append(torch.matmul(Cm[1][m], torch.matmul(tmp2, Sn[3])))
        for n in range(len(Cn[0])):
            C[0].append(torch.matmul(Cn[0][n], torch.matmul(tmp1, Sm[0])))
            C[1].append(Cn[1][n] + torch.matmul(Cn[0][n], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3]))))
        return [S11, S21, S12, S22], C

    def _is_evanescent(self, kz_complex, evanscent):
        ratio = torch.real(kz_complex) / torch.imag(kz_complex)
        return torch.abs(ratio) < evanscent

    def _kz_power_vectors(self, evanscent):
        kz_in_complex = torch.sqrt(self.eps_in[:, None] * self.mu_in[:, None] - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_out_complex = torch.sqrt(self.eps_out[:, None] * self.mu_out[:, None] - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        ev_in = self._is_evanescent(kz_in_complex, evanscent)
        ev_out = self._is_evanescent(kz_out_complex, evanscent)
        kz_in = torch.where(ev_in, torch.real(torch.zeros_like(kz_in_complex)), torch.real(kz_in_complex))
        kz_out = torch.where(ev_out, torch.real(torch.zeros_like(kz_out_complex)), torch.real(kz_out_complex))
        return torch.cat([kz_in, kz_in], dim=-1), torch.cat([kz_out, kz_out], dim=-1)

    def _xy_power_normalization(self, order_indices, ref_order_index, direction, port, polarization, evanscent):
        kz_in, kz_out = self._kz_power_vectors(evanscent)
        kx = torch.cat([torch.real(self.Kx_norm_dn), torch.real(self.Kx_norm_dn)], dim=-1)
        ky = torch.cat([torch.real(self.Ky_norm_dn), torch.real(self.Ky_norm_dn)], dim=-1)
        if polarization == "xx":
            numerator_pol, denominator_pol = kx, kx
        elif polarization == "xy":
            numerator_pol, denominator_pol = kx, ky
        elif polarization == "yx":
            numerator_pol, denominator_pol = ky, kx
        else:
            numerator_pol, denominator_pol = ky, ky

        if direction == "forward" and port == "transmission":
            numerator_kz, denominator_kz = kz_out, kz_in
        elif direction == "forward" and port == "reflection":
            numerator_kz, denominator_kz = kz_in, kz_in
        elif direction == "backward" and port == "reflection":
            numerator_kz, denominator_kz = kz_out, kz_out
        else:
            numerator_kz, denominator_kz = kz_in, kz_out

        ref_idx = ref_order_index.reshape(()).item()
        norm = torch.sqrt(
            (1 + (numerator_pol[:, order_indices] / numerator_kz[:, order_indices]) ** 2)
            / (1 + (denominator_pol[:, ref_idx] / denominator_kz[:, ref_idx])[:, None] ** 2)
        )
        norm = norm * torch.sqrt(numerator_kz[:, order_indices] / denominator_kz[:, ref_idx][:, None])
        return norm

    def _ps_power_normalization(self, order_indices, ref_idx, direction, port, evanscent):
        kz_in, kz_out = self._kz_power_vectors(evanscent)
        if direction == "forward" and port == "transmission":
            numerator_kz, denominator_kz = kz_out, kz_in
        elif direction == "forward" and port == "reflection":
            numerator_kz, denominator_kz = kz_in, kz_in
        elif direction == "backward" and port == "reflection":
            numerator_kz, denominator_kz = kz_out, kz_out
        else:
            numerator_kz, denominator_kz = kz_in, kz_out
        return torch.sqrt(numerator_kz[:, order_indices] / denominator_kz[:, ref_idx][:, None])
