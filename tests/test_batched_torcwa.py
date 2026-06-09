import torch
import torcwa


def _single_torcwa_txx(freq, eps_layer, thickness, order, period, dtype, device):
    sim = torcwa.rcwa(freq=freq, order=[order, order], L=[period, period], dtype=dtype, device=device)
    sim.add_input_layer(eps=torch.tensor(1.0 + 0.0j, dtype=dtype, device=device))
    sim.add_output_layer(eps=torch.tensor(2.25 + 0.0j, dtype=dtype, device=device))
    sim.set_incident_angle(
        inc_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
        azi_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
    )
    sim.add_layer(thickness=thickness, eps=eps_layer)
    sim.solve_global_smatrix()
    return sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    ).reshape(())


def test_batched_rcwa_matches_torcwa_for_homogeneous_layer():
    import torch_rcwa

    device = torch.device("cpu")
    dtype = torch.complex128
    freqs = torch.tensor([1 / 3.0, 1 / 4.0], dtype=torch.float64, device=device)
    layer_eps = torch.tensor([4.0 + 0.0j, 4.5 + 0.0j], dtype=dtype, device=device)

    sim = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=torch.tensor([2.25 + 0.0j, 2.25 + 0.0j], dtype=dtype, device=device))
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=0.3, eps=layer_eps)
    sim.solve_global_smatrix()

    batched = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    expected = torch.stack(
        [
            _single_torcwa_txx(freqs[i], layer_eps[i], 0.3, 1, 2.8, dtype, device)
            for i in range(len(freqs))
        ]
    )

    torch.testing.assert_close(batched, expected, rtol=2e-5, atol=2e-6)


def test_batched_rcwa_backpropagates_to_patterned_layer():
    import torch_rcwa

    device = torch.device("cpu")
    dtype = torch.complex128
    freqs = torch.tensor([1 / 3.0, 1 / 3.5], dtype=torch.float64, device=device)
    eps = torch.full((2, 8, 8), 1.0 + 0.0j, dtype=dtype, device=device)
    eps[:, 2:6, 2:6] = 4.0 + 0.0j
    eps.requires_grad_(True)

    sim = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=0.2, eps=eps)
    sim.solve_global_smatrix()

    txx = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    loss = torch.sum(torch.abs(txx) ** 2)
    loss.backward()

    assert eps.grad is not None
    assert torch.isfinite(eps.grad).all()
    assert torch.sum(torch.abs(eps.grad)) > 0


def test_add_layer_conv_matches_patterned_layer():
    import torch_rcwa

    device = torch.device("cpu")
    dtype = torch.complex128
    freqs = torch.tensor([1 / 3.0, 1 / 3.5], dtype=torch.float64, device=device)
    eps = torch.full((2, 8, 8), 1.0 + 0.0j, dtype=dtype, device=device)
    eps[:, 2:6, 2:6] = 4.0 + 0.0j

    direct = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    direct.add_input_layer(eps=1.0)
    direct.add_output_layer(eps=2.25)
    direct.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    direct.add_layer(thickness=0.2, eps=eps)
    direct.solve_global_smatrix()

    conv = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    conv.add_input_layer(eps=1.0)
    conv.add_output_layer(eps=2.25)
    conv.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    conv.add_layer_conv(thickness=0.2, eps_conv=conv.convolution_matrix(eps))
    conv.solve_global_smatrix()

    direct_t = direct.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    conv_t = conv.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )

    torch.testing.assert_close(conv_t, direct_t, rtol=2e-5, atol=2e-6)


def test_batched_rcwa_accepts_per_sample_thickness():
    import torch_rcwa

    device = torch.device("cpu")
    dtype = torch.complex128
    freqs = torch.tensor([1 / 3.0, 1 / 3.5], dtype=torch.float64, device=device)
    layer_eps = torch.tensor([4.0 + 0.0j, 4.2 + 0.0j], dtype=dtype, device=device)
    thickness = torch.tensor([0.2, 0.35], dtype=dtype, device=device)

    sim = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=thickness, eps=layer_eps)
    sim.solve_global_smatrix()

    batched = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    expected = torch.stack(
        [
            _single_torcwa_txx(freqs[i], layer_eps[i], thickness[i], 1, 2.8, dtype, device)
            for i in range(len(freqs))
        ]
    )

    torch.testing.assert_close(batched, expected, rtol=2e-5, atol=2e-6)


def test_batched_rcwa_matches_torcwa_for_ps_polarization_and_multiple_orders():
    import torch_rcwa

    device = torch.device("cpu")
    dtype = torch.complex128
    freqs = torch.tensor([1 / 3.0, 1 / 4.0], dtype=torch.float64, device=device)
    layer_eps = torch.tensor([4.0 + 0.0j, 4.2 + 0.0j], dtype=dtype, device=device)
    orders = [[0, 0], [1, 0], [-1, 0]]

    sim = torch_rcwa.rcwa(freq=freqs, order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=0.3, eps=layer_eps)
    sim.solve_global_smatrix()
    batched = sim.S_parameters(
        orders=orders,
        direction="forward",
        port="transmission",
        polarization="ss",
        ref_order=[0, 0],
    )

    expected = []
    for i in range(len(freqs)):
        single = torcwa.rcwa(freq=freqs[i], order=[1, 1], L=[2.8, 2.8], dtype=dtype, device=device)
        single.add_input_layer(eps=torch.tensor(1.0 + 0.0j, dtype=dtype, device=device))
        single.add_output_layer(eps=torch.tensor(2.25 + 0.0j, dtype=dtype, device=device))
        single.set_incident_angle(
            inc_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
            azi_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
        )
        single.add_layer(thickness=0.3, eps=layer_eps[i])
        single.solve_global_smatrix()
        expected.append(
            single.S_parameters(
                orders=orders,
                direction="forward",
                port="transmission",
                polarization="ss",
                ref_order=[0, 0],
            )
        )
    expected = torch.stack(expected)

    torch.testing.assert_close(batched, expected, rtol=2e-5, atol=2e-6)
