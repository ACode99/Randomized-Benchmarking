"""
=============================================================================
Randomized Benchmarking (RB) Simulation — Single Qubit  +  Error Injection
=============================================================================
Framework : Helsen et al. (2020), Magesan et al. (2012)

THEORY
------
Noise model : Pauli depolarizing channel
    Λ(ρ) = (1-ε)ρ + (ε/3)(XρX + YρY + ZρZ)
          = (1 - 4ε/3)ρ + (4ε/3)·I/2

Clifford 2-design twirl → decay parameter  p = 1 − 4ε/3

RB model  : P(m) = A·pᵐ + B

EPC (Magesan 2012):
    EPC = (d²−1)/d² · (1−p) = (3/4)·(1−p)  for d=2
    With p = 1−4ε/3  →  EPC = ε  ✓

QUTIP USAGE
-----------
  • All density matrices are qutip.Qobj (oper type)
  • Noise channel implemented via Kraus operators (qutip.kraus_to_super)
  • Unitary gates applied as U * rho * U.dag()
  • Bloch sphere rendered with qutip.Bloch injected into matplotlib axes
  • Expectation values via qutip.expect
  • Purity computed as Tr(ρ²) via qutip

SUPPORTED ERROR MODELS
----------------------
  1. none           – baseline Pauli depolarizing
  2. drift          – error rate grows linearly during the sequence
  3. gate-dependent – each gate draws a random ε ± variation
  4. coherent       – systematic Z-phase over/under-rotation
  5. heating        – quadratic thermal accumulation
  6. leakage        – population loss to higher levels
  7. combined       – drift + gate-dependent + heating together

USAGE
-----
    python3 rb_simulation_with_errors.py                  # baseline
    python3 rb_simulation_with_errors.py --error drift
    python3 rb_simulation_with_errors.py --error combined --magnitude 1.5
    python3 rb_simulation_with_errors.py --compare        # all 7 models
=============================================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (3D projection)
from scipy.optimize import curve_fit
import qutip as qt
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLIFFORD GROUP  (QuTiP Qobj gates)
# ═════════════════════════════════════════════════════════════════════════════

def generate_clifford_group():
    """
    Generate all 24 single-qubit Clifford group elements via BFS from
    generators H and S.

    Returns
    -------
    clifford_qobj : list of 24 qutip.Qobj unitary operators
    clifford_np   : list of 24 numpy 2×2 complex arrays (for fast arithmetic)
    """
    X = np.array([[0, 1],  [1, 0]],  dtype=complex)
    Z = np.array([[1, 0],  [0, -1]], dtype=complex)
    H_np = (X + Z) / np.sqrt(2)
    S_np = np.array([[1, 0], [0, 1j]], dtype=complex)

    def _norm(m):
        for v in m.flatten():
            if abs(v) > 1e-10:
                return m * np.conj(v) / abs(v)
        return m

    def _key(m):
        return tuple(np.round(_norm(m).flatten(), 8))

    group = {_key(np.eye(2, dtype=complex)): np.eye(2, dtype=complex)}
    queue = [np.eye(2, dtype=complex)]
    while queue:
        cur = queue.pop(0)
        for g in [H_np, S_np]:
            for prod in [g @ cur, cur @ g]:
                k = _key(prod)
                if k not in group:
                    group[k] = prod
                    queue.append(prod)

    clifford_np   = list(group.values())
    assert len(clifford_np) == 24
    # Wrap every element as a QuTiP Qobj
    clifford_qobj = [qt.Qobj(m) for m in clifford_np]
    return clifford_qobj, clifford_np


def build_inverse_lookup(clifford_np):
    """Dictionary  normalized_key → numpy matrix  for O(1) inverse lookup."""
    def _norm(m):
        for v in m.flatten():
            if abs(v) > 1e-10:
                return m * np.conj(v) / abs(v)
        return m
    def _key(m):
        return tuple(np.round(_norm(m).flatten(), 8))
    return {_key(c): c for c in clifford_np}


def get_clifford_inverse(U_comp_np, lookup):
    """Return the Clifford element equal to U_comp†, as a numpy array."""
    U_dag = U_comp_np.conj().T
    for v in U_dag.flatten():
        if abs(v) > 1e-10:
            k = tuple(np.round((U_dag * np.conj(v) / abs(v)).flatten(), 8))
            return lookup.get(k, U_dag)
    return U_dag


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NOISE CHANNEL  (QuTiP Kraus operators)
# ═════════════════════════════════════════════════════════════════════════════

def build_kraus_ops(eps):
    """
    Build Kraus operators for the Pauli depolarizing channel with rate ε:

        Λ(ρ) = (1-ε)ρ + (ε/3)(XρX + YρY + ZρZ)

    Completeness:  Σ Kᵢ†Kᵢ = I  (verified automatically by QuTiP)

    Returns list of 4 qutip.Qobj Kraus operators.
    """
    K = [
        np.sqrt(1 - eps)   * qt.qeye(2),
        np.sqrt(eps / 3)   * qt.sigmax(),
        np.sqrt(eps / 3)   * qt.sigmay(),
        np.sqrt(eps / 3)   * qt.sigmaz(),
    ]
    return K


def apply_channel_qutip(rho_qt, kraus_ops):
    """
    Apply a Kraus channel to a QuTiP density matrix:
        Λ(ρ) = Σᵢ Kᵢ ρ Kᵢ†

    Parameters
    ----------
    rho_qt    : qutip.Qobj density matrix
    kraus_ops : list of qutip.Qobj Kraus operators

    Returns
    -------
    qutip.Qobj output density matrix
    """
    return sum(K * rho_qt * K.dag() for K in kraus_ops)


def apply_noisy_gate_qutip(rho_qt, U_np, kraus_ops):
    """
    Apply ideal unitary U, then the Kraus depolarizing channel:
        Λ_noisy(ρ) = Λ_dep(U ρ U†)

    Parameters
    ----------
    rho_qt    : qutip.Qobj input density matrix
    U_np      : numpy 2×2 unitary (Clifford element)
    kraus_ops : list of qutip.Qobj Kraus operators

    Returns
    -------
    qutip.Qobj output density matrix
    """
    U_qt      = qt.Qobj(U_np)
    rho_ideal = U_qt * rho_qt * U_qt.dag()
    return apply_channel_qutip(rho_ideal, kraus_ops)


# ── Error-model gate functions  (all operate on QuTiP Qobj) ──────────────────

def _gate_none(rho_qt, U_np, eps, gate_num=0, total=1, rng=None, magnitude=1.0):
    """Baseline: gate-independent Pauli depolarizing channel."""
    return apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps))


def _gate_drift(rho_qt, U_np, eps, gate_num=0, total=1, rng=None, magnitude=1.0):
    """
    TIME-DEPENDENT DRIFT: ε grows linearly through the sequence.
    Models temperature / flux-bias instability.
    """
    eps_t = eps * (1 + 0.40 * magnitude * gate_num / max(total, 1))
    return apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps_t))


def _gate_gate_dependent(rho_qt, U_np, eps, gate_num=0, total=1, rng=None,
                         magnitude=1.0):
    """
    GATE-DEPENDENT NOISE: each gate draws ε from Uniform[ε(1-v/2), ε(1+v/2)].
    Models per-gate pulse calibration imperfections.
    """
    v        = 0.50 * magnitude
    eps_gate = max(eps * (1 - v/2 + v * rng.random()), 1e-6)
    return apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps_gate))


def _gate_coherent(rho_qt, U_np, eps, gate_num=0, total=1, rng=None,
                   magnitude=1.0):
    """
    COHERENT ERRORS: systematic Z-rotation error on every gate.
    U_actual = U · exp(−iφZ/2)
    Produces oscillatory residuals on top of exponential RB decay.
    """
    phi      = 0.10 * magnitude
    err_z    = np.array([[np.exp(-1j*phi/2), 0],
                          [0, np.exp( 1j*phi/2)]], dtype=complex)
    U_actual = U_np @ err_z
    return apply_noisy_gate_qutip(rho_qt, U_actual, build_kraus_ops(eps))


def _gate_heating(rho_qt, U_np, eps, gate_num=0, total=1, rng=None,
                  magnitude=1.0):
    """
    THERMAL HEATING: ε grows quadratically with sequence depth.
    Models heat accumulation from repeated microwave pulses.
    """
    eps_heat = eps * (1 + 2.0 * magnitude * (gate_num / max(total, 1))**2)
    return apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps_heat))


def _gate_leakage(rho_qt, U_np, eps, gate_num=0, total=1, rng=None,
                  magnitude=1.0):
    """
    LEAKAGE: fraction of population escapes to |2⟩, |3⟩, … per gate.
    Modelled by uniform attenuation followed by re-normalisation.
    """
    leakage  = 0.008 * magnitude
    rho_dep  = apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps))
    rho_out  = (1 - leakage) * rho_dep
    tr       = rho_out.tr().real
    return rho_out / tr if tr > 1e-12 else rho_out


def _gate_combined(rho_qt, U_np, eps, gate_num=0, total=1, rng=None,
                   magnitude=1.0):
    """
    COMBINED / REALISTIC: drift + gate-dependent variation + heating.
        ε_eff = ε · (1+drift) · (1+gate_var) · (1+heat)
    """
    drift   = 0.20 * magnitude * gate_num / max(total, 1)
    v       = 0.30 * magnitude
    gate_v  = -v/2 + v * rng.random()
    heat    = 0.50 * magnitude * (gate_num / max(total, 1))**2
    eps_eff = max(eps * (1+drift) * (1+gate_v) * (1+heat), 1e-6)
    return apply_noisy_gate_qutip(rho_qt, U_np, build_kraus_ops(eps_eff))


_ERROR_FN = {
    'none':           _gate_none,
    'drift':          _gate_drift,
    'gate-dependent': _gate_gate_dependent,
    'coherent':       _gate_coherent,
    'heating':        _gate_heating,
    'leakage':        _gate_leakage,
    'combined':       _gate_combined,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RB SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_rb_simulation(clifford_qobj, clifford_np, sequence_lengths,
                      n_sequences, eps, error_type='none',
                      error_magnitude=1.0, rng=None):
    """
    4-step RB protocol using QuTiP density matrices throughout.

    Steps per realisation (m, r):
      1. Prepare ρ₀ = |0⟩⟨0|  as a qutip.Qobj
      2. Sample m random Cliffords g₁,…,gₘ from C₁
      3. Compute exact group inverse  g_inv
      4. Apply (m+1) noisy gates via QuTiP Kraus channel
      5. Measure  P = Tr(|0⟩⟨0| · ρ_out) = qt.expect(proj0, rho_out)

    Returns
    -------
    survival_mean  : 1-D ndarray ⟨P(m)⟩
    survival_std   : 1-D ndarray SEM
    all_data       : dict m → 1-D ndarray of individual P values
    rho_trajectory : list of qutip.Qobj — state after each m for m in
                     sequence_lengths (single representative sequence)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gate_fn    = _ERROR_FN.get(error_type, _gate_none)
    n_clifford = len(clifford_np)
    lookup     = build_inverse_lookup(clifford_np)

    # QuTiP objects
    rho0  = qt.ket2dm(qt.basis(2, 0))   # |0⟩⟨0|  as Qobj
    proj0 = qt.ket2dm(qt.basis(2, 0))   # |0⟩⟨0|  projector

    survival_mean, survival_std, all_data = [], [], {}
    rho_trajectory = []                  # one representative state per m

    print(f"\n{'='*66}")
    print(f"  Error model : {error_type:<16}  |  magnitude = {error_magnitude}×")
    print(f"  ε = {eps:.4f}  |  n_seq = {n_sequences}  |  "
          f"theory p = {1-4*eps/3:.6f}")
    print(f"{'='*66}")
    print(f"  {'m':>5}  {'P(m)':>10}  {'SEM':>10}  "
          f"{'theory':>10}  {'purity':>8}")
    print(f"  {'-'*52}")

    for m in sequence_lengths:
        probs = np.empty(n_sequences)

        for r in range(n_sequences):
            rho     = rho0.copy()
            indices = rng.integers(0, n_clifford, size=m)

            # Composite unitary (numpy) for exact inverse
            U_comp = np.eye(2, dtype=complex)
            for idx in indices:
                U_comp = clifford_np[idx] @ U_comp
            g_inv = get_clifford_inverse(U_comp, lookup)

            # Forward gates (QuTiP Kraus channel)
            for gate_num, idx in enumerate(indices):
                rho = gate_fn(rho, clifford_np[idx], eps,
                              gate_num=gate_num, total=m,
                              rng=rng, magnitude=error_magnitude)

            # Inverse gate
            rho = gate_fn(rho, g_inv, eps,
                          gate_num=m, total=m,
                          rng=rng, magnitude=error_magnitude)

            # Survival probability via QuTiP expectation value
            probs[r] = qt.expect(proj0, rho).real

        # Save representative final state (last realisation) for Bloch sphere
        rho_trajectory.append(rho)

        all_data[m] = probs
        pm     = probs.mean()
        sem    = probs.std(ddof=1) / np.sqrt(n_sequences)
        th     = 0.5 + 0.5 * (1 - 4*eps/3)**(m + 1)
        purity = (rho * rho).tr().real   # Tr(ρ²) via QuTiP
        survival_mean.append(pm)
        survival_std.append(sem)
        print(f"  {m:>5}  {pm:>10.6f}  {sem:>10.6f}  "
              f"{th:>10.6f}  {purity:>8.4f}")

    print(f"{'='*66}\n")
    return (np.array(survival_mean), np.array(survival_std),
            all_data, rho_trajectory)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CURVE FITTING
# ═════════════════════════════════════════════════════════════════════════════

def rb_model(m, A, p, B):
    """RB decay model: P(m) = A·pᵐ + B"""
    return A * np.power(p, m.astype(float)) + B


def fit_rb_curve(sequence_lengths, survival_mean, survival_std, d=2):
    """
    Fit P(m) = A·pᵐ + B.
    EPC = (d²−1)/d² · (1−p)   [Magesan 2012 / Nielsen 2002]
    """
    m_arr  = np.array(sequence_lengths, dtype=float)
    sigma  = np.array(survival_std) + 1e-12
    popt, pcov = curve_fit(rb_model, m_arr, survival_mean,
                           p0=[0.5, 0.93, 0.50],
                           sigma=sigma, absolute_sigma=True,
                           bounds=([0, 0, 0], [1, 1, 1]),
                           maxfev=50000)
    perr    = np.sqrt(np.diag(pcov))
    epc     = (d**2 - 1) / d**2 * (1 - popt[1])
    epc_err = (d**2 - 1) / d**2 * perr[1]
    return popt, pcov, epc, epc_err


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BLOCH SPHERE UTILITIES  (QuTiP)
# ═════════════════════════════════════════════════════════════════════════════

def bloch_vector(rho_qt):
    """
    Return the Bloch vector (x, y, z) of a single-qubit density matrix
    using QuTiP expectation values:
        rᵢ = Tr(σᵢ ρ)
    """
    return [qt.expect(qt.sigmax(), rho_qt).real,
            qt.expect(qt.sigmay(), rho_qt).real,
            qt.expect(qt.sigmaz(), rho_qt).real]


def purity(rho_qt):
    """Purity = Tr(ρ²) computed via QuTiP."""
    return (rho_qt * rho_qt).tr().real


def build_bloch_trajectory(eps, m_max=100):
    """
    Compute the Bloch vector trajectory as ρ passes through successive
    Pauli depolarizing channels (identity gate, gate-independent noise).

    For the ideal starting state |0⟩ the Bloch vector is (0, 0, 1).
    After k applications of Λ_dep(ε) it becomes (0, 0, p^k)
    where p = 1 − 4ε/3.

    Returns list of (x, y, z) tuples for k = 0, 1, …, m_max+1.
    """
    kraus = build_kraus_ops(eps)
    rho   = qt.ket2dm(qt.basis(2, 0))
    traj  = [bloch_vector(rho)]
    for _ in range(m_max + 1):
        rho = apply_channel_qutip(rho, kraus)
        traj.append(bloch_vector(rho))
    return traj


def add_bloch_to_ax(ax3d, rho_qt, title, point_color='#c0392b',
                    vector_color='#1a6b3c', show_trajectory=None):
    """Pure-matplotlib Bloch sphere — avoids qt.Bloch spawning extra figures."""
    bv  = bloch_vector(rho_qt)
    pur = purity(rho_qt)

    # Transparent sphere surface
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(sx, sy, sz, color='#aaddff', alpha=0.07,
                      linewidth=0, antialiased=False)

    # Wireframe circles: equator + two meridians
    th   = np.linspace(0, 2 * np.pi, 120)
    wire = dict(lw=0.6, alpha=0.35, color='#888888')
    ax3d.plot(np.cos(th),       np.sin(th),       np.zeros_like(th), **wire)
    ax3d.plot(np.cos(th),       np.zeros_like(th), np.sin(th),       **wire)
    ax3d.plot(np.zeros_like(th), np.cos(th),       np.sin(th),       **wire)

    # Cartesian axes + labels
    for vec, lbl in [([1, 0, 0], 'x'), ([0, 1, 0], 'y'), ([0, 0, 1], 'z')]:
        ax3d.plot([-vec[0], vec[0]], [-vec[1], vec[1]], [-vec[2], vec[2]],
                  color='#bbbbbb', lw=0.8, alpha=0.5)
        ax3d.text(vec[0] * 1.22, vec[1] * 1.22, vec[2] * 1.22, lbl,
                  fontsize=5, ha='center', va='center', color='#555555')

    # Pole labels
    ax3d.text(0, 0,  1.28, '|0⟩', fontsize=5, ha='center', color='#333333')
    ax3d.text(0, 0, -1.28, '|1⟩', fontsize=5, ha='center', color='#333333')

    # Optional trajectory path
    if show_trajectory is not None and len(show_trajectory) > 1:
        xs = [p[0] for p in show_trajectory]
        ys = [p[1] for p in show_trajectory]
        zs = [p[2] for p in show_trajectory]
        ax3d.plot(xs, ys, zs, color=vector_color, lw=1.5, alpha=0.55)

    # Bloch vector arrow
    ax3d.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                color=vector_color, arrow_length_ratio=0.18,
                linewidth=2.0, normalize=False)

    # State point
    ax3d.scatter([bv[0]], [bv[1]], [bv[2]],
                 color=point_color, s=55, zorder=10, depthshade=False)

    ax3d.set_xlim([-1.2, 1.2])
    ax3d.set_ylim([-1.2, 1.2])
    ax3d.set_zlim([-1.2, 1.2])
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_axis_off()

    # text2D keeps the title inside the axes box (set_title floats outside 3D axes)
    ax3d.text2D(0.5, 0.97,
                f'{title}\nBloch z = {bv[2]:.4f}  |  Purity = {pur:.4f}',
                transform=ax3d.transAxes, fontsize=6, ha='center', va='top',
                color='#222222')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

_THEME = dict(
    bg_fig='#fffdf8', bg_ax='#faf7f2', grid='#e8e2d8',
    edge='#bbbbbb', label='#222222', title='#111111',
    tick='#444444', legend_bg='#fffdf8', legend_edge='#cccccc',
    footer='#888888',
)

_COLORS = {
    'none':           '#1a6b3c',
    'drift':          '#8b2500',
    'gate-dependent': '#c0392b',
    'coherent':       '#6d4c41',
    'heating':        '#e67e22',
    'leakage':        '#4a235a',
    'combined':       '#2c3e50',
}


def _apply_theme():
    plt.rcParams.update({
        'figure.facecolor':  _THEME['bg_fig'],
        'axes.facecolor':    _THEME['bg_ax'],
        'axes.edgecolor':    _THEME['edge'],
        'axes.labelcolor':   _THEME['label'],
        'axes.titlecolor':   _THEME['title'],
        'xtick.color':       _THEME['tick'],
        'ytick.color':       _THEME['tick'],
        'grid.color':        _THEME['grid'],
        'text.color':        _THEME['label'],
        'legend.facecolor':  _THEME['legend_bg'],
        'legend.edgecolor':  _THEME['legend_edge'],
        'font.family':       'DejaVu Sans',
        'font.size':         7,
    })


def show_scrollable(fig_or_pairs, title='RB Simulation', win_height=820):
    """Show one or more figures each in a separate scrollable tkinter window.

    fig_or_pairs : single Figure  OR  list of (Figure, title_str) pairs.
    When multiple pairs are given every window opens simultaneously.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
    except ImportError:
        pairs = (fig_or_pairs if isinstance(fig_or_pairs, list)
                 else [(fig_or_pairs, title)])
        for fig, _ in pairs:
            plt.show()
        return

    pairs = (fig_or_pairs if isinstance(fig_or_pairs, list)
             else [(fig_or_pairs, title)])

    root = tk.Tk()
    root.withdraw()   # invisible root — only Toplevel windows are shown

    def _build(fig, ttl, offset_x=0):
        plt.close(fig)
        win = tk.Toplevel(root)
        win.title(ttl)

        outer = tk.Frame(win)
        outer.pack(fill=tk.BOTH, expand=True)

        vbar = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk_cv = tk.Canvas(outer, yscrollcommand=vbar.set)
        tk_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=tk_cv.yview)

        inner = tk.Frame(tk_cv)
        tk_cv.create_window((0, 0), window=inner, anchor='nw')

        mpl_cv  = FigureCanvasTkAgg(fig, master=inner)
        toolbar = NavigationToolbar2Tk(mpl_cv, inner)
        toolbar.update()
        mpl_cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        mpl_cv.draw()

        def _on_configure(_):
            tk_cv.configure(scrollregion=tk_cv.bbox('all'))
        inner.bind('<Configure>', _on_configure)

        def _on_wheel(event):
            tk_cv.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        def _on_wheel_linux(event):
            tk_cv.yview_scroll(-1 if event.num == 4 else 1, 'units')

        # Rebind scroll to this window's canvas whenever it gains focus
        def _bind_scroll(_=None):
            root.bind_all('<MouseWheel>', _on_wheel)
            root.bind_all('<Button-4>',   _on_wheel_linux)
            root.bind_all('<Button-5>',   _on_wheel_linux)
        win.bind('<FocusIn>', _bind_scroll)
        _bind_scroll()

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        fw = int(fig.get_figwidth() * fig.dpi) + 40
        win.geometry(
            f'{min(fw, sw - 20)}x{min(win_height, sh - 60)}+{offset_x}+0')

    for i, (fig, ttl) in enumerate(pairs):
        _build(fig, ttl, offset_x=i * 60)

    root.mainloop()


def plot_single_run(sequence_lengths, survival_mean, survival_std,
                    popt, epc, epc_err, eps, n_sequences, all_data,
                    rho_trajectory, error_type, error_magnitude):
    """
    Figure 1 : RB decay curve (survival probability vs sequence length)
    Figure 2 : Residuals | Box plots | Bloch sphere (ideal) | Bloch sphere (decayed)
    Both windows open simultaneously and are independently scrollable.
    """
    _apply_theme()

    A_fit, p_fit, B_fit = popt
    m_arr    = np.array(sequence_lengths, dtype=float)
    m_fine   = np.linspace(0, max(sequence_lengths), 800)
    P_fit    = rb_model(m_fine, *popt)
    P_data   = rb_model(m_arr,  *popt)
    p_theory = 1 - 4*eps/3
    P_theory = rb_model(m_fine, 0.5, p_theory, 0.5)
    col      = _COLORS.get(error_type, '#333333')
    traj     = build_bloch_trajectory(eps, m_max=max(sequence_lengths))

    # ── Figure 1: RB decay curve ─────────────────────────────────────────────
    fig1 = plt.figure(figsize=(6.5, 3.0), facecolor=_THEME['bg_fig'])
    ax   = fig1.add_subplot(111)
    ax.set_facecolor(_THEME['bg_ax'])
    ax.fill_between(m_fine, P_fit - 0.010, P_fit + 0.010,
                    alpha=0.22, color=col, linewidth=0)
    ax.plot(m_fine, P_theory, '--', color='#888888', lw=1.4, alpha=0.80,
            label=f'Theory  p = 1−4ε/3 = {p_theory:.5f}')
    ax.plot(m_fine, P_fit, '-', color=col, lw=2.0,
            label='RB Fit   P(m) = A·pᵐ + B')
    ax.errorbar(m_arr, survival_mean, yerr=survival_std,
                fmt='o', color=col, markersize=5, capsize=4,
                capthick=1.4, elinewidth=1.4, linewidth=0, zorder=6,
                label=f'Simulation  (N={n_sequences} seq/point, QuTiP Kraus)')
    ax.axhline(B_fit, color='#888888', lw=1.0, ls=':', alpha=0.85,
               label=f'SPAM floor  B = {B_fit:.4f}')
    ax.set_xlabel('Sequence Length  m', fontsize=7, labelpad=3)
    ax.set_ylabel('Survival Probability  P(m)', fontsize=7, labelpad=3)
    ax.set_title(
        f'Single-Qubit RB  ·  Error: {error_type}  (×{error_magnitude})  ·  '
        f'QuTiP Kraus  ·  C₁(24)  ·  Helsen et al. (2020)',
        fontsize=7, pad=4)
    ax.set_ylim([max(-0.05, B_fit - 0.15), 1.08])
    ax.set_xlim([-1, max(sequence_lengths) + 3])
    ax.grid(True, alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.01, 0.40,
            "━━ FITTED PARAMETERS ━━\n"
            f"   A  =  {A_fit:.6f}\n"
            f"   p  =  {p_fit:.6f}\n"
            f"   B  =  {B_fit:.6f}\n"
            "━━ EXTRACTED METRICS ━━\n"
            f"   F_avg  = 1−EPC  = {1-epc:.6f}\n"
            f"   EPC   = ¾(1−p) = {epc:.6f}\n"
            f"              ±{epc_err:.6f}\n"
            "━━━ GROUND TRUTH ━━━━\n"
            f"   ε (Pauli rate) = {eps:.6f}\n"
            f"   p_theory       = {p_theory:.6f}\n"
            f"   EPC_theory     = {eps:.6f}\n"
            f"   |ΔEPC|         = {abs(epc-eps):.6f}",
            transform=ax.transAxes, fontsize=5.5, va='top', ha='left',
            fontfamily='monospace', color='#111111',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8f0',
                      edgecolor=col, alpha=0.97))
    ax.legend(loc='upper left', fontsize=6, framealpha=0.95,
              labelcolor='#111111')
    fig1.tight_layout(pad=0.8)

    # ── Figure 2: residuals + box plots + Bloch spheres ──────────────────────
    fig2 = plt.figure(figsize=(6.5, 5.0), facecolor=_THEME['bg_fig'])
    gs2  = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.30, wspace=0.25,
                             left=0.10, right=0.96, top=0.94, bottom=0.08,
                             height_ratios=[1.0, 1.4])

    # ── Top-left: residuals ──────────────────────────────────────────────────
    ax_r = fig2.add_subplot(gs2[0, 0])
    ax_r.set_facecolor(_THEME['bg_ax'])
    residuals = survival_mean - P_data
    width     = max(sequence_lengths) / len(sequence_lengths) * 0.62
    ax_r.bar(m_arr, residuals, width=width,
             color=[col if r >= 0 else '#aaaaaa' for r in residuals],
             alpha=0.82, edgecolor='white', linewidth=0.5)
    ax_r.errorbar(m_arr, residuals, yerr=survival_std,
                  fmt='none', color='#888888', capsize=3, elinewidth=1.0)
    ax_r.axhline(0, color=col, lw=1.5, ls='--', alpha=0.7)
    ax_r.set_xlabel('Sequence Length  m', fontsize=7)
    ax_r.set_ylabel('Residual', fontsize=7)
    ax_r.set_title('Fit Residuals  (P_sim − P_fit)', fontsize=7)
    ax_r.grid(True, alpha=0.4)
    ax_r.spines[['top', 'right']].set_visible(False)
    rms = np.sqrt(np.mean(residuals**2))
    ax_r.text(0.97, 0.94, f'RMS = {rms:.5f}',
              transform=ax_r.transAxes, ha='right', va='top', fontsize=6,
              color='#333333',
              bbox=dict(facecolor='white', edgecolor='#cccccc',
                        alpha=0.90, boxstyle='round,pad=0.4'))

    # ── Top-right: box plots ─────────────────────────────────────────────────
    ax_b = fig2.add_subplot(gs2[0, 1])
    ax_b.set_facecolor(_THEME['bg_ax'])
    sel_idx = np.linspace(0, len(sequence_lengths) - 1,
                          min(8, len(sequence_lengths)), dtype=int)
    sel_ms  = [sequence_lengths[i] for i in sel_idx]
    ax_b.boxplot(
        [all_data[m] for m in sel_ms],
        positions=list(range(len(sel_ms))), patch_artist=True,
        medianprops=dict(color=col, lw=2.5),
        boxprops=dict(facecolor=_THEME['legend_bg'], color=col, lw=1.5),
        whiskerprops=dict(color='#888888', lw=1.2),
        capprops=dict(color='#888888', lw=1.5),
        flierprops=dict(marker='.', color='#aaaaaa', markersize=3, alpha=0.6),
    )
    ax_b.set_xticks(range(len(sel_ms)))
    ax_b.set_xticklabels([str(m) for m in sel_ms], fontsize=6)
    ax_b.set_xlabel('Sequence Length  m', fontsize=7)
    ax_b.set_ylabel('Survival Probability', fontsize=7)
    ax_b.set_title('Sequence-to-Sequence Distribution  (selected m)', fontsize=7)
    ax_b.set_ylim([-0.05, 1.08])
    ax_b.grid(True, alpha=0.4, axis='y')
    ax_b.spines[['top', 'right']].set_visible(False)

    # ── Bottom-left: Bloch sphere — ideal qubit ──────────────────────────────
    ax_bl = fig2.add_subplot(gs2[1, 0], projection='3d')
    rho_ideal = qt.ket2dm(qt.basis(2, 0))
    add_bloch_to_ax(ax_bl, rho_ideal,
                    title='Ideal Qubit  |0⟩\n(no noise, m = 0)',
                    point_color='#1a6b3c',
                    vector_color='#1a6b3c',
                    show_trajectory=None)

    # ── Bottom-right: Bloch sphere — state after RB decay ───────────────────
    ax_br = fig2.add_subplot(gs2[1, 1], projection='3d')
    rho_final = rho_trajectory[-1]
    m_final   = sequence_lengths[-1]
    add_bloch_to_ax(ax_br, rho_final,
                    title=f'After RB Decay  (m = {m_final})\n'
                          f'error model: {error_type}',
                    point_color=col,
                    vector_color=col,
                    show_trajectory=traj[:m_final + 2])

    fig2.text(0.5, 0.02,
              f'QuTiP {qt.__version__} · NumPy · SciPy curve_fit · Matplotlib  |  '
              f'ε={eps}  ·  {n_sequences} seq/point  ·  error: {error_type}',
              ha='center', fontsize=5, color=_THEME['footer'])

    show_scrollable([(fig1, 'RB — Decay Curve'),
                     (fig2, 'RB — Residuals & Bloch Spheres')])


def plot_comparison(sequence_lengths, results, eps, n_sequences):
    """
    4-panel comparison figure:
      Top-left   : decay curves for all error models
      Top-right  : EPC bar chart
      Bottom-left: Bloch sphere — ideal qubit
      Bottom-right: Bloch sphere — baseline (none) after full decay
    """
    _apply_theme()

    m_fine      = np.linspace(0, max(sequence_lengths), 500)
    m_arr       = np.array(sequence_lengths, dtype=float)
    error_types = list(results.keys())
    traj        = build_bloch_trajectory(eps, m_max=max(sequence_lengths))

    fig = plt.figure(figsize=(6, 5), facecolor=_THEME['bg_fig'])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.21, wspace=0.22,
                            left=0.07, right=0.96, top=0.96, bottom=0.03,
                            height_ratios=[1.6, 1.4])

    # ── Top-left: decay curves ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(_THEME['bg_ax'])
    for et in error_types:
        r   = results[et]
        col = _COLORS.get(et, '#333333')
        ax1.plot(m_fine, rb_model(m_fine, r['A'], r['p_fit'], r['B']),
                 '-', lw=2.2, color=col,
                 label=f"{et.replace('-',' ').title():<18}  "
                       f"p={r['p_fit']:.5f}")
        ax1.errorbar(m_arr, r['mean'], yerr=r['std'],
                     fmt='o', color=col, markersize=4, alpha=0.5,
                     capsize=3, elinewidth=1.0, linewidth=0)
    ax1.set_xlabel('Sequence Length  m', fontsize=7)
    ax1.set_ylabel('Survival Probability  P(m)', fontsize=7)
    ax1.set_title('RB Decay Curves: All Error Models\n(QuTiP Kraus channel)',
                  fontsize=7, pad=4)
    ax1.legend(fontsize=5.5, loc='upper right', framealpha=0.95)
    ax1.set_ylim([0.38, 1.06])
    ax1.grid(True, alpha=0.4)
    ax1.spines[['top', 'right']].set_visible(False)

    # ── Top-right: EPC bar chart ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(_THEME['bg_ax'])
    x_pos      = np.arange(len(error_types))
    epc_vals   = [results[et]['epc']     for et in error_types]
    epc_errs   = [results[et]['epc_err'] for et in error_types]
    bar_colors = [_COLORS.get(et, '#333333') for et in error_types]
    bars = ax2.bar(x_pos, epc_vals, color=bar_colors,
                   alpha=0.85, edgecolor='white', linewidth=1.5, width=0.6)
    ax2.errorbar(x_pos, epc_vals, yerr=epc_errs,
                 fmt='none', color='#333333', capsize=6, elinewidth=2.0)
    ax2.axhline(eps, color='#333333', linestyle='--', lw=2, alpha=0.7,
                label=f'Ground truth  ε = {eps}')
    for bar, val in zip(bars, epc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=5.5, color='#111111', fontweight='bold')
    ax2.set_ylabel('EPC = (3/4)·(1−p)', fontsize=7)
    ax2.set_title('Extracted EPC per Error Model', fontsize=7, pad=4)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([et.replace('-', '\n').title() for et in error_types],
                        fontsize=6)
    ax2.set_ylim([0, max(epc_vals) * 1.35])
    ax2.grid(True, alpha=0.35, axis='y')
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.95)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── Bottom-left: ideal Bloch sphere ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    rho_ideal = qt.ket2dm(qt.basis(2, 0))
    add_bloch_to_ax(ax3, rho_ideal,
                    title='Ideal Qubit  |0⟩  (m = 0, no noise)',
                    point_color='#1a6b3c',
                    vector_color='#1a6b3c')

    # ── Bottom-right: decayed Bloch sphere (baseline) ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    rho_decayed = results['none']['rho_final']
    m_final     = sequence_lengths[-1]
    add_bloch_to_ax(ax4, rho_decayed,
                    title=f'After Baseline RB Decay  (m = {m_final})\n'
                          f'Pauli depolarizing  ε = {eps}',
                    point_color=_COLORS['none'],
                    vector_color=_COLORS['none'],
                    show_trajectory=traj[:m_final+2])

    fig.suptitle(
        'Single-Qubit RB: Error Model Comparison  +  Bloch Sphere Visualization\n'
        'QuTiP Kraus operators  ·  C₁ Clifford group (|G|=24)  ·  '
        'Helsen et al. (2020)',
        fontsize=7, y=0.98)
    fig.text(0.5, 0.008,
             f'QuTiP {qt.__version__} · NumPy · SciPy · Matplotlib  |  '
             f'ε={eps}  ·  {n_sequences} seq/point',
             ha='center', fontsize=5, color=_THEME['footer'])

    show_scrollable(fig, 'RB Error Model Comparison')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Single-qubit RB + error injection (QuTiP Kraus channel)')
    parser.add_argument('--error',     default='none',
                        choices=list(_ERROR_FN.keys()),
                        help='Error model  (default: none)')
    parser.add_argument('--magnitude', type=float, default=1.0,
                        help='Error scale factor  (default: 1.0)')
    parser.add_argument('--eps',       type=float, default=0.05,
                        help='Pauli error rate ε  (default: 0.05)')
    parser.add_argument('--nseq',      type=int,   default=300,
                        help='Sequences per length  (default: 300)')
    parser.add_argument('--seed',      type=int,   default=2024,
                        help='RNG seed  (default: 2024)')
    parser.add_argument('--compare',   action='store_true',
                        help='Run all 7 error models + comparison plot')
    args = parser.parse_args()

    EPS              = args.eps
    N_SEQUENCES      = args.nseq
    SEED             = args.seed
    SEQUENCE_LENGTHS = list(range(1, 101, 5))

    # ── Build and verify Clifford group ────────────────────────────────────
    print("\n" + "="*66)
    print("  SINGLE-QUBIT RB + ERROR INJECTION  (QuTiP Kraus channel)")
    print("  Framework: Helsen et al. (2020)  ·  Magesan et al. (2012)")
    print("="*66)
    print("\n[1] Building Clifford group C₁ via BFS(H, S)...")
    clifford_qobj, clifford_np = generate_clifford_group()
    lookup = build_inverse_lookup(clifford_np)
    print(f"    |C₁| = {len(clifford_np)} elements  (QuTiP Qobj + numpy arrays)")

    def _nkey(m):
        for v in m.flatten():
            if abs(v) > 1e-10:
                return tuple(np.round((m * np.conj(v)/abs(v)).flatten(), 8))
        return tuple(np.round(m.flatten(), 8))

    rng_v = np.random.default_rng(0)
    fails = sum(_nkey(clifford_np[i] @ clifford_np[j]) not in lookup
                for i, j in [(rng_v.integers(0,24), rng_v.integers(0,24))
                             for _ in range(2000)])
    u_ok  = all(np.allclose(c @ c.conj().T, np.eye(2), atol=1e-9)
                for c in clifford_np)
    print(f"    Closure  (2000 products): {'PASS' if fails==0 else f'FAIL ({fails})'}")
    print(f"    Unitarity:                {'PASS' if u_ok else 'FAIL'}")

    # Kraus channel verification
    K_test = build_kraus_ops(EPS)
    comp   = sum(K.dag() * K for K in K_test)
    print(f"    Kraus completeness ΣK†K=I: "
          f"{'PASS' if np.allclose(comp.full(), np.eye(2), atol=1e-9) else 'FAIL'}")

    # ── Run ────────────────────────────────────────────────────────────────
    if args.compare:
        print(f"\n[2] Running all {len(_ERROR_FN)} error models (QuTiP) ...")
        results = {}
        for et in _ERROR_FN:
            rng = np.random.default_rng(SEED)
            sm, ss, ad, rho_traj = run_rb_simulation(
                clifford_qobj, clifford_np, SEQUENCE_LENGTHS,
                N_SEQUENCES, EPS, error_type=et,
                error_magnitude=1.0, rng=rng)
            popt, _, epc, epc_err = fit_rb_curve(SEQUENCE_LENGTHS, sm, ss)
            results[et] = dict(mean=sm, std=ss, data=ad,
                               A=popt[0], p_fit=popt[1], B=popt[2],
                               epc=epc, epc_err=epc_err,
                               rho_final=rho_traj[-1])

        p_th = 1 - 4*EPS/3
        print(f"\n{'='*72}")
        print("  RESULTS SUMMARY")
        print(f"{'='*72}")
        print(f"  {'Error Model':<18} {'p_fit':>10} {'Δp':>10} "
              f"{'EPC':>10} {'ΔEPC':>10} {'Purity':>8}")
        print(f"  {'-'*68}")
        for et, r in results.items():
            pur = purity(r['rho_final'])
            print(f"  {et:<18} {r['p_fit']:>10.6f} {r['p_fit']-p_th:>+10.6f} "
                  f"{r['epc']:>10.6f} {r['epc']-EPS:>+10.6f} {pur:>8.4f}")
        print(f"  {'-'*68}")
        print(f"  {'Theory':<18} {p_th:>10.6f} {'0.000000':>10} "
              f"{EPS:>10.6f} {'0.000000':>10} {'0.5000':>8}")
        print(f"{'='*72}\n")

        print("[3] Displaying comparison plot + Bloch spheres ...")
        plot_comparison(SEQUENCE_LENGTHS, results, EPS, N_SEQUENCES)

    else:
        error_type = args.error
        magnitude  = args.magnitude

        print(f"\n[2] Running simulation  "
              f"(error={error_type}, magnitude={magnitude}, QuTiP Kraus)...")
        rng = np.random.default_rng(SEED)
        survival_mean, survival_std, all_data, rho_trajectory = run_rb_simulation(
            clifford_qobj, clifford_np, SEQUENCE_LENGTHS,
            N_SEQUENCES, EPS, error_type=error_type,
            error_magnitude=magnitude, rng=rng)

        print("[3] Fitting P(m) = A·pᵐ + B ...")
        popt, _, epc, epc_err = fit_rb_curve(
            SEQUENCE_LENGTHS, survival_mean, survival_std)
        A_fit, p_fit, B_fit = popt
        p_theory = 1 - 4*EPS/3

        rho_final = rho_trajectory[-1]
        print(f"\n{'='*66}")
        print(f"  RESULTS")
        print(f"{'='*66}")
        print(f"  Fitted A            = {A_fit:.7f}")
        print(f"  Fitted p            = {p_fit:.7f}  "
              f"(theory: {p_theory:.7f})")
        print(f"  Fitted B            = {B_fit:.7f}")
        print(f"  F_avg = 1−EPC       = {1-epc:.7f}")
        print(f"  EPC   = ¾·(1−p)     = {epc:.7f} ± {epc_err:.7f}")
        print(f"  EPC_theory          = {EPS:.7f}")
        print(f"  |ΔEPC|              = {abs(epc-EPS):.7f}")
        print(f"  Relative error      = {100*abs(epc-EPS)/EPS:.3f}%")
        bv = bloch_vector(rho_final)
        print(f"  Final Bloch vector  = ({bv[0]:.4f}, {bv[1]:.4f}, {bv[2]:.4f})")
        print(f"  Final purity Tr(ρ²) = {purity(rho_final):.6f}")
        print(f"{'='*66}\n")

        print("[4] Displaying 5-panel figure (decay + residuals + "
              "box plots + Bloch spheres) ...")
        plot_single_run(SEQUENCE_LENGTHS, survival_mean, survival_std,
                        popt, epc, epc_err, EPS, N_SEQUENCES, all_data,
                        rho_trajectory, error_type, magnitude)

        print(f"\n[DONE]  EPC = {epc:.5f} ± {epc_err:.5f}  |  "
              f"EPC_true = {EPS:.5f}  |  error = {error_type}")
