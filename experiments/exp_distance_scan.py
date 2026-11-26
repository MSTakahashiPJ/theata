"""Scan logical error rate p_L across code distances d=3, 5, 7.

Runs end-to-end pipeline (compile → recompile → noise inject → MWPM decode)
for each distance and produces a 2D scatter plot p_L vs d with 1-sigma error
bars. Each distance uses its forward / reverse_cnot circuit pair.

Output: results/noise_injection/distance_scan_{timestamp}.{json,png}

Usage:
    python experiments/exp_distance_scan.py --shots 50000
    python experiments/exp_distance_scan.py --distances 3 5 7 --rounds 3
    python experiments/exp_distance_scan.py --rounds 5 --carry-phonons
"""

import sys
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'circuits'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from pauli_lookup import PauliErrorLookup
from stim_recompiler import recompile_multi_round
from exp_noise_injection import (
    phonon_to_pauli_lookup_rates,
    inject_noise_approach_b,
    inject_transport_dephasing,
    default_noise_params,
)
from exp_multi_round import placement_greedy_alt


PAULI_LOOKUP = PauliErrorLookup()
LOOKUP_MODE = "strict"


# (forward_module_name, reverse_module_name, default_observable_qubits)
# Observable = top row (data 0..d-1), which is a horizontal Z chain — logical Z
# for the rotated surface code convention used by these snake circuits.
DISTANCE_CIRCUITS = {
    3: ('snake_reuse1', 'snake_reuse1_reverse', [0, 1, 2]),
    5: ('d5_reuse1', 'd5_reuse1_reverse_cnot', [0, 1, 2, 3, 4]),
    7: ('d7_reuse1', 'd7_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6]),
    9: ('d9_reuse1', 'd9_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    11: ('d11_reuse1', 'd11_reuse1_reverse_cnot', list(range(11))),
    13: ('d13_reuse1', 'd13_reuse1_reverse_cnot', list(range(13))),
}


def _compile_round(circuit_name, initial_placement, arch, initial_zone_contents=None,
                   initial_phonons=None, initial_clock_us=None):
    mod = __import__(circuit_name)
    program = parse(mod.CIRCUIT)
    dag = DAGBuilder(program, stabilizer_parallel=False).build()
    if initial_zone_contents is not None:
        ordered = {}
        for zone_id in sorted(initial_zone_contents.keys()):
            for ion_id in initial_zone_contents[zone_id]:
                ordered[ion_id] = zone_id
        initial_placement = ordered
    compiler = QCCDCompiler(arch, config=CompilerConfig())
    result = compiler.compile(dag, initial_placement,
                              initial_phonons=initial_phonons,
                              initial_clock_us=initial_clock_us)
    return result


def _carry_over_from(result):
    """CompileResult から次ラウンドへ渡す (initial_phonons, initial_clock_us)。

    initial_phonons は compile() が受ける {'axial': {...}, 'transversal': {...},
    'transversal2': {...}} 形式 (hackathon/make_test_circuits.compile_rounds と同じ)。
    フォノン追跡が無効 (phonon_data が None) なら (None, None) を返す。
    """
    pd = result.phonon_data
    if not pd:
        return None, None
    phonons = {'axial': pd.get('per_ion_axial', {}),
               'transversal': pd.get('per_ion_transversal', {}),
               'transversal2': pd.get('per_ion_transversal2', {})}
    return phonons, pd.get('planning_clock_us', 0.0)


def build_noisy_circuit(forward, reverse, rounds, observable, mode='lumped',
                        override_n_bar=None, phonon_scale=1.0,
                        warn_on_clip=True, carry_phonons=False):
    """
    Args:
        carry_phonons: True なら QEC ラウンド境界でフォノン数と計画時近似時計を
            0 にリセットせず、前ラウンドの最終値を次ラウンドの初期値にする。
            False (既定) は論文設定 (ラウンドごとに 0 から再出発)。
    """
    fwd_mod = __import__(forward)
    half = (len(fwd_mod.DATA_QUBITS) + 1) // 2
    mem_cap = max(20, half + len(fwd_mod.ANCILLA_QUBITS) + 4)
    arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=mem_cap)

    # greedy_alt placement (canonical, matches exp_build_examples.py)
    initial = placement_greedy_alt(
        fwd_mod.DATA_QUBITS, fwd_mod.ANCILLA_QUBITS, fwd_mod)

    results = []
    current_placement = initial
    current_zone_contents = None
    current_phonons = None
    current_clock_us = None
    for r in range(1, rounds + 1):
        circuit_name = forward if r % 2 == 1 else reverse
        result = _compile_round(circuit_name, current_placement, arch,
                                initial_zone_contents=current_zone_contents,
                                initial_phonons=current_phonons,
                                initial_clock_us=current_clock_us)
        results.append(result)
        current_placement = result.final_placement
        current_zone_contents = result.final_zone_contents
        if carry_phonons:
            current_phonons, current_clock_us = _carry_over_from(result)

    circuit = recompile_multi_round(results, observable_column=observable)

    all_phonon_gates = []
    for r in results:
        pd = r.phonon_data or {}
        all_phonon_gates.extend(pd.get('at_gates', []))
    # PAULI_CHANNEL_2 lookup (strict): transversal phonon × phonon_scale → 15 Pauli probs
    # override_n_bar が指定されていれば全 MS で固定値 (= heating 寄与 0)
    pauli_rates = phonon_to_pauli_lookup_rates(
        all_phonon_gates, PAULI_LOOKUP, mode=LOOKUP_MODE,
        override_n_bar=override_n_bar, phonon_scale=phonon_scale,
        warn_on_clip=warn_on_clip)
    noise_params = default_noise_params()
    noisy = inject_noise_approach_b(circuit, pauli_rates, noise_params)
    noisy = inject_transport_dephasing(noisy, results, noise_params)

    return noisy, results


def simulate(circuit, shots, seed=None):
    """Returns (n_errors, n_shots)."""
    sampler = (circuit.compile_detector_sampler(seed=seed)
               if seed is not None else circuit.compile_detector_sampler())
    # approximate_disjoint_errors=True: PAULI_CHANNEL_2 を含む回路で必須
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det_samples, obs_actual = sampler.sample(shots, separate_observables=True)
    obs_predicted = matcher.decode_batch(det_samples)
    errors_per_shot = np.any(obs_predicted != obs_actual, axis=1)
    return int(errors_per_shot.sum()), shots


def main():
    parser = argparse.ArgumentParser(description='Distance scan')
    parser.add_argument('--distances', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--shots', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='results/noise_injection',
                        help='Output directory for plot+JSON')
    parser.add_argument('--no-heating', action='store_true',
                        help='Force n_bar=0.01 for all MS gates (= zero heating '
                             'contribution scenario). Lookup table is no longer '
                             'exceeded, so d=9 becomes simulatable too.')
    parser.add_argument('--phonon-scale', type=float, default=1.0,
                        help='Multiplier on transversal phonon at each MS '
                             '(1.0 = no scaling, 0.5 = half heating, etc.). '
                             '--no-heating で上書きされる。')
    parser.add_argument('--quiet-clip', action='store_true',
                        help='lookup table 上限超過の per-gate 警告を抑制し、'
                             '距離ごとの 1 行サマリだけ表示する')
    parser.add_argument('--carry-phonons', action='store_true',
                        help='QEC ラウンド境界でフォノン数と計画時近似時計を '
                             '0 にリセットせず、前ラウンドの最終値を引き継ぐ。'
                             '既定 (リセットあり) は加熱の効果を下限に見積もる'
                             '論文設定なので、こちらが上限側になる。')
    args = parser.parse_args()

    override_n_bar = 0.01 if args.no_heating else None
    phonon_scale = args.phonon_scale
    if args.no_heating:
        label_suffix = '_no_heating'
    elif phonon_scale != 1.0:
        # ファイル名に scale を埋め込む (例: _scale0.5)
        label_suffix = f'_scale{phonon_scale}'
    else:
        label_suffix = ''
    if args.carry_phonons:
        label_suffix += '_carry'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {
        'rounds': args.rounds,
        'shots_per_d': args.shots,
        'seed': args.seed,
        'transport_mode': 'lumped',
        'heating_mode': (
            'off (n_bar=0.01 forced)' if args.no_heating else
            (f'scaled ({phonon_scale}x transversal phonon)' if phonon_scale != 1.0
             else 'on (actual transversal phonon)')
        ),
        'override_n_bar': override_n_bar,
        'phonon_scale': phonon_scale,
        'carry_phonons': args.carry_phonons,
        'round_boundary': ('carry over previous round final phonons + clock'
                           if args.carry_phonons else 'reset to zero'),
        'results': [],
    }

    for d in args.distances:
        if d not in DISTANCE_CIRCUITS:
            print(f"[skip] d={d} not configured")
            continue
        fwd, rev, obs = DISTANCE_CIRCUITS[d]
        print(f"=== d={d} ({fwd} / {rev}, observable={obs}) ===")
        noisy, results = build_noisy_circuit(
            fwd, rev, args.rounds, obs,
            override_n_bar=override_n_bar, phonon_scale=phonon_scale,
            warn_on_clip=not args.quiet_clip,
            carry_phonons=args.carry_phonons)
        # Quick info
        n_qubits = noisy.num_qubits
        n_inst = sum(1 for _ in noisy)
        print(f"  num_qubits: {n_qubits}, instructions: {n_inst}")

        # Decode
        n_err, n_shots = simulate(noisy, args.shots, seed=args.seed + d)
        p_L = n_err / n_shots
        stderr = math.sqrt(p_L * (1 - p_L) / n_shots) if n_shots > 0 else 0
        per_round = 1 - (1 - p_L) ** (1.0 / args.rounds) if p_L > 0 else 0
        print(f"  errors: {n_err}/{n_shots}")
        print(f"  p_L (over {args.rounds} rounds): {p_L:.4e} +/- {stderr:.4e}")
        print(f"  per-round (approx):              {per_round:.4e}")

        data['results'].append({
            'd': d,
            'forward': fwd,
            'reverse': rev,
            'observable': obs,
            'errors': n_err,
            'shots': n_shots,
            'p_L': p_L,
            'stderr_1sigma': stderr,
            'per_round_p_L': per_round,
        })

    # Save JSON
    json_path = outdir / f'distance_scan{label_suffix}_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot
    if not data['results']:
        print("[skip plot] no results")
        return 0

    ds = [r['d'] for r in data['results']]
    pLs = [r['p_L'] for r in data['results']]
    errs = [r['stderr_1sigma'] for r in data['results']]

    # 短い title (グラフ枠で見切れにくい)
    if args.no_heating:
        title = 'error vs distance (heating free)'
    elif phonon_scale != 1.0:
        title = f'error vs distance (heating x{phonon_scale})'
    else:
        title = 'error vs distance'
    if args.carry_phonons:
        title += ' (phonon carry-over)'
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ds, pLs, yerr=errs, fmt='o-', capsize=4, markersize=8,
                color='C0', label=f'{args.rounds} rounds, {args.shots} shots/d')
    ax.set_xlabel('Code distance d')
    ax.set_ylabel(r'Logical error rate $p_L$')
    ax.set_yscale('log')
    ax.set_xticks(ds)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = outdir / f'distance_scan{label_suffix}_{timestamp}.png'
    fig.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
