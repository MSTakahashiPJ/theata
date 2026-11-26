"""QEC ラウンド境界でのフォノン引き継ぎ (phonon carry-over) のテスト。

背景:
  THeatA は既定でラウンドごとに新しい QCCDState を作るため、フォノン数と
  計画時近似時計がラウンド境界で 0 に戻る (論文設定)。
  QCCDCompiler.compile(initial_phonons=..., initial_clock_us=...) と
  exp_distance_scan の --carry-phonons は、この 0 リセットをやめて
  前ラウンド最終値を次ラウンドの初期値にする opt-in オプション。

検証項目:
- 既定 (オプションなし) では phonon_at_gates が従来と完全一致する
- R=1 ではオプションの有無によらず一致する
- 引き継ぎありでは、ラウンド r+1 開始時の値がラウンド r の最終値に等しい
- time_us と tr1 がラウンド境界で連続している

Usage:
    python tests/test_phonon_carryover.py
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent
for sub in ('src', 'circuits', 'experiments'):
    sys.path.insert(0, str(ROOT / sub))

import matplotlib                                   # noqa: E402
matplotlib.use('Agg')

from qccd_parser import parse                       # noqa: E402
from dag_builder import DAGBuilder                  # noqa: E402
from qccd_compiler import (                         # noqa: E402
    QCCDArchitecture, QCCDCompiler, CompilerConfig, MovementPlanner,
)
from sim_params import load_time_parameters         # noqa: E402

PARAMS_DIR = ROOT / 'params'
BASELINE = ROOT / 'tests' / 'data' / 'phonon_carryover_baseline.json'

_CODE = """
QUBIT 0 AT (1, 0)
QUBIT 1 AT (1, 1)
QUBIT 2 AT (1, 2)
MS 0 1 0.25
MS 1 2 0.25
MEASURE 2
"""
_PLACEMENT = {0: 1, 1: 1, 2: 1}


def _uniform(value):
    """全イオン共通の値を initial_phonons のペイロード形式で返す"""
    return {mode: {ion: value for ion in _PLACEMENT}
            for mode in ('axial', 'transversal', 'transversal2')}


def _compile(tp=None, initial_phonons=None, initial_clock_us=None):
    arch = QCCDArchitecture.simple_mlm()
    dag = DAGBuilder(parse(_CODE)).build()
    comp = QCCDCompiler(arch, config=CompilerConfig(), time_params=tp)
    return comp.compile(dag, dict(_PLACEMENT),
                        initial_phonons=initial_phonons,
                        initial_clock_us=initial_clock_us)


def _phonons_of(result):
    """compile() に渡せる形式で per-ion フォノンを取り出す"""
    pd = result.phonon_data
    return {'axial': pd['per_ion_axial'],
            'transversal': pd['per_ion_transversal'],
            'transversal2': pd['per_ion_transversal2']}


# ------------------------------------------------------------------
# compile() レベル: 小さい回路でアフィン性と引き継ぎ意味論を見る
# ------------------------------------------------------------------

def test_default_is_unchanged():
    """引数なしは「全モード 0 / 時計 0 から」と完全に同じ結果"""
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    res_none = _compile(tp)
    res_zero = _compile(tp, initial_phonons=_uniform(0.0), initial_clock_us=0.0)
    assert res_none.phonon_data['at_gates'] == res_zero.phonon_data['at_gates']
    assert (res_none.phonon_data['planning_clock_us']
            == res_zero.phonon_data['planning_clock_us'])
    print("test_default_is_unchanged: OK")


def test_uniform_seed_shifts_every_gate_by_the_same_amount():
    """一様な初期値 c は全ゲート記録を厳密に +c だけ平行移動させる。

    split / merge / shuttle / swap / advance_time はすべてアフィンで、
    merge の平均化は一様な定数を保つため、冷却なしなら厳密に成り立つ。
    """
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    base = _compile(tp)
    c_ax, c_tr1, c_tr2 = 0.7, 1.3, 2.1
    seeded = _compile(tp, initial_phonons={
        'axial': {ion: c_ax for ion in _PLACEMENT},
        'transversal': {ion: c_tr1 for ion in _PLACEMENT},
        'transversal2': {ion: c_tr2 for ion in _PLACEMENT}})

    g0 = base.phonon_data['at_gates']
    g1 = seeded.phonon_data['at_gates']
    assert len(g0) == len(g1) and len(g0) > 0
    for a, b in zip(g0, g1):
        assert a['gate'] == b['gate']
        assert a['time_us'] == b['time_us']   # 時計はフォノン値に依存しない
        for ion, pa in a['phonons'].items():
            pb = b['phonons'][ion]
            assert abs((pb[0] - pa[0]) - c_ax) < 1e-9
            assert abs((pb[1] - pa[1]) - c_tr1) < 1e-9
            assert abs((pb[2] - pa[2]) - c_tr2) < 1e-9
    print("test_uniform_seed_shifts_every_gate_by_the_same_amount: OK")


def test_initial_clock_offsets_recorded_times():
    """initial_clock_us は記録時刻と planning_clock_us をそのままずらす"""
    tp = load_time_parameters()          # レート 0 → フォノンは時計に依存しない
    base = _compile(tp)
    offset = 12345.0
    shifted = _compile(tp, initial_clock_us=offset)

    assert (abs(shifted.phonon_data['planning_clock_us']
                - (base.phonon_data['planning_clock_us'] + offset)) < 1e-9)
    for a, b in zip(base.phonon_data['at_gates'],
                    shifted.phonon_data['at_gates']):
        assert abs((b['time_us'] - a['time_us']) - offset) < 1e-9
    # フォノン値そのものは変わらない (レート 0 なので時計は加熱に効かない)
    for a, b in zip(base.phonon_data['at_gates'],
                    shifted.phonon_data['at_gates']):
        assert a['phonons'] == b['phonons']
    print("test_initial_clock_offsets_recorded_times: OK")


def test_initial_clock_is_needed_for_background_heating():
    """背景加熱ありでは、時計を引き継がないと加熱が t=0 から測り直しになる"""
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    r1 = _compile(tp)
    seed = _phonons_of(r1)
    clock = r1.phonon_data['planning_clock_us']
    assert clock > 0.0

    without_clock = _compile(tp, initial_phonons=seed)
    with_clock = _compile(tp, initial_phonons=seed, initial_clock_us=clock)

    # フォノン値は同一 (加熱は dt にしか依存せず、絶対時刻には依存しない)
    assert (without_clock.phonon_data['per_ion_transversal']
            == with_clock.phonon_data['per_ion_transversal'])
    # 記録時刻だけがちょうど clock ぶんずれる
    for a, b in zip(without_clock.phonon_data['at_gates'],
                    with_clock.phonon_data['at_gates']):
        assert abs((b['time_us'] - a['time_us']) - clock) < 1e-9
    assert without_clock.phonon_data['at_gates'][0]['time_us'] < clock
    assert with_clock.phonon_data['at_gates'][0]['time_us'] >= clock
    print("test_initial_clock_is_needed_for_background_heating: OK")


class _CapturingPlanner(MovementPlanner):
    """plan 開始時の state を記録する planner (引き継ぎ値の検証用)"""

    seeded_clock_us = None

    def plan(self, dag, initial_placement=None, state=None, **kwargs):
        if state is not None:
            self.seeded_axial = dict(state.phonon_axial)
            self.seeded_transversal = dict(state.phonon_transversal)
            self.seeded_transversal2 = dict(state.phonon_transversal2)
            self.seeded_clock_us = state.clock_us
        return super().plan(dag, initial_placement, state, **kwargs)


class _CapturingCompiler(QCCDCompiler):
    def __init__(self, architecture, config=None, time_params=None):
        super().__init__(architecture, config, time_params=time_params)
        self._planner = _CapturingPlanner(architecture, self.config,
                                          time_params=time_params)


def test_next_round_starts_at_previous_round_end():
    """ラウンド r+1 の開始値 == ラウンド r の最終値 (3 モード + 時計)"""
    tp = load_time_parameters(PARAMS_DIR / 'mainz_heating.json')
    arch = QCCDArchitecture.simple_mlm()

    r1 = _compile(tp)
    pd1 = r1.phonon_data

    comp = _CapturingCompiler(arch, CompilerConfig(), time_params=tp)
    comp.compile(DAGBuilder(parse(_CODE)).build(), dict(_PLACEMENT),
                 initial_phonons=_phonons_of(r1),
                 initial_clock_us=pd1['planning_clock_us'])
    planner = comp._planner

    assert planner.seeded_axial == pd1['per_ion_axial']
    assert planner.seeded_transversal == pd1['per_ion_transversal']
    assert planner.seeded_transversal2 == pd1['per_ion_transversal2']
    assert planner.seeded_clock_us == pd1['planning_clock_us']

    # 引き継がない場合は state を作らない = 従来経路 (対照)
    comp0 = _CapturingCompiler(arch, CompilerConfig(), time_params=tp)
    comp0.compile(DAGBuilder(parse(_CODE)).build(), dict(_PLACEMENT))
    assert comp0._planner.seeded_clock_us is None
    print("test_next_round_starts_at_previous_round_end: OK")


# ------------------------------------------------------------------
# exp_distance_scan レベル: マルチラウンドの実際の経路
# ------------------------------------------------------------------

def _rounds_at_gates(results):
    return [(r.phonon_data or {}).get('at_gates', []) for r in results]


def _build(rounds, carry_phonons):
    from exp_distance_scan import build_noisy_circuit, DISTANCE_CIRCUITS
    fwd, rev, obs = DISTANCE_CIRCUITS[3]
    _, results = build_noisy_circuit(fwd, rev, rounds, obs, warn_on_clip=False,
                                     carry_phonons=carry_phonons)
    return results


def test_default_matches_stored_baseline():
    """carry_phonons=False の d=3, R=3 が基準データと完全一致する"""
    reference = json.loads(BASELINE.read_text(encoding='utf-8'))
    results = _build(reference['rounds'], carry_phonons=False)

    got = []
    for per_round in _rounds_at_gates(results):
        got.extend(per_round)
    # JSON ラウンドトリップで phonons の int キーが文字列になるので揃える
    got_json = json.loads(json.dumps(got, sort_keys=True))
    assert got_json == reference['at_gates'], (
        "carry_phonons=False の phonon_at_gates が基準データと一致しない "
        "(既定動作が変わっている)")
    assert ([(r.phonon_data or {}).get('planning_clock_us') for r in results]
            == reference['planning_clock_us'])
    print(f"test_default_matches_stored_baseline: OK ({len(got)} gates)")


def test_single_round_identical_with_and_without_option():
    """R=1 では引き継ぎの有無で結果が一致する (引き継ぐ前ラウンドがない)"""
    off = _rounds_at_gates(_build(1, carry_phonons=False))
    on = _rounds_at_gates(_build(1, carry_phonons=True))
    assert off == on
    print("test_single_round_identical_with_and_without_option: OK")


def test_carry_over_is_continuous_across_round_boundaries():
    """引き継ぎあり: 時計と tr1 がラウンド境界で途切れない"""
    rounds = 3
    off = _build(rounds, carry_phonons=False)
    on = _build(rounds, carry_phonons=True)

    # ラウンド 1 は両者同一 (引き継ぐ前ラウンドがない)
    assert off[0].phonon_data['at_gates'] == on[0].phonon_data['at_gates']

    for r in range(1, rounds):
        prev, cur = on[r - 1].phonon_data, on[r].phonon_data
        # 時計の連続性: 当ラウンド最初のゲートは前ラウンド終端以降に起きる
        assert cur['at_gates'][0]['time_us'] >= prev['planning_clock_us'] - 1e-9
        assert cur['planning_clock_us'] > prev['planning_clock_us']
        # リセットありなら逆に 0 付近から再出発する (対照)
        off_cur = off[r].phonon_data
        assert (off_cur['at_gates'][0]['time_us']
                < off[r - 1].phonon_data['planning_clock_us'])

        # tr1 の連続性: 加熱と merge 平均は値を [min, max] の外へ出さないので、
        # 当ラウンド最初のゲートの tr1 は前ラウンド終端の最小値を下回らない
        prev_min_tr1 = min(prev['per_ion_transversal'].values())
        first_gate_tr1 = [p[1] for p in cur['at_gates'][0]['phonons'].values()]
        assert min(first_gate_tr1) >= prev_min_tr1 - 1e-9
        # かつ、リセットありの同じゲートより必ず熱い
        off_first = [p[1] for p in off_cur['at_gates'][0]['phonons'].values()]
        assert min(first_gate_tr1) > min(off_first)
    print("test_carry_over_is_continuous_across_round_boundaries: OK")


def test_carry_over_grows_with_rounds():
    """引き継ぎありでは最終 tr1 がラウンド数とともに単調に増える"""
    finals = {}
    for rounds in (1, 2, 4):
        results = _build(rounds, carry_phonons=True)
        finals[rounds] = max(
            results[-1].phonon_data['per_ion_transversal'].values())
    assert finals[1] < finals[2] < finals[4], finals
    print(f"test_carry_over_grows_with_rounds: OK (max tr1 {finals})")


if __name__ == "__main__":
    test_default_is_unchanged()
    test_uniform_seed_shifts_every_gate_by_the_same_amount()
    test_initial_clock_offsets_recorded_times()
    test_initial_clock_is_needed_for_background_heating()
    test_next_round_starts_at_previous_round_end()
    test_default_matches_stored_baseline()
    test_single_round_identical_with_and_without_option()
    test_carry_over_is_continuous_across_round_boundaries()
    test_carry_over_grows_with_rounds()
    print("\nAll phonon carry-over tests passed.")
