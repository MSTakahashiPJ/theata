"""tests/data/phonon_carryover_baseline.json を再生成する。

このファイルは「フォノン引き継ぎオプションを追加する *前* の
phonon_at_gates」を固定した回帰データ。既定動作 (carry_phonon=False)
を変えてしまう変更を test_phonon_carryover.py が検出するために使う。

意図的に基準値を更新するとき以外は走らせないこと。

Usage:
    python tests/gen_phonon_carryover_baseline.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
for sub in ('src', 'circuits', 'experiments'):
    sys.path.insert(0, str(ROOT / sub))

import matplotlib
matplotlib.use('Agg')

from exp_distance_scan import build_noisy_circuit, DISTANCE_CIRCUITS  # noqa: E402

DISTANCE = 3
ROUNDS = 3
OUT = ROOT / 'tests' / 'data' / 'phonon_carryover_baseline.json'


def main():
    fwd, rev, obs = DISTANCE_CIRCUITS[DISTANCE]
    _, results = build_noisy_circuit(fwd, rev, ROUNDS, obs, warn_on_clip=False)

    at_gates = []
    for r in results:
        at_gates.extend((r.phonon_data or {}).get('at_gates', []))
    payload = {
        'description': ('Reference phonon_at_gates for d=3, R=3 with '
                        'carry_phonon disabled (pre-carry-over baseline). '
                        'Regenerate with tests/gen_phonon_carryover_baseline.py.'),
        'distance': DISTANCE,
        'rounds': ROUNDS,
        'forward': fwd,
        'reverse': rev,
        'planning_clock_us': [(r.phonon_data or {}).get('planning_clock_us')
                              for r in results],
        'at_gates': at_gates,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')
    print(f"wrote {OUT} ({len(at_gates)} gates)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
