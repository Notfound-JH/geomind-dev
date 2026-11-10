"""Offline verification of both Piper articulations in native target recordings."""
from pathlib import Path
import argparse
import json
import numpy as np
from geomind.episodes.resolver import _check


def audit_dual(root):
    root = Path(root)
    report = json.loads((root / 'physics/report.json').read_text())
    rows = []
    for case in report['results']:
        identity = case['episode_id'] + ('_negative' if case['negative_control'] else '')
        saved = np.load(root / 'physics/traces' / f'{identity}_dual_controls.npz', allow_pickle=False)
        data = saved['data']; n = len(saved['joint_names']); names = list(saved['arm_names'])
        _check(names == ['left', 'right'], 'Expected two independently recorded arms')
        _check(data.shape[1] == 2 + 6*n and np.isfinite(data).all(), 'Invalid dual trace')
        _check(np.all(np.diff(data[:,0]) == 1), 'Dual clock is discontinuous')
        _check(len(data) == case['physics_steps'], 'Dual trace clock mismatch')
        active = str(saved['active_arm'])
        _check(active == case['active_arm'], 'Active-arm evidence mismatch')
        _check(case['arm_count'] == 2, 'Single-arm case in dual audit')
        per_arm = {}
        for i, name in enumerate(names):
            x = data[:,2+3*n*i:2+3*n*(i+1)]; limits = saved['joint_limits'][i]
            q = x[:,n:2*n]
            _check(np.all(x[:,:n] >= limits[:,0]-1e-5) and np.all(x[:,:n] <= limits[:,1]+1e-5), 'Dual action outside limits')
            _check(np.all(q >= limits[:,0]-.005) and np.all(q <= limits[:,1]+.005), 'Dual measured state outside limits')
            # Arm joints are the first six, followed by the gripper's joints.
            drift = float(np.max(abs(q[:, :6] - q[0, :6])))
            if name != active:
                _check(drift < .02, 'Idle arm did not maintain its home pose')
            per_arm[name] = {'max_arm_displacement_rad':drift,'idle_hold_verified':name != active,
                             'limits_verified':True}
        rows.append({'episode':identity,'active_arm':active,'arms':per_arm})
    _check(report['physical_execution_verified'], 'Primary task verification failed')
    _check({r['active_arm'] for r in rows if not r['episode'].endswith('_negative')} == {'left','right'}, 'Both arms were not exercised')
    result = {'passed':True,'mode':'offline recomputation of both measured articulations',
              'cases':rows,'scope':'independent reach and idle hold; not coordinated manipulation or contact certification'}
    (root / 'dual_audit.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(audit_dual(parser.parse_args().run),indent=2))
