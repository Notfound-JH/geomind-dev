"""Run the standard research pipeline with its existing physics video option."""
import json
from pathlib import Path
import subprocess
from geomind.tools.research_pipeline import main
original_run = subprocess.run
def run_with_video(command, *args, **kwargs):
    if isinstance(command, list) and 'geomind.tools.run_episode_physics' in command:
        command = [*command, '--video']
        output = Path(command[command.index('--output') + 1]).parent
        (output / 'physical_command.json').write_text(json.dumps(command, indent=2) + '\n')
    return original_run(command, *args, **kwargs)
subprocess.run = run_with_video
raise SystemExit(main(['--request', 'configs/research_requests/next_base_repeat_20261109.json']))
