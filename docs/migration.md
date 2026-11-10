> 历史记录：本文描述 2026-09 的 Python 3.10 / Isaac Sim 4.5 迁移。当前 27 类任务交付使用 Python 3.11 / Isaac Sim 5.1，请按 [installation_zh.md](installation_zh.md) 安装；本文件保留原版本事实，不作为当前安装入口。

# Server migration

## Supported development workflow

Use a Git checkout and an **editable installation**, not a standalone wheel.
Task assets, shell runners and some scene loaders resolve paths relative to the
checkout. A wheel alone is not a complete simulation distribution.

The default environment name is `isaac_curobo`. The inspected environment contains:

| Component | Observed version |
| --- | --- |
| Python | 3.10.19 |
| Isaac Sim | 4.5.0.0 |
| IsaacLab distribution | 0.51.1 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.4 |
| Gymnasium | 1.2.1 |
| h5py | 3.15.1 |
| pytest | 9.0.2 |

These are an observed baseline, **not a claim of GPU compatibility on every
server**. In particular, choose a driver compatible with the GPU, Isaac Sim and
CUDA-enabled PyTorch. Do not install Isaac Sim 5.x into this Python 3.10 baseline.
The Conda YAML is a bootstrap recipe, not a complete solver lock.

## CPU development and tests

```bash
git clone https://github.com/Notfound-JH/GeoMIND.git
cd GeoMIND
# Check out the migration branch/commit supplied with the handoff.
conda create -n isaac_curobo python=3.10.19 pip -y
conda activate isaac_curobo
python -m pip install -e '.[dev]'
python scripts/check_development_setup.py
python -m pytest tests -q
python -m geomind.runners.run_task --help
```

Run commands from the repository root. The unit tests do not replace a GPU
simulation smoke test. Install `.[analysis]` for report/plot tools and `.[viewer]`
for the optional Qt image viewer. Analysis scripts require experiment data,
which is intentionally not part of this code bundle.

## GPU simulation setup

1. Install a compatible NVIDIA driver and validate it with `nvidia-smi`.
2. Create/activate `isaac_curobo`; the optional broader bootstrap recipe is
   `conda env create -f environment.isaac_curobo.yml`.
3. Install the Isaac Sim 4.5 distribution following its official installation
   requirements. An environment name alone does not install the simulator.
4. Clone the pinned IsaacLab revision into `IsaacLab/` inside this checkout.
   Apply the local compatibility patches in
   [migration/isaaclab/README.md](../migration/isaaclab/README.md).
5. Install a compatible CUDA-enabled PyTorch, then the IsaacLab extensions with
   `cd IsaacLab && ./isaaclab.sh -i && cd ..`. Check the resulting versions:
   installer scripts may change dependencies.
6. Install cuRobo from its separately pinned source if using motion planners.
   See the patch README for the local source provenance and limitations.
7. Reinstall this project with `python -m pip install -e '.[dev,analysis]'`.
8. Configure external assets and run an IsaacLab smoke test before collecting data.

The old environment's IsaacLab editable metadata pointed outside this checkout.
On the new server, reinstall extensions from **this checkout's `IsaacLab/`**.
Check imported paths rather than trusting a copied Conda environment:

```bash
python -m pip show isaacsim isaaclab torch gymnasium
python -c "import geomind, geomind_lab; print(geomind.__file__); print(geomind_lab.__file__)"
```

Do not copy an existing virtual environment or its editable `.pth` files.

## Assets and local settings

The Git repository includes JSON annotations and asset catalog metadata, not
the external USD/MDL assets, model weights or training/evaluation datasets.
Transfer those separately, subject to their licenses, then configure:

```bash
export GEOMIND_ASSETS_ROOT=/absolute/path/to/geomind_assets
export GEOMIND_LAB_ASSET_CATALOG_PATH="$PWD/assets/scene_profiles/assets_catalog_with_metadata.json"
# If a scene uses external MDL materials, set their actual directory:
export MDL_SYSTEM_PATH=/absolute/path/to/materials
export REMOTE_POLICY_URL=http://127.0.0.1:8000/policy
export GEOMIND_MODEL_POLICY_HOOK=geomind_client:policy
```

Do not commit credentials or machine-specific `.env` files. Shell launchers
support `CONDA_SH` for nonstandard Conda locations. Legacy scene-construction
utilities and historical analysis scripts may still contain workstation paths;
they are not the portable evaluation entry point. Their directory is currently
named `scene_construction ` (with a trailing space).

## Validation gates

- A fresh checkout installs with `pip install -e '.[dev]'`.
- CPU tests run without depending on the original workstation.
- Registry/config loading and a mock rollout succeed.
- On a GPU server, Isaac Sim starts headlessly and loads the required assets.
- A representative real episode produces a valid result and observation.
- Remote policy runs actually load their intended hook; missing hooks can fall
  back to oracle in the existing runtime.

Run `python scripts/check_development_setup.py --simulation` to check installed
distributions and asset-root prerequisites without starting the simulator.
This preflight does not validate CUDA execution or every asset reference.

### Existing test baseline (2026-09-24)

In the original `isaac_curobo` environment, `python -m pytest tests -q` reports
120 passed, 7 failed and 20 subtests passed. Six failures in
`test_benchmark_manifest_utils.py` expect layout sets without the newly added VLA
variants; one in `test_task_metadata_registry.py` expects a Chinese instruction
where the current annotation is English. Packaging does not silently update
these assertions or change task semantics. A clean migration should reproduce
this baseline; additional failures require investigation.

An exported index snapshot was also installed in a separate editable virtual
environment and reproduced the same results. All four project imports resolved
to the exported checkout, and the development preflight passed. Because network
access was unavailable, this validation reused the original environment's
third-party packages (`--system-site-packages`); it is not a fresh dependency
download or a clean GPU installation test. A source distribution was built
successfully from that snapshot.

The mock `run_task` smoke test completed and wrote a result with
`metric_valid=true`, but `success=false` (the existing early-termination/dwell
behavior). This proves the CPU execution path runs, not that the task succeeds.

Known pre-existing limitation: `run_env_policy` passes `ablation_id` and `pair_id`
to a runtime that has not yet accepted those arguments. Do not use that entry
point as proof of successful migration until the interface is fixed. Use
`run_task` for the basic mock smoke test. Historical benchmark metrics and
history/memory labels have not been corrected by this packaging work.

## What is intentionally excluded

IsaacLab and cuRobo vendor trees, Conda/virtual environments, model checkpoints,
external assets, trajectories, videos, `run_eval_*`, `analysis_outputs`, logs,
paper result artifacts and editor settings. Existing local copies remain on disk.
Preserve the exact Git commit, external asset revision and model revision for
each experiment; restoring source code does not reproduce omitted experiment data.
