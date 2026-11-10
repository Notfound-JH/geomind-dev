"""Locate transferred GEOMIND assets without depending on a creator's home.

No simulator imports, filesystem writes, or implicit downloads happen here.
Legacy prefixes are read compatibility aliases; new bundles use relative USD
references and an explicit GEOMIND_ASSET_BUNDLE_ROOT.
"""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BUNDLE_DIRECTORY = 'geomind-assets-v1'


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bundle_root() -> Path:
    override=os.environ.get('GEOMIND_ASSET_BUNDLE_ROOT')
    return Path(override).expanduser().resolve() if override else repo_root()/'external'/BUNDLE_DIRECTORY


def _location(variable: str, relative: str, fallback: Path) -> Path:
    override=os.environ.get(variable)
    if override:return Path(override).expanduser().resolve()
    candidate=bundle_root()/relative
    return candidate if candidate.is_dir() or os.environ.get('GEOMIND_ASSET_BUNDLE_ROOT') else fallback


def geomind_assets_root() -> Path:
    fallback=next((p for p in [repo_root()/'geomind_assets',repo_root().parent/'geomind_assets',
        Path('/home/projects/geomind_assets'),Path('/home/elm/projects/geomind_assets')] if p.is_dir()),repo_root()/'geomind_assets')
    return _location('GEOMIND_ASSETS_ROOT','geomind_assets',fallback)


def isaac_assets_root() -> Path:
    return _location('GEOMIND_ISAAC_ASSETS_ROOT','isaacsim_assets',Path('/home/elm/isaacsim_assets'))


def grscenes_home() -> Path:
    return _location('GEOMIND_GRSCENES_HOME','grscenes_home',repo_root().parent/'InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes')


def material_cache_root() -> Path:
    return _location('GEOMIND_MATERIAL_CACHE','material_cache',repo_root()/'outputs/scene_atlas_materials_v1/material_cache')


def repo_assets_root() -> Path:
    return _location('GEOMIND_REPO_ASSETS_ROOT','repo_assets/assets',repo_root()/'assets')


def scene_atlas_root() -> Path:
    override=os.environ.get('GEOMIND_SCENE_ATLAS_ROOT')
    if override:return Path(override).expanduser().resolve()
    # Galleries write their indexes/evidence. The transferred bundle is an
    # immutable input; configure creates this machine-local copy first.
    if bundle_root().is_dir() or os.environ.get('GEOMIND_ASSET_BUNDLE_ROOT'):
        return repo_root()/'.geomind_local/scene_atlas'
    return repo_root()/'scene_atlas'


def material_search_paths() -> list[Path]:
    roots=[grscenes_home()/'Materials',grscenes_home(),material_cache_root()/'nvidia',
        isaac_assets_root()/'Assets/Isaac/5.1/NVIDIA/Materials',
        isaac_assets_root()/'Assets/Isaac/4.5/NVIDIA/Materials']
    return [p for p in roots if p.is_dir()]


def configure_material_search_path(environment=None) -> dict[str,str]:
    target=os.environ if environment is None else environment
    existing=[x for x in target.get('MDL_SYSTEM_PATH','').split(os.pathsep) if x]
    target['MDL_SYSTEM_PATH']=os.pathsep.join(dict.fromkeys([*(str(p) for p in material_search_paths()),*existing]))
    return target


def resolve_asset_path(value: str | Path, *, require_exists: bool=False) -> str:
    """Resolve a bundle/repo URI or a historical local asset prefix.

    Unknown URLs and unrelated strings stay unchanged. Metadata or evidence
    paths outside these declared roots are never guessed or silently moved.
    """
    raw=str(value)
    if raw.startswith('bundle://'):
        relative=raw[len('bundle://'):]
        path=(bundle_root()/relative).resolve()
        if not path.is_relative_to(bundle_root().resolve()):raise ValueError('Asset bundle path escapes its root')
        result=str(path)
    elif raw.startswith('repo://'):
        path=(repo_root()/raw[len('repo://'):]).resolve()
        if not path.is_relative_to(repo_root().resolve()):raise ValueError('Repository path escapes its root')
        result=str(path)
    else:
        if raw.startswith('file:'):raw=raw[5:]
        aliases=[
            ('/home/elm/projects/geomind_lab/outputs/scene_atlas_materials_v1/material_cache',material_cache_root()),
            ('/home/elm/projects/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes',grscenes_home()),
            ('/home/elm/projects/geomind_lab/scene_atlas',scene_atlas_root()),
            ('/home/elm/projects/geomind_lab/assets',repo_assets_root()),
            ('/home/elm/projects/geomind_assets',geomind_assets_root()),
            ('/home/projects/geomind_assets',geomind_assets_root()),
            ('/home/elm/isaacsim_assets',isaac_assets_root()),
        ]
        result=raw
        for old,new in aliases:
            if raw==old or raw.startswith(old+'/'):
                result=str(new/raw[len(old):].lstrip('/'));break
        if result==raw and '/Assets/ArchVis/' in raw:
            candidate=isaac_assets_root()/'Assets/Isaac/4.5/NVIDIA/Assets/ArchVis'/raw.split('/Assets/ArchVis/',1)[1]
            if candidate.is_file():result=str(candidate)
        if result==raw and raw.startswith(('http://','https://','omniverse://')):
            url=urlparse(raw)
            for marker in ('/Assets/Isaac/4.5/','/Assets/Isaac/5.1/'):
                if marker in url.path:
                    candidate=isaac_assets_root()/('Assets/Isaac/'+marker.split('/')[3]+'/'+url.path.split(marker,1)[1])
                    if candidate.is_file():result=str(candidate)
            if '/Materials/' in url.path:
                relative=url.path.split('/Materials/',1)[1]
                for version in ('5.1','4.5'):
                    candidate=isaac_assets_root()/f'Assets/Isaac/{version}/NVIDIA/Materials'/relative
                    if candidate.is_file():result=str(candidate);break
    if require_exists and not Path(result).is_file():raise FileNotFoundError(f'Asset is unavailable: {result}')
    return result


def resolve_asset_paths(value: Any) -> Any:
    """Rebind explicit path strings in configuration data, preserving its shape."""
    if isinstance(value,dict):return {key:resolve_asset_paths(item) for key,item in value.items()}
    if isinstance(value,list):return [resolve_asset_paths(item) for item in value]
    if isinstance(value,tuple):return tuple(resolve_asset_paths(item) for item in value)
    if isinstance(value,str) and (value.startswith(('bundle://','repo://','file:/','/home/elm/','/home/projects/geomind_assets')) or '/Assets/ArchVis/' in value):
        return resolve_asset_path(value)
    return value


def asset_bundle_provenance() -> dict | None:
    """Bind a run to the transferred data manifest, without claiming all files were rehashed."""
    path=bundle_root()/'asset_bundle_manifest.json'
    if not path.is_file():return None
    data=path.read_bytes();manifest=json.loads(data)
    return {'bundle_root':str(bundle_root()),'manifest_path':str(path),
            'manifest_sha256':hashlib.sha256(data).hexdigest(),
            'bundle_name':manifest.get('bundle_name'),'manifest_files':len(manifest.get('files',[])),
            'all_asset_files_rehashed_by_this_function':False}
