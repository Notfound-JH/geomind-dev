#!/usr/bin/env python3
"""Bind an extracted GEOMIND asset bundle to a checkout; no simulator needed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def _inside(root,relative):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
        raise ValueError('Manifest contains an invalid relative asset path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Manifest asset path escapes the bundle')
    return path


def _rebind(value,bundle,repo):
    if isinstance(value,dict):return {key:_rebind(item,bundle,repo) for key,item in value.items()}
    if isinstance(value,list):return [_rebind(item,bundle,repo) for item in value]
    if isinstance(value,str) and value.startswith('bundle://'):
        return str(_inside(bundle,value[len('bundle://'):]))
    if isinstance(value,str) and value.startswith('repo://'):
        return str(_inside(repo,value[len('repo://'):]))
    return value


def configure(bundle,repo,*,verify_all=False,isaac_python=None):
    bundle=Path(bundle).expanduser().resolve();repo=Path(repo).expanduser().resolve()
    manifest_path=bundle/'asset_bundle_manifest.json'
    if not manifest_path.is_file():raise FileNotFoundError(f'Extract the asset ZIP first: {manifest_path}')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('bundle_name')!='geomind-assets-v1':
        raise ValueError('Unsupported asset bundle manifest/version')
    files=manifest.get('files')
    if not isinstance(files,list) or not files:raise ValueError('Asset bundle has no manifest file entries')
    names=[entry.get('path') for entry in files]
    if len(names)!=len(set(names)):raise ValueError('Duplicate path in asset bundle manifest')
    failures=[]
    for entry in files:
        path=_inside(bundle,entry['path'])
        if not path.is_file() or path.stat().st_size!=entry['size_bytes']:
            failures.append(entry['path'])
        elif verify_all and _sha(path)!=entry['sha256']:
            failures.append(entry['path'])
    if failures:raise ValueError('Incomplete or changed asset extraction: '+', '.join(failures[:8]))
    expected={
        'GEOMIND_ASSET_BUNDLE_ROOT':bundle,
        'GEOMIND_REPO_ROOT':repo,
        'GEOMIND_ASSETS_ROOT':bundle/'geomind_assets',
        'GEOMIND_LAB_ASSETS_ROOT':bundle/'geomind_assets',
        'GEOMIND_ISAAC_ASSETS_ROOT':bundle/'isaacsim_assets',
        'GEOMIND_GRSCENES_HOME':bundle/'grscenes_home',
        'GEOMIND_MATERIAL_CACHE':bundle/'material_cache',
        'GEOMIND_REPO_ASSETS_ROOT':bundle/'repo_assets/assets',
        'GEOMIND_SCENE_ATLAS_ROOT':repo/'.geomind_local/scene_atlas',
    }
    for key,path in expected.items():
        if key=='GEOMIND_SCENE_ATLAS_ROOT':continue
        if not path.is_dir():raise FileNotFoundError(f'Missing {key}: {path}')
    local=repo/'.geomind_local';local.mkdir(parents=True,exist_ok=True)
    atlas_source=bundle/'repo_assets/scene_atlas'
    if not atlas_source.is_dir():raise FileNotFoundError('Missing immutable scene atlas input: '+str(atlas_source))
    atlas_local=expected['GEOMIND_SCENE_ATLAS_ROOT']
    origin=atlas_local/'.geomind_copy_origin.json'
    manifest_hash=_sha(manifest_path)
    if atlas_local.exists():
        if not origin.is_file() or json.loads(origin.read_text()).get('asset_bundle_manifest_sha256')!=manifest_hash:
            raise ValueError('Existing local scene atlas has another origin; preserved without overwrite: '+str(atlas_local))
    else:
        shutil.copytree(atlas_source,atlas_local)
        origin.write_text(json.dumps({'asset_bundle_manifest_sha256':manifest_hash,'immutable_source':'repo_assets/scene_atlas'},indent=2)+'\n')
    catalog_source=expected['GEOMIND_REPO_ASSETS_ROOT']/'scene_profiles/assets_catalog_with_metadata.json'
    catalog=_rebind(json.loads(catalog_source.read_text()),bundle,repo)
    checked=0
    for group,entries in catalog.get('assets',{}).items():
        for entry in entries:
            raw=entry.get('usd_path')
            if raw is not None:
                path=Path(raw)
                if not path.is_file():raise FileNotFoundError(f'Catalog asset {entry.get("name")}: {raw}')
                if not path.resolve().is_relative_to(bundle):raise ValueError('Catalog refers outside the extracted asset bundle')
                checked+=1
    catalog_path=local/'assets_catalog.json'
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    expected['GEOMIND_LAB_ASSET_CATALOG_PATH']=catalog_path
    links=[]
    rows={entry['path']:entry for entry in files}
    def merge(source,destination):
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(source,target_is_directory=source.is_dir())
            links.append({'path':str(destination),'status':'created_missing_link'});return
        if destination.resolve()==source.resolve():
            return
        if source.is_dir() and destination.is_dir():
            for child in source.iterdir():merge(child,destination/child.name)
            return
        if source.is_file() and destination.is_file():
            record=rows[source.relative_to(bundle).as_posix()]
            allowed={record['sha256'],record.get('source_sha256')}
            if _sha(destination) in allowed:
                links.append({'path':str(destination),'status':'preserved_matching_metadata_or_asset'});return
        raise ValueError('Existing checkout asset differs from bundle; preserved without overwrite: '+str(destination))
    merge(expected['GEOMIND_REPO_ASSETS_ROOT'],repo/'assets')
    destination=repo/'scene_atlas'
    if destination.is_symlink() and destination.resolve()==atlas_source.resolve():
        destination.unlink()  # Replace only a known old bundle link, never data.
    if not destination.exists() and not destination.is_symlink():
        destination.symlink_to(atlas_local,target_is_directory=True)
        links.append({'path':str(destination),'status':'linked_machine_local_writable_atlas'})
    elif destination.resolve()!=atlas_local.resolve():
        links.append({'path':str(destination),'status':'preserved_existing_non_bundle_atlas; environment selects local copy'})
    if isaac_python:
        executable=Path(isaac_python).expanduser().resolve()
        if not executable.is_file():raise FileNotFoundError(f'Isaac Python does not exist: {executable}')
        expected['GEOMIND_ISAAC_PYTHON']=executable
    mdl=[expected['GEOMIND_GRSCENES_HOME']/'Materials',expected['GEOMIND_GRSCENES_HOME'],
         expected['GEOMIND_MATERIAL_CACHE']/'nvidia',
         expected['GEOMIND_ISAAC_ASSETS_ROOT']/'Assets/Isaac/5.1/NVIDIA/Materials',
         expected['GEOMIND_ISAAC_ASSETS_ROOT']/'Assets/Isaac/4.5/NVIDIA/Materials']
    mdl=[str(path) for path in mdl if path.is_dir()]
    env=['# Generated locally by scripts/configure_portable_assets.py; do not commit this file.']
    env.extend('export '+key+'='+shlex.quote(str(value)) for key,value in expected.items())
    env.append('export MDL_SYSTEM_PATH='+shlex.quote(os.pathsep.join(mdl))+':"${MDL_SYSTEM_PATH:-}"')
    env.append('export PYTHONPATH='+shlex.quote(str(repo))+':"${PYTHONPATH:-}"')
    (repo/'.env.geomind').write_text('\n'.join(env)+'\n')
    receipt={'bundle_name':manifest['bundle_name'],'asset_bundle_manifest_sha256':_sha(manifest_path),
             'asset_bundle_root':str(bundle),'repo_root':str(repo),'catalog_assets_checked':checked,
             'manifest_files_checked':len(files),'full_file_sha256_verified':verify_all,'links':links,
             'environment_file':str(repo/'.env.geomind'),'configuration_only':True,
             'physical_execution_performed':False,'scene_atlas_is_machine_local_writable_copy':True,'sdk_mdl_dependencies':manifest.get('sdk_provided',[])}
    (local/'asset_configuration_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--assets-root',type=Path,default=None)
    parser.add_argument('--verify-all',action='store_true',help='Hash all extracted assets against the ZIP manifest')
    parser.add_argument('--isaac-python',type=Path,help='Optional selected IsaacSim5.1 Python interpreter')
    args=parser.parse_args()
    bundle=args.assets_root or Path(os.environ.get('GEOMIND_ASSET_BUNDLE_ROOT',args.repo_root/'external/geomind-assets-v1'))
    receipt=configure(bundle,args.repo_root,verify_all=args.verify_all,isaac_python=args.isaac_python)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    print('Next: source '+shlex.quote(str(args.repo_root.resolve()/'.env.geomind')))
    return 0

if __name__=='__main__':raise SystemExit(main())
