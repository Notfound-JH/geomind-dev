"""Persist the asset-only filtering used before the normalized bulk build.

MDL-declared thumbnail resources are added back explicitly by
final_asset_stage_corrections.py; arbitrary thumbnail caches stay excluded.
"""
from pathlib import Path
import argparse,collections,json

def curate(data):
 blocked={'.git','.aws','.ssh','__pycache__','.cache','credentials','node_modules','.thumbs'}
 removed=[x['archive_path'] for x in data['files'] if blocked&set(Path(x['archive_path']).parts)
          or Path(x['archive_path']).name.startswith('.env') or Path(x['archive_path']).suffix=='.pyc']
 data['files']=[x for x in data['files'] if x['archive_path'] not in set(removed)]
 data['excluded_non_asset_paths']=sorted(set(data.get('excluded_non_asset_paths',[]))|set(removed))
 data['total_files']=len(data['files']);data['total_bytes']=sum(x['bytes'] for x in data['files'])
 summary=collections.defaultdict(lambda:{'files':0,'bytes':0})
 for x in data['files']:
  key=x['archive_path'].split('/')[0];summary[key]['files']+=1;summary[key]['bytes']+=x['bytes']
 data['summary']=dict(summary);return data

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('inventory',type=Path);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.write_text(json.dumps(curate(json.loads(a.inventory.read_text())),ensure_ascii=False,indent=2)+'\n')
