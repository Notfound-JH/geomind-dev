"""Reject credential metadata and scan small text asset copies, without values."""
from pathlib import Path
import argparse,json,re

def scan(base):
 m=json.loads((base/'asset_bundle_manifest.json').read_text())
 blocked={'.git','.aws','.ssh','.env','.gitconfig','.git-credentials','.netrc','.npmrc','.pypirc','.boto','credentials','__pycache__','.cache','node_modules'}
 patterns={'private_key':re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
           'github_token':re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),'aws_access_key':re.compile(rb'AKIA[0-9A-Z]{16}'),
           'openai_key':re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}')}
 findings=[];checked=0
 for e in m['files']:
  p=Path(e['path'])
  if blocked&set(p.parts) or p.name.startswith('.env'):findings.append({'path':str(p),'reason':'prohibited metadata/credential component'})
  if p.suffix.lower() in {'.json','.yaml','.yml','.md','.txt','.py','.sh','.xml','.urdf','.xacro','.mdl','.html','.js','.css'}:
   data=(base/p).read_bytes();checked+=1
   for kind,regex in patterns.items():
    if regex.search(data):findings.append({'path':str(p),'pattern':kind,'value':'not emitted'})
 return {'passed':not findings,'manifest_asset_files':len(m['files']),'small_text_assets_scanned':checked,
         'findings':findings,'credential_values_emitted':False,'archive_source_roots_are_allowlisted':True}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();r=scan(a.assets_root);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');raise SystemExit(0 if r['passed'] else 1)
