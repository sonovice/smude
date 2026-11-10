#!/usr/bin/env python3
"""Seed/restore an isolated UI fixture on a named simulator; never physical iOS."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

mode, simulator, output = sys.argv[1:]
output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
subprocess.run(['xcrun','simctl','terminate',simulator,'com.gridnotelabs.smude.design'],capture_output=True)
container=Path(subprocess.check_output(['xcrun','simctl','get_app_container',simulator,'com.gridnotelabs.smude.design','data'],text=True).strip())
root=container/'Library/Application Support'
paths=['SmudeProjects/library.json','SmudeProjects/library.previous.json','nsh/com.gridnotelabs.smude.design/storage.json']
backup=output/f'backup-{simulator}'; backup.mkdir(exist_ok=True)
if mode=='restore':
    for i,path in enumerate(paths):
        target=root/path; saved=backup/f'{i}.json'
        if saved.exists(): target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(saved,target)
        elif (backup/'complete').exists():target.unlink(missing_ok=True)
    print('Restored simulator library and preferences');sys.exit(0)
if not (backup/'complete').exists():
    for i,path in enumerate(paths):
        if (root/path).exists():shutil.copy2(root/path,backup/f'{i}.json')
    (backup/'complete').write_text('Original simulator files backed up; absence is significant.')
repo=Path(__file__).resolve().parents[4]
audit=repo/'outputs/research/mobile-ux-audit-20261110'
lib=json.loads((audit/'fixture-ios.json').read_text())
renders=root/'SmudeProjects/renders';renders.mkdir(parents=True,exist_ok=True)
for name,source in [('ux27-source.png','source-fixture.png'),('ux27-score.png','score-fixture.png')]:shutil.copy2(audit/source,renders/name)
for collection in lib['projects']:
    for page in collection['pages']:
        if page.get('source'): page['source']='renders/ux27-source.png'
        if page.get('output'): page['output']='renders/ux27-score.png'
        page['review']=False
    if collection['id']==27000:collection['pages']=collection['pages'][:6]
# Isolated no-image and retained-preview cases; both Failed prevent a background job.
template=copy.deepcopy(lib['projects'][0]['pages'][0]);template.update(id=28101,capture=28101,status='Failed',source=None,output=None)
lib['projects'].append(dict(id=28100,title='Missing image',pages=[template]))
template=copy.deepcopy(lib['projects'][0]['pages'][0]);template.update(id=28201,capture=28201,status='Failed')
lib['projects'].append(dict(id=28200,title='Previous preview',pages=[template]))
(root/paths[0]).parent.mkdir(parents=True,exist_ok=True);(root/paths[0]).write_text(json.dumps(lib))
prefs=root/paths[2];prefs.parent.mkdir(parents=True,exist_ok=True)
d=json.loads(prefs.read_text())if prefs.exists()else{};d['smude-project-view']='list';prefs.write_text(json.dumps(d))
print('Prepared simulator UI fixture',simulator)
