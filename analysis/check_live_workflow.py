"""Check UI links and named widget schemas directly against this ComfyUI."""
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
with urlopen('http://127.0.0.1:8188/object_info', timeout=30) as r:
    catalog = json.load(r)
wf = json.loads((root/'LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO+ReActor-SeamlessChain-v2-TEST19s.json').read_text(encoding='utf-8'))
nodes = {n['id']: n for n in wf['nodes']}
links = {link[0]: link for link in wf['links']}
errors = []
for n in nodes.values():
    if n.get('mode', 0) != 0:
        continue
    schema = catalog.get(n['type'])
    if schema is None:  # front-end-only Set/Get/notes are not server classes
        if n['type'] not in {'SetNode','GetNode','Note','MarkdownNote','Reroute'}:
            errors.append(f"Unknown active node: {n['id']} {n['type']}")
        continue
    for port in n.get('inputs', []):
        link_id = port.get('link')
        if link_id is None:
            continue
        link = links.get(link_id)
        if link is None or link[1] not in nodes:
            errors.append(f"Missing link {link_id}")
            continue
        source = nodes[link[1]]
        source_schema = catalog.get(source['type'])
        if source_schema and link[2] >= len(source_schema['output']):
            errors.append(f"Invalid output {source['id']}[{link[2]}]")
    inputs = schema['input'].get('required', {}) | schema['input'].get('optional', {})
    linked = {p['name'] for p in n.get('inputs',[]) if p.get('link') is not None}
    for name,value in n.get('widgets_values_named',{}).items():
        if name in linked or name not in inputs:
            continue
        kind = inputs[name][0]
        if kind == 'STRING' and not isinstance(value,str):
            errors.append(f"{n['id']}.{name}: expected string")
state = next(n for n in nodes.values() if n['type']=='LTXChainState')
assert state['widgets_values'][-1] == state['widgets_values_named']['continuity_lead_seconds'] == 1
report = {'state_outputs':len(catalog['LTXChainState']['output']),
          'continuity_setting_loaded':'continuity_lead_seconds' in catalog['LTXChainState']['input']['optional'],
          'errors':errors,
          'scope':'UI link ranges and named widget string types; not a generation or full server prompt validation'}
(root/'analysis/live-workflow-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
raise SystemExit(bool(errors))
