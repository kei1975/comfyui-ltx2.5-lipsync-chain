import copy
import json
from pathlib import Path
import uuid

root = Path(__file__).resolve().parents[1]
source = Path('E:/download/ComfyUI-Easy-Install-Windows/ComfyUI-Easy-Install/ComfyUI/user/default/workflows/LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO+ReActor-SeamlessChain.json')
wf = json.loads(source.read_text(encoding='utf-8'))
wf['id'] = str(uuid.uuid4())
state = next(n for n in wf['nodes'] if n['type'] == 'LTXChainState')
state['inputs'].append({'name':'continuity_lead_seconds', 'type':'INT',
                        'widget':{'name':'continuity_lead_seconds'}, 'link':None})
state['widgets_values'].append(1)
state['widgets_values_named']['continuity_lead_seconds'] = 1
state['title'] = 'LTX Chain: State — Continuity v2 (1s motion context)'
base = 'LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO+ReActor-SeamlessChain-v2'
(root/(base+'.json')).write_text(json.dumps(wf,ensure_ascii=False,indent=2),encoding='utf-8')
test = copy.deepcopy(wf)
test['id'] = str(uuid.uuid4())
state = next(n for n in test['nodes'] if n['type']=='LTXChainState')
for name, index, value in [('length_mode',2,'seconds'), ('length_seconds',3,19)]:
    state['widgets_values_named'][name] = value
    state['widgets_values'][index] = value
step = next(n for n in test['nodes'] if n['type']=='LTXChainStep')
step['widgets_values'][0] = 'continuity-test-19s'
step['widgets_values_named']['final_name'] = 'continuity-test-19s'
(root/(base+'-TEST19s.json')).write_text(json.dumps(test,ensure_ascii=False,indent=2),encoding='utf-8')
print('Created full-length and 19-second test workflows; original unchanged.')
