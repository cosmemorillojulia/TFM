import json, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
print(f"#### NOTEBOOK: {path}  ({len(nb['cells'])} cells) ####")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        print(f"\n--- [{i}] MARKDOWN ---")
        print(src)
    elif c['cell_type'] == 'code':
        print(f"\n--- [{i}] CODE ---")
        print(src)
