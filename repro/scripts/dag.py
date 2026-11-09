"""Build the run order of the repo experiments from the local folders each runner reads and writes."""
import re, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')
files = subprocess.run(['git', 'ls-tree', '-r', 'HEAD', '--name-only'], capture_output=True, text=True).stdout.split('\n')
tools = [f for f in files if re.match(r'tools/run_.*\.py$', f)]
own, needs, ext = {}, {}, {}
for f in tools:
    s = subprocess.run(['git', 'show', 'HEAD:' + f], capture_output=True, text=True, encoding='utf8').stdout
    name = f[len('tools/run_'):-3]
    found = set(re.findall(r'([a-z0-9_]+_results)', s))
    ident = name if re.search(r'_v\d+$', name) else name
    base = re.sub(r'_v001$', '', name)
    mine = {x for x in found if x.startswith(base) or x.startswith(name)}
    needs[name] = found - mine; own[name] = mine
    ext[name] = sorted(set(re.findall(r'external/([a-z0-9_]+)', s)))
    needs[name] |= {'data' if 'data_A_cleaned' in s else ''} - {''}
    imp = set(re.findall(r'import (?:tools\.)?run_([a-z0-9_]+)', s)) | set(re.findall(r'load_module\("run_([a-z0-9_]+)"', s))
    needs[name] |= {'imp:' + i for i in imp}
writer = {}
for k, v in own.items():
    for x in v: writer.setdefault(x, []).append(k)
print(f'{len(tools)} runners')
for k in sorted(needs):
    deps = sorted(d for d in needs[k] if d != 'data')
    print(f'{k:18s} ext={",".join(ext[k]) or "-":22s} needs={" ".join(deps) or "-"}'[:125])
