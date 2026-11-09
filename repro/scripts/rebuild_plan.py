"""Rebuild the frozen holdout manifest from the report table and compare its SHA256."""
import hashlib, itertools, sys
sel = {('criteo','causalpfn'):(4096,'T'),('criteo','dopfn'):(1024,'TY'),('criteo','tabpfn_t'):(4096,'T'),
 ('hillstrom','causalpfn'):(1024,'R'),('hillstrom','dopfn'):(4096,'TY'),('hillstrom','tabpfn_t'):(2048,'R'),
 ('lzd','causalpfn'):(2048,'TY'),('lzd','dopfn'):(512,'TY'),('lzd','tabpfn_t'):(4096,'R'),
 ('orange','causalpfn'):(512,'T'),('orange','dopfn'):(1024,'R'),('orange','tabpfn_t'):(1024,'T'),
 ('retailhero','causalpfn'):(4096,'TY'),('retailhero','dopfn'):(1024,'T'),('retailhero','tabpfn_t'):(4096,'TY')}
name = {'R':'random','T':'balanced_treatment','TY':'balanced_treatment_outcome'}
want = 'dd8d349f336399e438caf7be4f2a250746b137b1dbba9c5d79cb7292a0f884e2'
D = ['criteo','hillstrom','lzd','orange','retailhero']; M = ['causalpfn','tabpfn_t','dopfn']
found = None
for dp in itertools.permutations(D):
    for mp in itertools.permutations(M):
        for mode in ('ds_first','model_first'):
            keys = [(d,m) for d in dp for m in mp] if mode == 'ds_first' else [(d,m) for m in mp for d in dp]
            rows = [(d,m,sel[d,m][0],name[sel[d,m][1]]) for d,m in keys] + [('criteo','dopfn',1024,'balanced_treatment')]
            txt = ''.join(f'{a} {b} {c} {e} {s}\n' for a,b,c,e in rows for s in (42,43,44))
            if hashlib.sha256(txt.encode()).hexdigest() == want:
                found = (dp, mp, mode, txt)
print('checked orderings; match:', bool(found))
if found:
    print(found[0], found[1], found[2]); open(sys.argv[1], 'w', encoding='utf8', newline='\n').write(found[3])
