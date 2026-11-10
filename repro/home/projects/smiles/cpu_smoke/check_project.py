"""Check the project structure and basic reproducibility conditions."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
import numpy as np
from cpu_smoke import synthetic_data, sample_and_split, preprocess_train_only, choose_context, FULL_PROTOCOL

def main():
    xa, ta, ya = synthetic_data(2400, 42)
    xb, tb, yb = synthetic_data(2400, 42)
    assert np.array_equal(xa.to_numpy(), xb.to_numpy())
    assert np.array_equal(ta, tb) and np.array_equal(ya, yb)
    tr, ev = sample_and_split(xa, ta, ya, 2400, 320)
    assert len(ev) == 320 and not set(tr).intersection(ev)
    xtr, xev = preprocess_train_only(xa, tr, ev)
    cx, ct, cy = choose_context(xtr, ta[tr], ya[tr], 256, 42)
    assert cx.shape == (256, xa.shape[1]) and xev.shape[0] == 320
    assert np.unique(ct).tolist() == [0, 1]
    assert FULL_PROTOCOL['context_sizes'] == [512,1024,2048,4096]
    print('PASS: deterministic data, split isolation, CPU context, fixed research protocol')

if __name__ == '__main__':
    main()
