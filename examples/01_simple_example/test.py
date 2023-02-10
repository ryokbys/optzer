
def read_output(fname='out.optzer'):
    with open(fname,'r') as f:
        lines = f.readlines()
    return lines

out = read_output('out.optzer')
outref = read_output('out.optzer.REF')

bestlossref = 1.0e+30
for l in reversed(outref):
    dat = l.split()
    if 'step,time' in l:
        bestiidref = int(dat[3])
        bestlossref = float(dat[4])
        break

for l in out:
    assert 'nan' not in l or 'NaN' not in l

    dat = l.split()
    if 'step,time,' in l:
        bestiid = int(dat[3])
        bestloss = float(dat[4])

# assert bestiid == bestiidref
assert abs(bestloss-bestlossref) < 0.01

print('pass')
