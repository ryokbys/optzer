# optzer/examples/02_pmd_LZP

This example shows how to use `optzer` to optimize the parameters of interatomic potential for Li-Zr-P-O system with a parallel molecular dynamics program, [pmd](https://github.com/ryokbys/nap). The potential forms are screened Coulomb, Morse, and angular and the target quantities are RDF, ADF and equilibrium volume, see the Ref. [1] for details. The files required to perform `optzer` are:

- `in.optzer` -- information about `optzer` setting
- `in.vars.optzer` -- initial values of potential parameters and search ranges of the parameters
- `in.params.{Coulomb|Morse|angular}` -- parameter file for Coulomb, Morse, and angular potentials
- `data.ref.xxx` -- target quantities (RDF, ADF and volume)
- `subjob.sh` -- shell-script that describe what to compute target quantities with the current parameters
- `in.pmd.NpT`, `pmdini`, and something -- files need to perform MD simulations described in `subjob.sh`

To see command-line options,
```bash
$ optzer -h
```

To perform `optzer` using four threads,
```bash
$ optzer --nproc=5 --random-seed=42 | tee out.optzer
```

Since the `optzer` uses random values to generate trial parameters and the random values would be different every time it is executed, the output obtained will not be identical to that in `out.optzer.REF`. But it will be OK if `tail out.optzer` looks like the following,
```bash
$ tail out.optzer
 step,time,best_iid,best_loss,vars=      0     13.8     4   2.4776  1.193  1.262  0.891  0.829  1.142  2.088  1.724  2.495  1.910  1.991  4.259  2.099  1.462  2.040  0.047  1.370
   iid,losses=        8     6.3359     0.9942    25.8651    33.19519
   iid,losses=        7     3.6779     0.8047    16.8127    21.29532
   iid,losses=        6     2.0658     0.7577     7.6172    10.44070
   iid,losses=        5     0.5379     0.4912     2.3523     3.38133
   iid,losses=        9     0.4758     0.0995     0.1739     0.74922
 step,time,best_iid,best_loss,vars=      1     29.8     9   0.7492  1.054  1.188  0.824  0.808  1.281  2.167  1.839  2.409  1.952  1.916  4.250  2.065  1.543  2.320 -0.220  2.910

 optzer finished since it exceeds the max interation.
 Elapsed time = 29.8 sec.
```
And when you use `optzer`, you had better use more processes than 4 like in this case to efficiently run the program.

## References

1. R. Kobayashi, Y. Miyaji, K. Nakano, M. Nakayama, “High-throughput production of force-fields for solid-state electrolyte materials”, APL Materials 8, 081111 (2020). [link](https://aip.scitation.org/doi/10.1063/5.0015373).
