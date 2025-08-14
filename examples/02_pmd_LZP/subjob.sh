#!/bin/bash
#=======================================================================
#  Script to be called from optzer and perfom pmd simulation
#  and extract RDF, ADF, and volume data.
#
#  Usage:
#    $ subjob.sh
#=======================================================================

#...copy filed required for pmd calculation
yes | cp ../in.pmd.* ../pmdini ./

#...cd to the directory and clean up
rm -f dump_* out.*

t0=`date +%s`

export OMP_NUM_THREADS=1

#...NpT MD
cp in.pmd.NpT in.pmd
pmd 2>&1 > out.pmd.NpT
# mpirun -np 1 ../../../pmd/pmd 2>&1 > out.pmd.NpT
head -n166 out.pmd.NpT
tail -n20 out.pmd.NpT
echo "NpT-MD done at" `date`
#...extract rdf, adf, vol and rename files
python ~/src/nap/nappy/rdf.py --fortran -d 0.05 -r 5.0 --gsmear=2 --skip=80 --specorder=Li,Zr,P,O --pairs=Li-O,Zr-O,P-O,Li-Li --out4fp -o data.pmd.rdf traj.extxyz 2>&1
python ~/src/nap/nappy/adf.py --fortran --gsmear=2 --triplets=Zr-O-O,P-O-O --skip=80 --out4fp -o data.pmd.adf traj.extxyz 2>&1
python ~/src/nap/nappy/vol_lat.py --skip=80 --out4fp --prefix data.pmd traj.extxyz 2>&1
t1=`date +%s`
etime=`expr $t1 - $t0`
echo "subjob.sh took" $etime "sec, done at" `date`
