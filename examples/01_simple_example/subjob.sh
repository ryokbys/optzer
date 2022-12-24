#!/bin/bash
#=======================================================================
#  Script to be called from optzer and perfom subjobs.
#
#  Usage:
#    $ ./subjob.sh
#=======================================================================


t0=`date +%s`

export OMP_NUM_THREADS=1

python ../simple_func.py --param-file in.params.simple

t1=`date +%s`
etime=`expr $t1 - $t0`
echo "subjob.sh took" $etime "sec, done at" `date`
