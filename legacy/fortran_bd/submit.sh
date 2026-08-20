#!/bin/sh
###################################################
#PBS -N mc_run1
#PBS -l nodes=1:ppn=1 -l walltime=100:00:00 -l mem=800MB
#PBS -q joe-6
#PBS -j oe
###################################################

WORKDIR=${PBS_O_WORKDIR}
cd ${WORKDIR}
echo "Nodes chosen are:"
cat $PBS_NODEFILE
NCPUS=`wc -l < $PBS_NODEFILE`

./bdpd>out.txt
