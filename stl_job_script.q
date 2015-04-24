#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l mem=64GB
#PBS -l walltime=1:00:00
#PBS -M djh389@nyu.edu
#PBS -l nodes=1:ppn=1:gpus=1:titan 
 
module load amber/mvapich2/intel/14.03
 
# make a unique directory (by using the PBS job id in its name)
# The /.*/ shortens the job id to the part before the first ".", see
# the "Parameter expansion" section of "man bash" for more
RUNDIR=${SCRATCH}/stl_job.${PBS_JOBID/.*/}
mkdir $RUNDIR
 
# copy the input data from where we placed in in $HOME to the $RUNDIR
cd $RUNDIR
cp $PBS_O_WORKDIR/*.lua .
 
# the command to start the simulation is:
# sander -O -i mdin -o amoeba_jac.mdout
# we can get timing and resource use information by running a command or program via "time":
# /usr/bin/time -v will report the time and memory used afterwards
/scratch/courses/DSGA1008/bin/th doall.lua -dataSource bin -batchSize 128 -dataDir /scratch/courses/DSGA1008/A2

# /usr/bin/time -v sander -O -i mdin -o amoeba_jac.mdout
echo "testing"
