#!/bin/bash
if [[ -n ${OMPI_COMM_WORLD_RANK+z} ]]; then
  # mpich
  export MPI_RANK=${OMPI_COMM_WORLD_RANK}
elif [[ -n ${MV2_COMM_WORLD_RANK+z} ]]; then
  # ompi
  export MPI_RANK=${MV2_COMM_WORLD_RANK}
elif [[ -n "${SLURM_PROCID+z}" ]]; then
  export MPI_RANK="$SLURM_PROCID"
elif [[ -n "${PMI_RANK+z}" ]]; then
  export MPI_RANK="$PMI_RANK"
fi
args="$*"
pid="$$"
outfile="results_${MPI_RANK}.csv"
if [ "$MPI_RANK" -lt 16 ]
then  
  eval "rocprof -i inputs-fp16.txt -o profs22b/${outfile} $*"
else 
  $*
fi
