#!/bin/bash
#module load conda
#module load gcc
export DEVITO_ARCH='gcc'
export DEVITO_OPENMP=1

#export DEVITO_AUTOTUNING=aggressive
#export DEVITO_DLE_OPTIONS=blockinner:True

export DEVITO_AUTOTUNING=none
export DEVITO_DLE_OPTIONS="blockinner:True;blockshape:64, 64, 64"
#export DEVITO_DLE_OPTIONS="blockinner:True;blockshape:16,16,16"

export DEVITO_LOGGING=DEBUG
python tools.py
# MZ: replace conda env file
