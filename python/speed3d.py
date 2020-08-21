'''
    3D FFT tester for Python Interface
    -- heFFTe --
    Univ. of Tennessee, Knoxville
'''

import sys, math
import cmath 
import numpy as np
from mpi4py import MPI
from heffte import *

#? syntax 

# * Allocate and initialize data 
def make_data():
    global work, work2
    work = np.zeros(fftsize, np.float32)
    for i in np.arange(fftsize):
        work[i] = i+1

# =============
#* Main program 
# =============
# MPI setup
fft_comm = MPI.COMM_WORLD
me = fft_comm.rank
nprocs = fft_comm.size

# parse command-line args

# define global parameters
global fft, fftsize

fftsize = 8
full_low = np.array([0, 0, 0], dtype=np.int32)
full_high = np.array([1, 1, 1], dtype=np.int32)
order = np.array([0, 1, 2], dtype=np.int32)

fft = fft3d(fft_comm, heffte_backend['fftw'], full_low, full_high, order, full_low, full_high, order)

# Initialize data
make_data()

print("Initial data:")
print(work)

fft_comm.Barrier()
time1 = MPI.Wtime()

fft.forward(work, work2, heffte_scale['full'])

fft_comm.Barrier()
time2 = MPI.Wtime()
t_exec = time2 - time1
Gflops = 5*fftsize*math.log(fftsize) / t_exec / 1E9

print("--------------------------")
print(f"Execution time = {t_exec:.2g}")
print(f"Gflop/s = {Gflops:.2g}")
print("---------------------------")
