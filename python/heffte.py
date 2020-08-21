'''
    Python Interface
    -- heFFTe --
    Univ. of Tennessee, Knoxville

Env setup:
    Define where libheffte.so is located:
        export heffte_lib_path=/ccs/home/aayala/tmprl/heffte/build_x_cpu
        export fftw_lib_path=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/fftw-3.3.8-azzdjlzx2j6dpqvzdir2nwvxypohyfq4/lib
    Set env:
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$heffte_lib_path
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$fftw_lib_path
        export PYTHONPATH=$PYTHONPATH:$heffte_lib_path

Run R2C test:
        jsrun -n2 python speed3d.py
'''

# Import modules
from ctypes import *
import sys, traceback
from numpy.ctypeslib import ndpointer

import numpy as np
from mpi4py import MPI

class heffte_plan(Structure):
    pass
heffte_plan._fields_ = [ ("backend_type", c_int), ("using_r2c", c_int), ("fft", c_void_p) ]
LP_plan = POINTER(heffte_plan)

class heffte_plan_options(Structure):
    pass
heffte_plan_options._fields_ = [ ("use_reorder", c_int), ("use_alltoall", c_int), ("use_pencils", c_int) ]


heffte_backend = {'fftw': 1, 'mkl': 2, 'cufft': 10}
heffte_scale   = {'none': 0, 'full': 1, 'symmetric': 2}

class heffte_options():
    def __init__(self, b_type='fftw'):

        if( b_type not in [*heffte_backend]):
            raise OSError( str(b_type) + "-backend is not allowed on heFFTe!" )
        self.backend = b_type

class fft3d:

    def __init__(self, comm,  backend, 
            inbox_low, inbox_high, inbox_order,
            outbox_low, outbox_high, outbox_order):

        # load heffte.so
        try:
            self.lib = CDLL("libheffte.so", RTLD_GLOBAL)
        except:
            etype, value, tb = sys.exc_info()
            traceback.print_exception(etype, value, tb)
            raise OSError("Could not load heFFTe dynamic library")

        # define ctypes API for each library method
        if MPI._sizeof(comm) == sizeof(c_int): MPI_Comm = c_int
        else: MPI_Comm = c_void_p
        self.comm_ptr = MPI._addressof(comm)
        self.comm_value = MPI_Comm.from_address(self.comm_ptr)

        # Plan create
        self.lib.heffte_plan_create.argtypes = [c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), \
                                                ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), \
                                                ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), \
                                                MPI_Comm, POINTER(heffte_plan_options), POINTER(LP_plan)]
        self.lib.heffte_plan_create.restype = c_int

        # FFT execution
        self.lib.heffte_forward_c2c.argtypes = [LP_plan, POINTER(c_float), POINTER(c_float), c_int]
        self.lib.heffte_forward_c2c.restype = None

        self.lib.heffte_forward_s2c.argtypes = [LP_plan, POINTER(c_float), c_void_p, c_int]
        self.lib.heffte_forward_s2c.restype = None

        # Plan destroy
        self.lib.heffte_plan_destroy.argtypes = [LP_plan]
        self.lib.heffte_plan_destroy.restype = c_int

        # Initialize
        self.fft_comm = comm
        self.plan = LP_plan()
        options = heffte_plan_options(0,1,1)

        tmp = self.lib.heffte_plan_create(backend, inbox_low, inbox_high, inbox_order,
                                    outbox_low, outbox_high, outbox_order,
                                    self.comm_value, options, self.plan)

        # if(tmp == 0):
            # print("----------------------------------")
            # print("FFT plan successfully created.")

    def __del__(self):
        self.lib.heffte_plan_destroy(self.plan)
        self.lib = None

    def forward(self, input, output, scale):
        if "numpy" not in str(type(input)) or "numpy" not in str(type(output)):
            print( "Input/Output data must be numpy arrays for computing the FFT.")
            sys.exit()

        c_in  = input.ctypes.data_as(POINTER(c_float))
        c_out = output.ctypes.data_as(c_void_p)

        self.lib.heffte_forward_s2c(self.plan, c_in, c_out, scale)

        print("---------------------------")
        print("\nComputed FFT:")
        print(output.view(dtype=np.complex64))
