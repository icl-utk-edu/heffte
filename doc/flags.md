HEFFTE Flags
============

1. cflag: communication  can be set to flag (default = point):
          point = point-to-point comm
          all = use MPI_all2all collective

2. eflag: exchange flag can be set to (default = pencil):
          pencil = pencil to pencil data exchange (4 stages for full FFT)
          brick = brick to pencil data exchange (6 stages for full FFT)

3. pflag: select pack/unpack for methods of data remapping, can be set to (default = memcpy):
          array = array based
          ptr = pointer based
          memcpy = memcpy based

4. tflag: print split timing of routines (default = deactivated)

5. rflag: call HEFFTE to compute only data reshapes without computation (default = deactivated)

6. oflag: print full array before/after computation (default = deactivated)

7. mode : FFT computation requirement for the test, can be set to (default = 0):
          0 = computes a forward and backward FFT.
          1 = computes only a forward FFT.

8. vflag: Check correctness of FFT computation (default = deactivated)

9. verb : Verbosity flag, prints hardware and extra information (default = deactivated)


HEFFTE also provides flags to tune the library during the setup time
to use the best configuration. Currently two flags are supported:

1. FHEFFTE_ESTIMATE: Minimal/no tuning.
2. FHEFFTE_MEASURE: Tunes for local FFT execution algorithm, as well as global transposes to
reduce communication time.
