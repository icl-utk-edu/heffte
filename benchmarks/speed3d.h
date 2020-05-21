/*
    -- heFFTe (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       Performance test for 3D FFTs using heFFTe
*/

#include "test_fft3d.h"

template<typename backend_tag, typename precision_type>
void benchmark_fft(std::array<int,3> size_fft){

    int me, nprocs;
    MPI_Comm fft_comm = MPI_COMM_WORLD;  // Change if need to compute FFT within a subcommunicator
    MPI_Comm_rank(fft_comm, &me);
    MPI_Comm_size(fft_comm, &nprocs);

    // Create input and output boxes on local processor
    box3d const world = {{0, 0, 0}, {size_fft[0]-1, size_fft[1]-1, size_fft[2]-1}};

    // Get grid of processors at input and output
    std::array<int,3> proc_i = heffte::proc_setup_min_surface(world, nprocs);
    std::array<int,3> proc_o = heffte::proc_setup_min_surface(world, nprocs);

    std::vector<box3d> inboxes  = heffte::split_world(world, proc_i);
    std::vector<box3d> outboxes = heffte::split_world(world, proc_o);

    // Define 3D FFT plan
    auto options = heffte::default_options<backend_tag>();


    heffte::fft3d<backend_tag> fft(inboxes[me], outboxes[me], fft_comm, options);

    std::array<int, 2> proc_grid = make_procgrid(nprocs);
    // writes out the proc_grid in the given dimension
    auto print_proc_grid = [&](int i){
        switch(i){
            case -1: cout << "(" << proc_i[0] << ", " << proc_i[1] << ", " << proc_i[2] << ")  "; break;
            case  0: cout << "(" << 1 << ", " << proc_grid[0] << ", " << proc_grid[1] << ")  "; break;
            case  1: cout << "(" << proc_grid[0] << ", " << 1 << ", " << proc_grid[1] << ")  "; break;
            case  2: cout << "(" << proc_grid[0] << ", " << proc_grid[1] << ", " << 1 << ")  "; break;
            case  3: cout << "(" << proc_o[0] << ", " << proc_o[1] << ", " << proc_o[2] << ")  "; break;
            default:
                throw std::runtime_error("printing incorrect direction");
        }
    };

    // the call above uses the following plan, get it twice to give verbose info of the grid-shapes
    logic_plan3d plan = plan_operations({inboxes, outboxes}, -1, heffte::default_options<backend_tag>());

    // Locally initialize input
    auto input = make_data<BENCH_INPUT>(inboxes[me]);
    auto reference_input = input; // safe a copy for error checking

    // define allocation for in-place transform
    std::vector<std::complex<precision_type>> output(std::max(fft.size_outbox(), fft.size_inbox()));
    std::copy(input.begin(), input.end(), output.begin());

    std::complex<precision_type> *output_array = output.data();
    #ifdef Heffte_ENABLE_CUDA
    cuda::vector<std::complex<precision_type>> cuda_output;
    if (std::is_same<backend_tag, backend::cufft>::value){
        cuda_output = cuda::load(output);
        output_array = cuda_output.data();
    }
    #endif

    // Define workspace array
    typename heffte::fft3d<backend_tag>::template buffer_container<std::complex<precision_type>> workspace(fft.size_workspace());

    // Warmup
    heffte::add_trace("mark warmup begin");
    fft.forward(output_array, output_array,  scale::full);
    fft.backward(output_array, output_array);

    // Execution
    int const ntest = 5;
    MPI_Barrier(fft_comm);
    double t = -MPI_Wtime();
    for(int i = 0; i < ntest; ++i) {
        heffte::add_trace("mark forward begin");
        fft.forward(output_array, output_array, workspace.data(), scale::full);
        heffte::add_trace("mark backward begin");
        fft.backward(output_array, output_array, workspace.data());
    }
    t += MPI_Wtime();
    MPI_Barrier(fft_comm);

    // Get execution time
    double t_max = 0.0;
	MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, fft_comm);

    // Validate result
    #ifdef Heffte_ENABLE_CUDA
    if (std::is_same<backend_tag, backend::cufft>::value){
        // unload from the GPU, if it was stored there
        output = cuda::unload(cuda_output);
    }
    #endif
    output.resize(input.size()); // match the size of the original input
    tassert(approx(output, input));

    // Print results
    if(me==0){
        t_max = t_max / (2.0 * ntest);
        double const fftsize  = 1.0 * world.count();
        double const floprate = 5.0 * fftsize * std::log(fftsize) * 1e-9 / std::log(2.0) / t_max;
        cout << "------------------------------- \n";
        cout << "heFFTe performance test\n";
        cout << "------------------------------- \n";
        cout << "Backend: " << backend::name<backend_tag>() << endl;
        cout << "Size: " << world.size[0] << "x" << world.size[1] << "x" << world.size[2] << endl;
        cout << "Nprc: " << nprocs << endl;
        cout << "Grids: ";
        print_proc_grid(-1);
        for(int i=0; i<4; i++)
            if (not match(plan.in_shape[i], plan.out_shape[i])) print_proc_grid((i<3) ? plan.fft_direction[i] : i);
        cout << "\n";
        cout << "Time: " << t_max << " (s)" << endl;
        cout << "Perf: " << floprate << " GFlops/s" << endl;
        cout << "Tolr: " << precision<std::complex<precision_type>>::tolerance << endl;
    }
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    #ifdef BENCH_C2C
    std::string bench_executable = "./speed3d_c2c";
    #else
    std::string bench_executable = "./speed3d_r2c";
    #endif

    std::string backends = "";
    #ifdef Heffte_ENABLE_FFTW
    backends += "fftw ";
    #endif
    #ifdef Heffte_ENABLE_CUDA
    backends += "cufft ";
    #endif
    #ifdef Heffte_ENABLE_MKL
    backends += "mkl ";
    #endif

    if (argc < 6){
        if (mpi::world_rank(0)){
            cout << "\nUsage:\n    mpirun -np x " << bench_executable << " <backend> <precision> <size-x> <size-y> <size-z>\n\n"
                 << "    options\n"
                 << "        backend is the 1-D FFT library\n"
                 << "            available options for this build: " << backends << "\n"
                 << "        precision is either float or double\n"
                 << "        size-x/y/z are the 3D array dimensions \n\n"
                 << "Examples:\n"
                 << "    mpirun -np 4 " << bench_executable << " fftw  double 128 128 128\n"
                 << "    mpirun -np 8 " << bench_executable << " cufft float  256 256 256\n\n";
        }

        MPI_Finalize();
        return 0;
    }

    std::array<int,3> size_fft = { 0, 0, 0 };

    std::string backend_string = argv[1];

    std::string precision_string = argv[2];
    if (precision_string != "float" and precision_string != "double"){
        if (mpi::world_rank(0)){
            std::cout << "Invalid precision!\n";
            std::cout << "Must use float or double" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    try{
        size_fft = { std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5])};
        for(auto s : size_fft) if (s < 1) throw std::invalid_argument("negative input");
    }catch(std::invalid_argument &e){
        if (mpi::world_rank(0)){
            std::cout << "Cannot convert the sizes into positive integers!\n";
            std::cout << "Encountered error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    init_tracing(bench_executable + "_" + backend_string + "_" + precision_string
                 + std::string(argv[3]) + "_" + std::string(argv[4]) + "_" + std::string(argv[5]));

    bool valid_backend = false;
    #ifdef Heffte_ENABLE_FFTW
    if (backend_string == "fftw"){
        if (precision_string == "float"){
            benchmark_fft<backend::fftw, float>(size_fft);
        }else{
            benchmark_fft<backend::fftw, double>(size_fft);
        }
        valid_backend = true;
    }
    #endif
    #ifdef Heffte_ENABLE_MKL
    if (backend_string == "mkl"){
        if (precision_string == "float"){
            benchmark_fft<backend::mkl, float>(size_fft);
        }else{
            benchmark_fft<backend::mkl, double>(size_fft);
        }
        valid_backend = true;
    }
    #endif
    #ifdef Heffte_ENABLE_CUDA
    if (backend_string == "cufft"){
        if (precision_string == "float"){
            benchmark_fft<backend::cufft, float>(size_fft);
        }else{
            benchmark_fft<backend::cufft, double>(size_fft);
        }
        valid_backend = true;
    }
    #endif

    if (not valid_backend){
        if (mpi::world_rank(0)){
            std::cout << "Invalid backend " << backend_string << "\n";
            std::cout << "The available backends are: " << backends << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    finalize_tracing();

    MPI_Finalize();
    return 0;
}
