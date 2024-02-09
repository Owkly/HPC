# Academic High Performance Computing Projects

## Description:
This repository contains two academic projects focusing on High Performance Computing (HPC), specifically utilizing MPI (Message Passing Interface) and OpenMP with intrinsic optimizations (AVX and NEON).

### Project Details:
- **Supervisor:** Charles Bouillaguet
- **Email:** charles.bouillaguet@lip6.fr
- **Project Contributors:** Yannick Zhang, Romain Capron
- **University/Institution:** Polytech Sorbonne, Université Pierre et Marie Curie (UPMC)

## Project Structure:

This repository consists of two main repertory, each containing:

- Initial problem statements and sequential codes
- Parallel implementations using MPI or OpenMP and intrinsic optimizations (AVX and NEON)
- Python scripts for result analysis
- Detailed reports documenting the methodology and interpretation of results


# Running the Codes:

To utilize the provided codes, follow these steps:

## Prerequisites:
1. Clone the repository:
    ```bash
    git clone https://github.com/Owkly/HPC
    ```

2. Navigate to the desired project directory:
    ```bash
    cd <project_directory>
    ```
	Replace `<project_directory>` with either `MPI` or `OMP_Intrinsic`.

3. Compile the codes using :
    ```bash
    make
    ```

4. Optionally, clean the directory (remove executables and temporary files) using:
    ```bash
    make clean
    ```

## Execute the desired program:

### MPI:
- For the sequential version:
    ```bash
    ./heatsink_sequential
    ```

- For the parallel version:
    ```bash
    make run NUM=<number_of_processes> TARGET=<executable_name>
    ```
    Replace `<number_of_processes>` with the desired number of processes and `<executable_name>` with the name of the target executable, either `heatsink_1D` or `heatsink_2D`.

- For graphical representation:
    ```bash
    make graph MODE=<mode>
    ```
    Replace `<mode>` with either `FAST`, `MEDIUM`, or `NORMAL`. (Default mode is `FAST`)

    NOTE: If you want to change the mode, you have to modify the specific C file (in the `#define` on line 23) and recompile the code using the `make` command.

### OpenMP and Intrinsic:

- For sequential execution:
    ```bash
    make run-seq
    ```

- For parallel version:
    ```bash
    make run TARGET=<executable_name> THREADS=<number_of_threads>
    ```
    Replace `<number_of_threads>` with the desired number of threads and `<executable_name>` with the name of the target executable (`fft_openmp` or `fft_intrinsic_vec`).
