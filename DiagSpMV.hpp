#ifndef DIAGSPMV_H_
#define DIAGSPMV_H_

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <SDKUtil/SDKCommon.hpp>
#include <SDKUtil/SDKApplication.hpp>
#include <SDKUtil/SDKCommandArgs.hpp>
#include <SDKUtil/SDKFile.hpp>
#include <iostream>
#include "Stencil.h"
//#include <fcntl.h>
//#include <unistd.h>

/**
 * Diagonal Sparse Matrix Vector Multiply
 * Class implements OpenCL Sparse Matrix Vector Multiply sample
 * Derived from SDKSample base class
 */

class DiagSpMV : public SDKSample
{
  cl_double           setupTime;       /**< Time for setting up OpenCL */
  cl_double     totalKernelTime;       /**< Time for kernel execution */
  cl_double    totalProgramTime;       /**< Time for program execution */
  cl_double referenceKernelTime;       /**< Time for reference implementation */
  cl_int                 radius;       /**< Radius of the stencil */
  cl_int                  width;       /**< Width of the grid */
  cl_int                 height;       /**< Height of the grid */
  cl_int                nPoints;       /**< Width * height */
  cl_int                  nDiag;       /**< Number of diagonals */
  std::string              file;       /**< Filename of matrix */
  Stencil*              stencil;       /**< Stencil parameters */
  cl_float              *matrix;       /**< Input array */
  cl_int                 pitchf;       /**< Matrix pitch in floats */
  cl_int               *offsets;       /**< Input offsets */
  cl_float              *vector;       /**< Input vector */
  cl_float              *output;       /**< Output vector */
  
  cl_context            context;       /**< CL context */
  cl_device_id         *devices;       /**< CL device list */
 
  cl_mem              devMatrix;       /**< CL memory representing matrix */
  cl_mem             devOffsets;       /**< CL memory representing offsets */
  cl_mem              devVector;       /**< CL memory representing vector */
  cl_mem              devOutput;       /**< CL memory representing output */
  
  cl_command_queue commandQueue;       /**< CL command queue */
  cl_program            program;       /**< CL program  */
  cl_kernel              kernel;       /**< CL kernel */
  size_t    kernelWorkGroupSize;       /**< Group Size returned by kernel */
  int                iterations;       /**< Number of iterations for kernel execution */

public:
    /** 
     * Constructor 
     * Initialize member variables
     * @param name name of sample (string)
     */
  DiagSpMV(std::string name)
    : SDKSample(name){
    radius = 1;
    width = 4;
    height = 4;
    file = "tiny.sma";
    setupTime = 0;
    totalKernelTime = 0;
    iterations = 1;
  }
  
  /** 
     * Constructor 
     * Initialize member variables
     * @param name name of sample (const char*)
     */
  DiagSpMV(const char* name)
    : SDKSample(name){
    radius = 1;
    width = 4;
    height = 4;
    file = "tiny.sma";
    setupTime = 0;
    totalKernelTime = 0;
    iterations = 1;
  }


  void print_state();
  
    /**
     * Allocate and initialize host memory array with random values
     * @return 1 on success and 0 on failure
     */
    int setupSpMV();

    /**
     * OpenCL related initialisations. 
     * Set up Context, Device list, Command Queue, Memory buffers
     * Build CL kernel program executable
     * @return 1 on success and 0 on failure
     */
    int setupCL();

    /**
     * Set values for kernels' arguments, enqueue calls to the kernels
     * on to the command queue, wait till end of kernel execution.
     * Get kernel start and end time if timing is enabled
     * @return 1 on success and 0 on failure
     */
    int runCLKernels();

    /**
     * Reference CPU implementation of inplace FastWalsh Transform
     * for performance comparison 
     * @param input input array which also stores the output 
     * @param length length of the array
     */
    void cpuReference(cl_float * input);

    /**
     * Override from SDKSample. Print sample stats.
     */
    void printStats();

    /**
     * Override from SDKSample. Initialize 
     * command line parser, add custom options
     */
    int initialize();

    /**
     * Override from SDKSample, adjust width and height 
     * of execution domain, perform all sample setup
     */
    int setup();

    /**
     * Override from SDKSample
     * Run OpenCL FastWalsh Transform 
     */
    int run();

    /**
     * Override from SDKSample
     * Cleanup memory allocations
     */
    int cleanup();

    /**
     * Override from SDKSample
     * Verify against reference implementation
     */
    int verifyResults();
};



#endif
