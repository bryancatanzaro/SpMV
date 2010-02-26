#include "DiagSpMV.hpp"

void writeArray(char* file, int ndim, int* dim, float* input) {
  int fd;
  fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
  int size = 1;
  for(int i = 0; i < ndim; i++) {
    size *= dim[i];
  }
  write(fd, &ndim, sizeof(int));
  write(fd, dim, sizeof(int) * ndim);
  write(fd, input, sizeof(float) * size);
  close(fd);
}

int iDivCeil(int n, int d) {
  return (n - 1)/d + 1;
}

int 
DiagSpMV::setupSpMV()
{
  //ATI GPUs like things to be 128 bit aligned
  int alignmentInFloats = 32; 
  nPoints = width * height;
  pitchf = iDivCeil(nPoints, alignmentInFloats) * alignmentInFloats;
  std::cout << "Pitch in floats: " << pitchf << std::endl;
  stencil = new Stencil(radius, width, height, pitchf);
  nDiag = stencil->getStencilArea();
  matrix = stencil->readStencilMatrix(file.c_str());
  // for(int i = 0; i < nDiag; i++) {
//     for(int j = 0; j < nPoints; j++) {
//       std::cout << matrix[i * pitchf + j] << ", ";
//     }
//     std::cout << std::endl;
//   }
  offsets = (int*)malloc(sizeof(int) * nDiag);
  stencil->copyOffsets(offsets);
  vector = (float*)malloc(sizeof(float) * nPoints);
  output = (float*)malloc(sizeof(float) * nPoints);
  
  for(int i = 0; i < nPoints; i++) {
    vector[i] = 1.0;
  }  
  return SDK_SUCCESS;
}

int
DiagSpMV::setupCL(void)
{
    cl_int status = 0;
    size_t deviceListSize;

    cl_device_type dType;
    
    if(deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu" 
    {
        dType = CL_DEVICE_TYPE_GPU;
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */

    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(!sampleCommon->checkVal(status,
                               CL_SUCCESS,
                               "clGetPlatformIDs failed."))
    {
        return SDK_FAILURE;
    }
    if (0 < numPlatforms) 
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if(!sampleCommon->checkVal(status,
                                   CL_SUCCESS,
                                   "clGetPlatformIDs failed."))
        {
            return SDK_FAILURE;
        }
        for (unsigned i = 0; i < numPlatforms; ++i) 
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);

            if(!sampleCommon->checkVal(status,
                                       CL_SUCCESS,
                                       "clGetPlatformInfo failed."))
            {
                return SDK_FAILURE;
            }

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
            {
                break;
            }
        }
        delete[] platforms;
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] = 
    {
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)platform, 
        0
    };
    /* Use NULL for backward compatibility */
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;

    context = clCreateContextFromType(
                  cprops,
                  dType,
                  NULL,
                  NULL,
                  &status);
 
    if(!sampleCommon->checkVal(status, 
                  CL_SUCCESS,
                  "clCreateContextFromType failed."))
        return SDK_FAILURE;

    /* First, get the size of device list data */
    status = clGetContextInfo(
                 context, 
                 CL_CONTEXT_DEVICES, 
                 0, 
                 NULL, 
                 &deviceListSize);
    if(!sampleCommon->checkVal(
            status, 
            CL_SUCCESS,
            "clGetContextInfo failed."))
        return SDK_FAILURE;

    /* Now allocate memory for device list based on the size we got earlier */
    devices = (cl_device_id *)malloc(deviceListSize);
    if(devices==NULL) {
        sampleCommon->error("Failed to allocate memory (devices).");
        return SDK_FAILURE;
    }

    /* Now, get the device list data */
    status = clGetContextInfo(
                 context, 
                 CL_CONTEXT_DEVICES, 
                 deviceListSize, 
                 devices, 
                 NULL);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS, 
            "clGetGetContextInfo failed."))
        return SDK_FAILURE;

    {
        /* The block is to move the declaration of prop closer to its use */
        cl_command_queue_properties prop = 0;
        if(timing)
            prop |= CL_QUEUE_PROFILING_ENABLE;

        commandQueue = clCreateCommandQueue(
                           context, 
                           devices[0], 
                           prop, 
                           &status);
        if(!sampleCommon->checkVal(
                status,
                0,
                "clCreateCommandQueue failed."))
            return SDK_FAILURE;
    }

    devMatrix = clCreateBuffer(
                               context,
                               CL_MEM_READ_ONLY,
                               sizeof(cl_float) * nDiag * stencil->getMatrixPitchInFloats(),
                               0,
                               &status);
    devOffsets = clCreateBuffer(
                               context,
                               CL_MEM_READ_ONLY,
                               sizeof(cl_int) * nDiag,
                               0,
                               &status);
    devVector = clCreateBuffer(
                               context,
                               CL_MEM_READ_ONLY,
                               sizeof(cl_float) * nPoints,
                               0,
                               &status);
    devOutput = clCreateBuffer(
                               context,
                               CL_MEM_WRITE_ONLY,
                               sizeof(cl_float) * nPoints,
                               0,
                               &status);
    
                               
    
    // inputBuffer = clCreateBuffer(
//                       context, 
//                       CL_MEM_READ_WRITE,
//                       sizeof(cl_float) * length,
//                       0, 
//                       &status);
//     if(!sampleCommon->checkVal(
//             status,
//             CL_SUCCESS,
//             "clCreateBuffer failed. (inputBuffer)"))
//         return SDK_FAILURE;

    /* create a CL program using the kernel source */
    streamsdk::SDKFile kernelFile;
    std::string kernelPath = sampleCommon->getPath();
    kernelPath.append("DiagSpMV.cl");
    kernelFile.open(kernelPath.c_str());
    const char * source = kernelFile.source().c_str();
    size_t sourceSize[] = { strlen(source) };
    program = clCreateProgramWithSource(
        context,
        1,
        &source,
        sourceSize,
        &status);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clCreateProgramWithSource failed."))
        return SDK_FAILURE;
    
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clCreateProgramWithBinary failed."))
        return SDK_FAILURE;

    /* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    char *build_log;
    size_t ret_val_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    build_log = new char[ret_val_size+1];
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    
    // to be carefully, terminate with \0
    // there's no information in the reference whether the string is 0 terminated or not
    build_log[ret_val_size] = '\0';
    
    
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clBuildProgram failed.")) {
      std::cout << "BUILD LOG:" << std::endl;
      std::cout << build_log << std::endl;

      return SDK_FAILURE;
    }

    /* get a kernel object handle for a kernel with the given name */
    kernel = clCreateKernel(program, "diagSpMV", &status);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clCreateKernel failed."))
        return SDK_FAILURE;

    return SDK_SUCCESS;
}


int 
DiagSpMV::runCLKernels(void)
{
    cl_int   status;
    cl_event *events = new cl_event[iterations];

    size_t globalThreads[1];
    size_t localThreads[1];

    /* Enqueue write input to devMatrix */
    status = clEnqueueWriteBuffer(commandQueue,
                                  devMatrix,
                                  CL_TRUE,
                                  0,
                                  stencil->getMatrixPitch() * nDiag,
                                  matrix,
                                  0,
                                  0,
                                  0);
    if(!sampleCommon->checkVal(
                        status,
                        CL_SUCCESS, 
                        "clEnqueueWriteBuffer failed."))
    {
        return SDK_FAILURE;
    }

        /* Enqueue write input to devOffsets */
    status = clEnqueueWriteBuffer(commandQueue,
                                  devOffsets,
                                  CL_TRUE,
                                  0,
                                  nDiag * sizeof(int),
                                  offsets,
                                  0,
                                  0,
                                  0);
    if(!sampleCommon->checkVal(
                        status,
                        CL_SUCCESS, 
                        "clEnqueueWriteBuffer failed."))
    {
        return SDK_FAILURE;
    }

        /* Enqueue write input to devVector */
    status = clEnqueueWriteBuffer(commandQueue,
                                  devVector,
                                  CL_TRUE,
                                  0,
                                  nPoints * sizeof(float),
                                  vector,
                                  0,
                                  0,
                                  0);
    if(!sampleCommon->checkVal(
                        status,
                        CL_SUCCESS, 
                        "clEnqueueWriteBuffer failed."))
    {
        return SDK_FAILURE;
    }

    
    

   
    globalThreads[0] = (nPoints - 1)/4 + 1;
    localThreads[0]  = 128;
    if (localThreads[0] > globalThreads[0]) {
      localThreads[0] = globalThreads[0];
    }

    if (globalThreads[0] % localThreads[0] > 0) {
      globalThreads[0] += localThreads[0] - ((int)globalThreads[0] % (int)localThreads[0]);
    }
    /* Check group size against kernelWorkGroupSize */
    status = clGetKernelWorkGroupInfo(kernel,
                                      devices[0],
                                      CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(size_t),
                                      &kernelWorkGroupSize,
                                      0);
    if(!sampleCommon->checkVal(
                        status,
                        CL_SUCCESS, 
                        "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }
  
    std::cout << "Global threads: " << globalThreads[0] << std::endl;
    std::cout << "Local threads: " << localThreads[0] << std::endl;
    std::cout << "Work group Size: " << kernelWorkGroupSize << std::endl;

    /*** Set appropriate arguments to the kernel ***/
    /* the input array - also acts as output*/
    
    status = clSetKernelArg(
                    kernel, 
                    0, 
                    sizeof(cl_int), 
                    (void *)&nPoints);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (nPoints)"))
        return SDK_FAILURE;
    
    status = clSetKernelArg(
                    kernel, 
                    1, 
                    sizeof(cl_int), 
                    (void *)&nDiag);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (nDiag)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
                    kernel, 
                    2, 
                    sizeof(cl_mem), 
                    (void *)&devMatrix);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (matrix)"))
        return SDK_FAILURE;


    int pitch_in_float_4 = pitchf/4;
    
    status = clSetKernelArg(
                    kernel, 
                    3, 
                    sizeof(cl_int), 
                    (void *)&pitch_in_float_4);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (pitchf)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
                    kernel, 
                    4, 
                    sizeof(cl_mem), 
                    (void *)&devOffsets);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (offsets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
                    kernel, 
                    5, 
                    sizeof(cl_mem), 
                    (void *)&devVector);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (vector)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
                    kernel, 
                    6, 
                    sizeof(cl_mem), 
                    (void *)&devOutput);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clSetKernelArg failed. (output)"))
        return SDK_FAILURE;


    int timer = sampleCommon->createTimer();
    sampleCommon->resetTimer(timer);
    sampleCommon->startTimer(timer);   

    std::cout << "Executing kernel for " << iterations << 
        " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    for(int i = 0; i < iterations; i++)
    {
      if (i == 0) {
        /* Enqueue a kernel run call */
        status = clEnqueueNDRangeKernel(
                                        commandQueue,
                                        kernel,
                                        1,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &events[0]);
      } else {
        /* Enqueue a kernel run call */
        status = clEnqueueNDRangeKernel(
                                        commandQueue,
                                        kernel,
                                        1,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        1,
                                        &events[i-1],
                                        &events[i]);
      }  
      if(!sampleCommon->checkVal(
                                 status,
                                 CL_SUCCESS,
                                 "clEnqueueNDRangeKernel failed.")) {
        std::cout << "Iteration: " << i << std::endl;
        return SDK_FAILURE;
      }
    }
    status = clWaitForEvents(1, &events[iterations-1]);
    if(!sampleCommon->checkVal(
                               status,
                               CL_SUCCESS,
                               "clWaitForEvents failed."))
      return SDK_FAILURE;
    
    sampleCommon->stopTimer(timer);
    totalKernelTime = (double)(sampleCommon->readTimer(timer)) / iterations;
    std::cout << "Kernel time per iteration: " << totalKernelTime << std::endl;
    
    status = clEnqueueReadBuffer(
                                 commandQueue,
                                 devOutput,
                                 CL_TRUE,
                                 0,
                                 nPoints *  sizeof(cl_float),
                                 output,
                                 0,
                                 NULL,
                                 NULL);
    if(!sampleCommon->checkVal(
                               status,
                               CL_SUCCESS,
                               "clEnqueueReadBuffer failed."))
      return SDK_FAILURE;

    int dims[2];
    dims[0] = 1;
    dims[1] = nPoints;
    writeArray("output.ary", 2, &dims[0], output); 
    
    
    return SDK_SUCCESS;

}


void 
DiagSpMV::cpuReference(cl_float * vinput)
{
    /* for each pass of the algorithm */
   //  for(cl_uint step=1; step < length; step <<=1)
//     {
//         /* length of each block */
//         cl_uint jump = step << 1;
//         /* for each blocks */
//         for(cl_uint group = 0; group < step; ++group)
//         {
//             /* for each pair of elements with in the block */
//             for(cl_uint pair = group; pair < length; pair += jump)
//             {
//                 /* find its partner */
//                 cl_uint match = pair + step;
                
//                 cl_float T1 = vinput[pair];
//                 cl_float T2 = vinput[match];
                
//                 /* store the sum and difference of the numbers in the same locations */
//                 vinput[pair] = T1 + T2;
//                 vinput[match] = T1 - T2;
//             }
//         }
//     }
}

int 
DiagSpMV::initialize()
{
    // Call base class Initialize to get default configuration
    if(!this->SDKSample::initialize())
        return SDK_FAILURE;

    // Now add customized options
    streamsdk::Option* o_radius = new streamsdk::Option;
    if(!o_radius)
    {
        sampleCommon->error("Memory allocation error.\n");
        return SDK_FAILURE;
    }
    
    o_radius->_sVersion = "r";
    o_radius->_lVersion = "radius";
    o_radius->_description = "Stencil radius";
    o_radius->_type = streamsdk::CA_ARG_INT;
    o_radius->_value = &radius;
    sampleArgs->AddOption(o_radius);
    delete o_radius;

    streamsdk::Option* o_width = new streamsdk::Option;
    if(!o_width)
    {
        sampleCommon->error("Memory allocation error.\n");
        return SDK_FAILURE;
    }
    
    o_width->_sVersion = "w";
    o_width->_lVersion = "width";
    o_width->_description = "Grid width";
    o_width->_type = streamsdk::CA_ARG_INT;
    o_width->_value = &width;
    sampleArgs->AddOption(o_width);
    delete o_width;

    streamsdk::Option* o_height = new streamsdk::Option;
    if(!o_height)
    {
        sampleCommon->error("Memory allocation error.\n");
        return SDK_FAILURE;
    }
    
    o_height->_sVersion = "h";
    o_height->_lVersion = "height";
    o_height->_description = "Grid height";
    o_height->_type = streamsdk::CA_ARG_INT;
    o_height->_value = &height;
    sampleArgs->AddOption(o_height);
    delete o_height;

    streamsdk::Option* o_file = new streamsdk::Option;
    if(!o_file)
    {
        sampleCommon->error("Memory allocation error.\n");
        return SDK_FAILURE;
    }
    
    o_file->_sVersion = "f";
    o_file->_lVersion = "file";
    o_file->_description = "Matrix file";
    o_file->_type = streamsdk::CA_ARG_STRING;
    o_file->_value = &file;
    sampleArgs->AddOption(o_file);
    delete o_file;
    
    
    streamsdk::Option* num_iterations = new streamsdk::Option;
    if(!num_iterations)
    {
        sampleCommon->error("Memory allocation error.\n");
        return SDK_FAILURE;
    }

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = streamsdk::CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;


    return SDK_SUCCESS;
}

int 
DiagSpMV::setup()
{
//     /* make sure the length is the power of 2 */
//     if(!sampleCommon->isPowerOf2(length))
//         length = sampleCommon->roundToPowerOf2(length);

    if(setupSpMV()!=SDK_SUCCESS)
        return SDK_FAILURE;
    
    int timer = sampleCommon->createTimer();
    sampleCommon->resetTimer(timer);
    sampleCommon->startTimer(timer);

    if(setupCL()!=SDK_SUCCESS)
        return SDK_FAILURE;

    sampleCommon->stopTimer(timer);

    setupTime = (cl_double)sampleCommon->readTimer(timer);

    return SDK_SUCCESS;
}


int 
DiagSpMV::run()
{
    

   
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels() != SDK_SUCCESS)
      return SDK_FAILURE;
   

    

    // if(!quiet) {
//         sampleCommon->printArray<cl_float>("Output", input, length, 1);
//     }

    return SDK_SUCCESS;
}

int 
DiagSpMV::verifyResults()
{
    // if(verify) {
//         /* reference implementation
//          * it overwrites the input array with the output
//          */
//         int refTimer = sampleCommon->createTimer();
//         sampleCommon->resetTimer(refTimer);
//         sampleCommon->startTimer(refTimer);
//         cpuReference(verificationInput, length);
//         sampleCommon->stopTimer(refTimer);
//         referenceKernelTime = sampleCommon->readTimer(refTimer);

//         /* compare the results and see if they match */
//         if(sampleCommon->compare(output, verificationInput, length))
//         {
//             std::cout<<"Passed!\n";
//             return SDK_SUCCESS;
//         }
//         else
//         {
//             std::cout<<"Failed\n";
//             return SDK_FAILURE;
//         }
//     }

    return SDK_SUCCESS;
}

void 
DiagSpMV::printStats()
{
    std::string strArray[3] = {"Size", "Time(sec)", "kernelTime(msec)"};
    std::string stats[3];

    totalTime = setupTime + totalKernelTime ;

    stats[0] = sampleCommon->toString(width*height, std::dec);
    stats[1] = sampleCommon->toString(totalTime, std::dec);
    stats[2] = sampleCommon->toString(totalKernelTime*1000, std::dec);

    this->SDKSample::printStats(strArray, stats, 3);
}

int 
DiagSpMV::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(kernel);
    if(!sampleCommon->checkVal(
        status,
        CL_SUCCESS,
        "clReleaseKernel failed."))
        return SDK_FAILURE;

    status = clReleaseProgram(program);
    if(!sampleCommon->checkVal(
        status,
        CL_SUCCESS,
        "clReleaseProgram failed."))
        return SDK_FAILURE;
 
//     status = clReleaseMemObject(inputBuffer);
//     if(!sampleCommon->checkVal(
//         status,
//         CL_SUCCESS,
//         "clReleaseMemObject failed."))
//         return SDK_FAILURE;

    status = clReleaseCommandQueue(commandQueue);
     if(!sampleCommon->checkVal(
        status,
        CL_SUCCESS,
        "clReleaseCommandQueue failed."))
        return SDK_FAILURE;

    status = clReleaseContext(context);
    if(!sampleCommon->checkVal(
            status,
            CL_SUCCESS,
            "clReleaseContext failed."))
        return SDK_FAILURE;

    /* release program resources (input memory etc.) */
    // if(input) 
//         free(input);

//     if(verificationInput) 
//         free(verificationInput);

    if(devices)
        free(devices);

    return SDK_SUCCESS;
}

void DiagSpMV::print_state() {
    std::cout << "Radius " << radius << std::endl;
    std::cout << "Width " << width << std::endl;
    std::cout << "Height " << height << std::endl;
    std::cout << "File " << file << std::endl;
  }

int 
main(int argc, char * argv[])
{
    DiagSpMV clDiagSpMV("OpenCL Diagonal Sparse Matrix Vector Multiply");

    clDiagSpMV.initialize();
    if(!clDiagSpMV.parseCommandLine(argc, argv))
        return SDK_FAILURE;
    clDiagSpMV.print_state();
    if(clDiagSpMV.setup()!=SDK_SUCCESS)
        return SDK_FAILURE;
   
    if(clDiagSpMV.run()!=SDK_SUCCESS)
        return SDK_FAILURE;
//     if(clDiagSpMV.verifyResults()!=SDK_SUCCESS)
//         return SDK_FAILURE;
    if(clDiagSpMV.cleanup()!=SDK_SUCCESS)
        return SDK_FAILURE;
    clDiagSpMV.printStats();

    return SDK_SUCCESS;
}
