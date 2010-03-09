__kernel
void diagSpMV(__const int n,
              __const int ndiag,
              __global float *matrix,
              __const int pitch_in_float,
              __global int *offsets,
              __global float *vector,
              __global float *result) {
  __local int l_offsets[256];
  int local_id = get_local_id(0);
  int offset_id = local_id;
  while ((offset_id < ndiag) && (offset_id < 256)) {
    l_offsets[offset_id] = offsets[offset_id];
    offset_id = offset_id + get_local_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);            
  unsigned int row = get_global_id(0);
  float accumulant = 0.0f;
  int d = 0;
  __global int* matrix_offset = matrix + get_global_id(0);

  if (row < n) {
    while (d < ndiag) {
      int col = row + l_offsets[d];
      
      if ((col >= 0) && (col < n)) {
        float m = *matrix_offset;

        float v = vector[col];
        accumulant += m * v; 
      }
      d++;
      matrix_offset += pitch_in_float;
    }
  }
  if (row < n)     result[row] = accumulant;
  
}
