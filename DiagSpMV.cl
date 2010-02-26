__kernel
void diagSpMV(__const int n,
              __const int ndiag,
              __global float *matrix,
              __const int pitch_in_floats,
              __constant int *offsets,
              __global float *vector,
              __global float *result) {
  unsigned int row = get_global_id(0);
  int col = row;
  float accumulant = 0;
  int matrix_offset = row;
  int d = 0;
  if (row < n) {
    while (d < ndiag) {
      col += offsets[d];
      if ((col >= 0) && (col < n)) {
        float m = matrix[matrix_offset];
        float v = vector[col];
        accumulant += m * v;
      }
      d++;
      matrix_offset += pitch_in_floats;
    }
    result[row] = accumulant;
  }
}
