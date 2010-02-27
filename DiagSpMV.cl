__kernel
void diagSpMV(__const int n,
              __const int ndiag,
              __global float4 *matrix,
              __const int pitch_in_float_4,
              __constant int *offsets,
              __global float *vector,
              __global float *result) {
  unsigned int row = get_global_id(0) * 4;
  int4 col = row;
  col.y += 1;
  col.z += 2;
  col.w += 3;
  float4 accumulant = 0;
  int matrix_offset = get_global_id(0);
  int d = 0;
  if (row < n) {
    while (d < ndiag) {
      col += offsets[d];
      float4 m = vload4(matrix_offset, matrix);
      float4 v;
      if ((col.x >=0) && (col.x < n - 4)) {
		v = vload4(col.x, vector);
	  } else {
		int4 in_bounds = (col >= 0) && (col < n);
		v.x = in_bounds.x ? vector[col.x] : 0;
		v.y = in_bounds.y ? vector[col.y] : 0;
		v.z = in_bounds.z ? vector[col.z] : 0;
		v.w = in_bounds.w ? vector[col.w] : 0;
	  }
      accumulant += m * v;
      d++;
      matrix_offset += pitch_in_float_4;
    }
  }
  if (row < n)     result[row] = accumulant.x;
  if (row + 1 < n) result[row+1] = accumulant.y;
  if (row + 2 < n) result[row+2] = accumulant.z;
  if (row + 3 < n) result[row+3] = accumulant.w;
}
