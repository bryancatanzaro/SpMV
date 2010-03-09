#ifndef DIA
#define DIA
#include <map>
#include <assert.h>

typedef unsigned int uint;

#define MAXDIAGS 1000

class Dia {
 public:
  Dia(const char* filename, int alignment_in_floats);
  uint get_ndiags();
  const int* const get_offsets();
  const float* const get_matrix();
  int get_matrix_pitch();
  int get_matrix_pitch_in_floats();
  uint get_n();
 private:
  uint ndiags;
  uint n;
  int* offsets;
  float* matrix;
  int alignment_in_floats;
  int pitch_in_floats;
};
#endif
