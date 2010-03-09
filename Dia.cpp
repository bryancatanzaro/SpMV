#include "Dia.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

Dia::Dia(const char* filename, int alignment_in_floats_in) : alignment_in_floats(alignment_in_floats_in) {
  FILE* fp = fopen(filename,"r");

  assert(fp != NULL);
  fread(&n,sizeof(int),1,fp);


  std::map<int, float*> diagonals;
  typedef std::map<int, float*>::iterator DiaIterator;

  int nnz = 0;
  fread(&nnz,sizeof(int),1,fp);

                         
  int nz = 0;
  int capacity = 100;
  int* indices = new int[capacity];  //Column indices
  double* values = new double[capacity];
  ndiags = 0;
  for (int row = 0; row < n; row++) {
    fread(&nz,sizeof(int),1,fp); //number of entries in this row
    if (nz > capacity) {
      capacity = nz;
      delete[] indices;
      delete[] values;
      indices = new int[capacity];
      values = new double[capacity];
    }

    fread(values,sizeof(double),nz,fp); //value
    fread(indices,sizeof(int),nz,fp);    //col index
    
    for (int i = 0; i < nz; i++) {
      int col = indices[i];
      int differential = col - row;
      DiaIterator diagonal_it = diagonals.find(differential);
      float* diagonal;
      if (diagonal_it == diagonals.end()) {
        diagonal = new float[n];
        memset(diagonal, 0, sizeof(float) * n);
        diagonals[differential] = diagonal;
        ndiags++;
      } else {
        diagonal = diagonal_it->second;
      }
     
      diagonal[row] = (float)values[i];
    } 
  }
  fclose(fp);
  
  delete[] indices;
  delete[] values;

  pitch_in_floats = ((n - 1)/alignment_in_floats + 1) * alignment_in_floats;

  matrix = new float[pitch_in_floats * ndiags];
  offsets = new int[ndiags];
  
  int row = 0;
  for(DiaIterator i = diagonals.begin(); i != diagonals.end(); i++) {
    int offset = i->first;
    float* storage = i->second;
    memcpy(row*pitch_in_floats + matrix, storage, sizeof(float) * n);
    delete[] storage;
    offsets[row] = offset;
    row++;
  }
  

 //  for(int i = 0; i < ndiags; i++) {
//     printf("%2d:  ", offsets[i]);
//     for(int j = 0; j < n; j++) {
//         printf("%1.2f ", matrix[i * pitch_in_floats + j]);
//     }
//     printf("\n");
//   }
  
  
 
}

uint Dia::get_ndiags() {
  return ndiags;
}

uint Dia::get_n() {
  return n;
}

const int* const Dia::get_offsets() {
  return offsets;
}

const float* const Dia::get_matrix() {
  return matrix;
}

int Dia::get_matrix_pitch() {
  return pitch_in_floats * sizeof(float);
}

int Dia::get_matrix_pitch_in_floats() {
  return pitch_in_floats;
}


