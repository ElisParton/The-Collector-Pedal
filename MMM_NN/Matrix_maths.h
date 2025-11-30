#ifndef MATRIX_MATHS_H
#define MATRIX_MATHS_H

typedef struct Matrix {
    double* entries;
    int rows;
    int cols;
} Matrix;

Matrix matrix_create(int row, int col, double* buffer);
Matrix matrix_multiplication(const Matrix A, const Matrix B, double* buffer);
Matrix matrix_addition(const Matrix A, const Matrix B, double* buffer);
Matrix matrix_subtraction(const Matrix A, const Matrix B, double* buffer);
void   matrix_print(Matrix A);

#endif