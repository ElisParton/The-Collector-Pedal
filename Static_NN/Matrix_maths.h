#ifndef MATRIX_MATHS_H
#define MATRIX_MATHS_H

#include <assert.h>

/**
 * @brief A struct representing a 2D matrix
 */
typedef struct Matrix
{
    double *entries; ///< Pointer to contiguous matrix data
    int rows;        ///< Number of rows
    int cols;        ///< Number of columns
} Matrix;

/*=========================================================
  Core construction / access
=========================================================*/

/**
 * @brief Wrap an existing buffer as a matrix
 */
Matrix matrix_create(int row, int col, double *buffer);

/**
 * @brief Get value at (r, c)
 *
 * No bounds checking. Undefined behavior if indices are invalid.
 */
static inline double matrix_get_entry(const Matrix m, int r, int c)
{
    return m.entries[r * m.cols + c];
}

/**
 * @brief Set value at (r, c)
 *
 * No bounds checking. Undefined behavior if indices are invalid.
 */
static inline void matrix_set_entry(Matrix m, int r, int c, double v)
{
    m.entries[r * m.cols + c] = v;
}

/*=========================================================
  Utility operations
=========================================================*/

Matrix matrix_set_all(Matrix A, double value);
Matrix matrix_slice(const Matrix A,
                    int row_start, int row_end,
                    int col_start, int col_end,
                    double *buffer);

void matrix_print(Matrix A);

/*=========================================================
  Algebraic operations
=========================================================*/

Matrix matrix_addition(const Matrix A, const Matrix B, double *buffer);
Matrix matrix_subtraction(const Matrix A, const Matrix B, double *buffer);

Matrix matrix_multiplication(const Matrix A, const Matrix B, double *buffer);
Matrix matrix_outer_product(const Matrix A, const Matrix B, double *buffer);
Matrix matrix_hadamard_product(const Matrix A, const Matrix B, double *buffer);

Matrix matrix_transpose(const Matrix A, double *buffer);
Matrix matrix_scalar_product(const Matrix A, double scalar, double *buffer);

double matrix_grand_sum(const Matrix A);

/*=========================================================
  Functional operations
=========================================================*/

Matrix matrix_apply_func(const Matrix A,
                         double (*func)(double),
                         double *buffer);

#endif /* MATRIX_MATHS_H */
