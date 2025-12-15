/**
 * @file Matrix_maths.c
 * @brief Matrix mathematics operations and utilities
 *
 * This module provides functions for performing various mathematical operations
 * on matrices, including but not limited to:
 * - Matrix creation and initialization
 * - Matrix arithmetic operations (addition, subtraction, multiplication)
 * - Matrix printing for debugging purposes
 *
 * @author Elis Parton
 * @date 30/11/25
 * @version 1.0
 *
 * @note Manual memory management is used. Ensure proper allocation and
 *       deallocation of matrix structures to avoid memory leaks.
 */
#include <stdio.h>
#include <assert.h>
/**A struct for a 2D Matrix*/
typedef struct Matrix
{
    double *entries; /// Data in the Matrix
    int rows;        /// Rows in the Matrix
    int cols;        /// Collumns in the Matrix
} Matrix;

//--------------------
// matrix_create
//--------------------
// Construct a Matrix object that wraps an existing contiguous memory buffer.
//
// Parameters:
//  - row: Number of rows in the matrix.
//  - col: Number of columns in the matrix.
//  - buffer: Pointer to a contiguous memory region that will hold the matrix entries.
//
// Returns:
//  - A Matrix struct whose entries field points directly to the supplied buffer.
//
// Preconditions:
//  - Caller must ensure buffer is non-NULL.
//  - row > 0 and col > 0.
//  - The caller owns the buffer memory and is responsible for its lifetime.
//
// Notes:
//  - No dynamic allocation occurs; this function simply wraps the provided memory.
//  - No deeper validation beyond simple asserts is performed.
Matrix matrix_create(int row, int col, double *buffer)
{
    assert(buffer != NULL);
    assert(col > 0);
    assert(row > 0);

    Matrix m;
    m.entries = buffer;
    m.rows = row;
    m.cols = col;
    return m;
}

//--------------------
// matrix_get_entry
//--------------------
// Retrieve the value stored at entry (r, c) of matrix m.
//
// Parameters:
//  - m: Matrix to read from.
//  - r: Row index (0-based).
//  - c: Column index (0-based).
//
// Returns:
//  - The double-precision value at m.entries[r * m.cols + c].
//
// Preconditions:
//  - Caller must ensure r and c are within bounds.
//  - m must be a valid Matrix with initialized entries.
//
// Notes:
//  - Implemented as a static inline for minimal overhead.
//  - No bounds checking is performed; out-of-range indices yield undefined behavior.
static inline double matrix_get_entry(const Matrix m, int r, int c)
{
    return m.entries[r * m.cols + c];
}

//--------------------
// matrix_set_entry
//--------------------
// Store a value v into entry (r, c) of matrix m.
//
// Parameters:
//  - m: Matrix to modify.
//  - r: Row index (0-based).
//  - c: Column index (0-based).
//  - v: Value to assign into the matrix at (r, c).
//
// Returns:
//  - Nothing (void).
//
// Preconditions:
//  - Caller must ensure r and c are within valid bounds.
//  - m must be a valid Matrix whose entries pointer is modifiable.
//
// Notes:
//  - Implemented as a static inline for efficiency.
//  - No bounds checks are performed; behavior is undefined if indices are invalid.
static inline void matrix_set_entry(Matrix m, int r, int c, double v)
{
    m.entries[r * m.cols + c] = v;
}

Matrix matrix_set_all(Matrix A, double value)
{
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            matrix_set_entry(A, r, c, value);
        }
    }
    return A;
}

Matrix matrix_slice(const Matrix A, int row_start, int row_end, int col_start, int col_end, double *buffer)
{
    assert(row_start >= 0 && row_start < A.rows);
    assert(row_end > row_start && row_end <= A.rows);
    assert(col_start >= 0 && col_start < A.cols);
    assert(col_end > col_start && col_end <= A.cols);

    int new_rows = row_end - row_start;
    int new_cols = col_end - col_start;
    Matrix S = matrix_create(new_rows, new_cols, buffer);

    for (int r = 0; r < new_rows; r++)
    {
        for (int c = 0; c < new_cols; c++)
        {
            double val = matrix_get_entry(A, row_start + r, col_start + c);
            matrix_set_entry(S, r, c, val);
        }
    }
    return S;
}

//--------------------
// matrix_print
//--------------------
// Print the contents of matrix A to stdout in a human-readable format.
//
// Parameters:
//  - A: Matrix to display.
//
// Returns:
//  - Nothing (void).
//
// Preconditions:
//  - A must be a valid Matrix with a non-NULL entries pointer.
//
// Notes:
//  - Intended primarily for debugging and inspection.
//  - Prints row by row using fixed %f formatting for each entry.
//  - Output layout is "Matrix (rows x cols)" followed by raw numeric entries.
void matrix_print(Matrix A)
{
    printf("Matrix (%d x %d):\n", A.rows, A.cols);
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double val = matrix_get_entry(A, r, c);
            printf("%f ", val);
        }
        printf("\n");
    }
    printf("\n");
}

//--------------------
// matrix_multiplication
//--------------------
// Compute the matrix product C = A * B and return C.
//
// Parameters:
//  - A: Left-hand operand matrix.
//  - B: Right-hand operand matrix.
//  - buffer: Pointer to a contiguous memory region used by matrix_create for result storage.
//
// Returns:
//  - Matrix C holding the product with dimensions A.rows x B.cols.
//
// Preconditions:
//  - Caller must ensure A.cols == B.rows.
//  - buffer must be large enough to hold A.rows * B.cols doubles.
//  - A and B must be valid Matrix objects with properly initialized entries.
//
// Notes:
//  - No runtime checks/asserts are performed here; behavior is undefined if preconditions are violated.
//  - Uses a straightforward triple-loop (O(n^3) in general) multiplication accumulating into a local double sum.
//
Matrix matrix_multiplication(const Matrix A, const Matrix B, double *buffer)
{
    // assert(A != NULL);
    // assert(B != NULL);
    // assert(A->cols == B->rows);

    Matrix C = matrix_create(A.rows, B.cols, buffer);

    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < B.cols; c++)
        {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++)
            {
                sum += matrix_get_entry(A, r, k) * matrix_get_entry(B, k, c);
            }
            matrix_set_entry(C, r, c, sum);
        }
    }
    return C;
}

//--------------------
// matrix_addition
//--------------------
// Compute the element-wise matrix sum C = A + B and return C.
//
// Parameters:
//  - A: First operand matrix.
//  - B: Second operand matrix.
//  - buffer: Pointer to a contiguous memory region used by matrix_create for result storage.
//
// Returns:
//  - Matrix C holding the element-wise sum with dimensions A.rows x A.cols.
//
// Preconditions:
//  - Caller must ensure A.rows == B.rows and A.cols == B.cols.
//  - buffer must be large enough to hold A.rows * A.cols doubles.
//  - A and B must be valid Matrix objects with properly initialized entries.
//
// Notes:
//  - No runtime checks/asserts are performed here; behavior is undefined if preconditions are violated.
//  - Uses a simple double loop to accumulate A[r][c] + B[r][c] directly into the result.
Matrix matrix_addition(const Matrix A, const Matrix B, double *buffer)
{
    // assert(A != NULL);
    // assert(B != NULL);
    // assert(A->cols == B->cols);
    // assert(A->rows == B->rows);

    Matrix C = matrix_create(A.rows, A.cols, buffer);

    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double sum = matrix_get_entry(A, r, c) + matrix_get_entry(B, r, c);
            matrix_set_entry(C, r, c, sum);
        }
    }
    return C;
}

//--------------------
// matrix_subtraction
//--------------------
// Compute the element-wise matrix difference C = A - B and return C.
//
// Parameters:
//  - A: Left operand matrix.
//  - B: Right operand matrix (subtrahend).
//  - buffer: Pointer to a contiguous memory region used by matrix_create for result storage.
//
// Returns:
//  - Matrix C holding the element-wise difference with dimensions A.rows x A.cols.
//
// Preconditions:
//  - Caller must ensure A.rows == B.rows and A.cols == B.cols.
//  - buffer must be large enough to hold A.rows * A.cols doubles.
//  - A and B must be valid Matrix objects with properly initialized entries.
//
// Notes:
//  - No runtime checks/asserts are performed here; behavior is undefined if preconditions are violated.
//  - Performs a direct element-wise subtraction A[r][c] - B[r][c] using a simple double loop.
Matrix matrix_subtraction(const Matrix A, const Matrix B, double *buffer)
{

    // assert(A != NULL);
    // assert(B != NULL);
    // assert(A->cols == B->cols);
    // assert(A->rows == B->rows);

    Matrix C = matrix_create(A.rows, A.cols, buffer);

    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double difference = matrix_get_entry(A, r, c) - matrix_get_entry(B, r, c);
            matrix_set_entry(C, r, c, difference);
        }
    }
    return C;
}

Matrix matrix_outer_product(const Matrix A, const Matrix B, double *buffer)
{
    assert(A.cols == 1);
    assert(B.cols == 1);

    Matrix C = matrix_create(A.rows, B.rows, buffer);

    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < B.rows; c++)
        {
            double product = matrix_get_entry(A, r, 0) * matrix_get_entry(B, c, 0);
            matrix_set_entry(C, r, c, product);
        }
    }
    return C;
}

Matrix matrix_hadamard_product(const Matrix A, const Matrix B, double *buffer)
{
    assert(A.cols == B.cols);
    assert(A.rows == B.rows);

    Matrix C = matrix_create(A.rows, A.cols, buffer);

    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double value = matrix_get_entry(A, r, c) * matrix_get_entry(B, r, c);
            matrix_set_entry(C, r, c, value);
        }
    }
    return C;
}

Matrix matrix_apply_func(const Matrix A, double (*func)(double), double *buffer)
{
    Matrix out = matrix_create(A.rows, A.cols, buffer);
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double val = matrix_get_entry(A, r, c);
            double result = func(val);
            matrix_set_entry(out, r, c, result);
        }
    }
    return out;
}

double matrix_grand_sum(const Matrix A)
{
    double sum = 0.0;
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            sum += matrix_get_entry(A, r, c);
        }
    }
    return sum;
}

Matrix matrix_transpose(const Matrix A, double *buffer)
{
    Matrix T = matrix_create(A.cols, A.rows, buffer);
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double val = matrix_get_entry(A, r, c);
            matrix_set_entry(T, c, r, val);
        }
    }
    return T;
}

Matrix matrix_scalar_product(const Matrix A, double scalar, double *buffer)
{
    Matrix S = matrix_create(A.rows, A.cols, buffer);
    for (int r = 0; r < A.rows; r++)
    {
        for (int c = 0; c < A.cols; c++)
        {
            double val = matrix_get_entry(A, r, c) * scalar;
            matrix_set_entry(S, r, c, val);
        }
    }
    return S;
}