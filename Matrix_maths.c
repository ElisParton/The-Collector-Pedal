#include <stdio.h>
#include <assert.h>
#define NDEBUG
/**A struct for a 2D Matrix*/
typedef struct Matrix {
    double* entries; /// Data in the Matrix
    int rows; /// Rows in the Matrix
    int cols; /// Collumns in the Matrix
} Matrix;
/**A function to create a matrix from a buffer.
 * Ensure that the buffer is the correct size before calling this function.
*/
Matrix create_matrix(int row, int col, double* buffer, int buf_len) {
    assert(buffer != NULL);
    assert(col > 0);
    assert(row > 0);
    assert(buf_len == row * col);

    Matrix m;
    m.entries = buffer;
    m.rows = row;
    m.cols = col;
    return m;
}

static inline double get_matrix_entry(const Matrix* m, int r, int c) {
    return m->entries[r * m->cols + c];
}

static inline void set_matrix_entry(Matrix* m, int r, int c, double v) {
    m->entries[r * m->cols + c] = v;
}

Matrix matrix_multiplication(const Matrix* A, const Matrix* B, double* buffer, int buf_len) {
    Matrix C = create_matrix(A->rows, B->cols, buffer, buf_len);
    assert(A != NULL);
    assert(B != NULL);
    assert(A->cols == B->rows);   

    for (int r = 0; r < A->rows; r++) {
        for (int c = 0; c < B->cols; c++) {

            double sum = 0.0;

            for (int k = 0; k < A->cols; k++) {
                sum += get(A, r, k) * get(B, k, c);
            }

            set(C, r, c, sum);
        }
    }
    return C;
}

Matrix matrix_addition()