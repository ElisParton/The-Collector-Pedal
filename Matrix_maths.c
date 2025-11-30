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

void matrix_multiplication(Matrix* C, const Matrix* A, const Matrix* B) {
    assert(A != NULL);
    assert(B != NULL);
    assert(C != NULL);
    assert(A->cols == B->rows);   
    assert(C->rows == A->rows);   
    assert(C->cols == B->cols);   

    for (int r = 0; r < A->rows; r++) {
        for (int c = 0; c < B->cols; c++) {

            double sum = 0.0;

            for (int k = 0; k < A->cols; k++) {
                sum += get(A, r, k) * get(B, k, c);
            }

            set(C, r, c, sum);
        }
    }
}
