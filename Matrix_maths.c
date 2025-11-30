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
Matrix matrix_create(int row, int col, double* buffer, int buf_len) {
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

static inline double matrix_get_entry(const Matrix m, int r, int c) {
    return m.entries[r * m.cols + c];
}

static inline void matrix_set_entry(Matrix m, int r, int c, double v) {
    m.entries[r * m.cols + c] = v;
}

void matrix_print(Matrix A) {
    printf("Matrix (%d x %d):\n", A.rows, A.cols);
    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < A.cols; c++) {
            double val = matrix_get_entry(A, r, c);
            printf("%f ", val); 
        }
        printf("\n");
    }
    printf("\n");
}

Matrix matrix_multiplication(const Matrix A, const Matrix B, double* buffer, int buf_len) {
    //assert(A != NULL);
    //assert(B != NULL);
    //assert(A->cols == B->rows);

    Matrix C = matrix_create(A.rows, B.cols, buffer, buf_len);   

    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < B.cols; c++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) {
                sum += matrix_get_entry(A, r, k) * matrix_get_entry(B, k, c);
            }
            matrix_set_entry(C, r, c, sum);
        }
    }
    return C;
}

Matrix matrix_addition(const Matrix A, const Matrix B, double* buffer, int buf_len) {
    //assert(A != NULL);
    //assert(B != NULL);
    //assert(A->cols == B->cols);
    //assert(A->rows == B->rows);

    Matrix C = matrix_create(A.rows, A.cols, buffer, buf_len);

    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < A.cols; c++) {
            double sum = matrix_get_entry(A, r, c) + matrix_get_entry(B, r, c);
            matrix_set_entry(C, r, c, sum);
        }
    }
    return C;
}

Matrix matrix_subtraction(const Matrix A, const Matrix B, double* buffer, int buf_len) {
    
    //assert(A != NULL);
    //assert(B != NULL);
    //assert(A->cols == B->cols);
    //assert(A->rows == B->rows);
    
    
    Matrix C = matrix_create(A.rows, A.cols, buffer, buf_len);

    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < A.cols; c++) {
            double difference = matrix_get_entry(A, r, c) - matrix_get_entry(B, r, c);
            matrix_set_entry(C, r, c, difference);
        }
    }
    return C;
}


int main() 
{
    double a_buf[3*3] = {
    1, 4, 8,
    3, 5, 4,
    12, 3, 18
    };

    double b_buf[3*3] = {
    2, 7, 4,
    3, 5, 9,
    8, 5, 1,
    };

    Matrix A = matrix_create(3, 3, a_buf, 3*3);
    Matrix B = matrix_create(3, 3, b_buf, 3*3);
    matrix_print(A);
    matrix_print(B);
    double c_buf[A.rows * B.cols];
    Matrix C = matrix_multiplication(A, B, c_buf, A.rows * B.cols);
    matrix_print(C);
    double d_buf[A.rows * A.cols];
    Matrix D = matrix_addition(A, B, d_buf, A.rows * A.cols);
    matrix_print(D);
    double e_buf[A.rows * A.cols];
    Matrix E = matrix_subtraction(A, B, e_buf, A.rows * A.cols);
    matrix_print(E);
    
    return 0;
}