#include <stdio.h>

/**A struct for a 2D Matrix*/
typedef struct Matrix {
    double* entries; /// Data in the Matrix
    int rows; /// Rows in the Matrix
    int cols  /// Collumns in the Matrix
} Matrix;
/**A function to create a matrix from a buffer*/
Matrix create_matrix(int row, int col, double* buffer) {
    if (row * col != sizeof(buffer)) {
        // Handle error: buffer size does not match specified dimensions
    }
    Matrix m;
    m.entries = buffer;
    m.rows = row;
    m.cols = col;
    return m;
}
Matrix multiply_matrices(Matrix* a, Matrix* b) {
}