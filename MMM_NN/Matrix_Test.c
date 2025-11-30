#include "Matrix_maths.c"

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