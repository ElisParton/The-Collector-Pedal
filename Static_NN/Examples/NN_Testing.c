#include "../NN_Tools.c"

double times10(double x)
{
    return x * 10.0;
}

int main()
{
    double mat_buffer[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double mat2_buffer[6];
    Matrix A = matrix_create(2, 3, mat_buffer);
    Matrix B = matrix_create(2, 3, mat2_buffer);
    matrix_map_to(A, times10, B);
    matrix_print(B);
    return 0;
}