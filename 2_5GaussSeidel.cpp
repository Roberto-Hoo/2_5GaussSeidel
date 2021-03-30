#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>

using namespace std;

// random +/- 1
double randsig(gsl_rng *rng);

int n = 16; // Tamanho da matriz
int tmax = 50; // Quantidade de interações
double tol = 1e-20; // O erro é menor que 10^-20
char caracter;
gsl_matrix *A = gsl_matrix_alloc(n, n);
gsl_vector *b = gsl_vector_alloc(n);
gsl_vector *x = gsl_vector_alloc(n);
gsl_vector *x0 = gsl_vector_alloc(n);
// gerador randômico
gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

int main(int argc, char *argv[]) {

    gsl_rng_set(rng, time(NULL));
    //gsl_rng_set(rng,10);
    // Inicializacao
    // Matriz estritamente diagonal dominante
    printf("Inicializacao ... \n");
    double sig;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) {
            double aux = gsl_rng_uniform(rng);
            gsl_matrix_set(A, i, j, randsig(rng) * aux);
            s += aux;
        }
        gsl_matrix_set(A, i, i, randsig(rng) * s);
        gsl_vector_set(b, i, randsig(rng) * gsl_rng_uniform(rng));
        gsl_vector_set(x0, i, randsig(rng) * gsl_rng_uniform(rng));
    }
    // Print the values of A using GSL print functions
    /*
    cout << "\nA = \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("A(%d,%d) = %g\n", i, j, gsl_matrix_get(A, i, j));
            printf("%6.3f ", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }

    // Print the values of b using GSL print functions
    cout << "\nb-Transposta = \n";
    for (int i = 0; i < n; i++) {
        printf("%13.10f ", gsl_vector_get(b, i));
    }

    // Print the values of b using GSL print functions
    cout << "\nx0-Transposta = \n";
    for (int i = 0; i < n; i++) {
        printf("%6.3f ", gsl_vector_get(x0, i));
    }
    */
    printf("\nfeito.");

    // Random Gauss-Seidel
    for (int t = 0; t < tmax; t++) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double s = 0;
            // Os termos à esquerda da diagonal
            for (int j = 0; j < i; j++)
                s += gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
            // Os termos à direita da diagonal
            for (int j = i + 1; j < n; j++)
                s += gsl_matrix_get(A, i, j) * gsl_vector_get(x0, j);

            gsl_vector_set(x, i, (gsl_vector_get(b, i) - s) / gsl_matrix_get(A, i, i));

        }
        // criterio de parada
        // ||x-x0||_2 < tol
        //  gsl_blas_daxpy(alpha, x, y)
        //  These functions compute the sum y := alpha * x + y for the vectors x and y
        gsl_blas_daxpy(-1.0, x, x0);

        /* double gsl_blas_dnrm2(const gsl_vector * x)
         * These functions compute the Euclidean norm ||x||_2 = sqrt{sum x_i^2} of the vector x.*/
        double e = gsl_blas_dnrm2(x0);
        printf("\nIter. %2d:  %22.19e = %22.19f", t, e, e);
        if (e < tol)
            break;
        /*
         *  int gsl_vector_memcpy(gsl_vector * dest, const gsl_vector * src)
         *  This function copies the elements of the vector src into the vector dest.
         *  The two vectors must have the same length.
        */
        gsl_vector_memcpy(x0, x); // x0 := x;
    }

    // Print the values of x
    cout << "\nx-Transposta = \n";
    for (int i = 0; i < n; i++) {
        printf("%6.3f ", gsl_vector_get(x, i));
    }

    // Print the values of Ax and Ax-b
    cout << "\nAx = \n";
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) {
            s = s + gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
        }
        printf("\nb[%2d]=%22.19f   Ax[%2d]=%22.19f  Erro(Ax-b)= %22.19f",
               i,gsl_vector_get(b, i), i, s, s - gsl_vector_get(b, i));
    }


    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_vector_free(x0);
    gsl_rng_free(rng);

    cout << "\n\n Tecle uma tecla e apos Enter para finalizar...\n";
    cin >> caracter;

    return 0;
}

/*
 * Retorna -1 se gsl_rng_uniform(rng) >= 0.5
 * e 1 se 0 =< gsl_rng_uniform(rng) < 0.5
 */
double randsig(gsl_rng *rng) {
    double signal = 1.0;
    if (gsl_rng_uniform(rng) >= 0.5)
        signal = -1.0;
    return signal;
}