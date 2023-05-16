 #include <stdio.h>

 #include <stdlib.h>

 #include <time.h>

 #include <string.h>

 #define min(x, y)(((x) < (y)) ? (x) : (y))

 //NetLib BLAS:
 //#include <cblas.h>
 /* BLAS from GNU Scientific Library (GSL): */
 //#include <gsl/gsl_cblas.h>
 /* openblas */
 //#include <openblas/cblas.h>
 /* Lapacke */
 //#include <lapacke/lapacke.h>
 // Intel Math Kernel Library
 #include <mkl.h>

 #include <omp.h>

 void drukuj_macierz(int m, int k, double * M, char N) {
   printf(" Lewy górny róg macierzy %c: \n", N);
   int i, j;
   for (i = 0; i < min(m, 4); i++) {
     for (j = 0; j < min(k, 4); j++) {
       printf("%12.0f", M[j + i * k]);
     }
     printf("\n");
   }
 }

 int main(int argc, char * argv[]) {
   double * A, * B, * C;
   int m, n, k, i, j, l;
   double alpha = 1.0, beta = 0.0;
   double t;

   if (argc == 4) {
     m = atoi(argv[1]);
     k = atoi(argv[2]);
     n = atoi(argv[3]);
   } else {
     m = 1000, k = 2000, n = 3000;
     printf("Poprawne wywolanie mnozenia macierzy c[m][n]=a[m][k] * b[k][n]: ./mnozenie_macierzy_blas m k n\n");
     exit(1);
   }
   printf("Funkcja dgemm oblicza C[m][n]=alpha*A[m][k]*B[k][n]+beta*C[m][n]\n");
   printf("Alokowanie pamięci dla macierzy C[%d][%d] = A[%d][%d] * B[%d][%d]\n", m, n, m, k, k, n);
   A = (double * ) malloc(m * k * sizeof(double));
   B = (double * ) malloc(k * n * sizeof(double));
   C = (double * ) calloc(m * n, sizeof(double));

   //printf("Wymagania dla pamieci: %zu GB\n", (m*k+k*n+m*n)*sizeof(double));
   printf("Wymagania dla pamieci: %lf GB\n", (m * k + k * n + m * n) * sizeof(double) / 1000000000.0);

   printf("Inicjalizowanie macierzy\n");
   #pragma omp parallel num_threads(omp_get_num_procs()) {
     #pragma omp
     for private(i)
     for (i = 0; i < (m * k); i++)
       A[i] = (double)(1);
     #pragma omp
     for private(i)
     for (i = 0; i < (k * n); i++)
       B[i] = (double)(1);
     #pragma omp
     for private(i)
     for (i = 0; i < (m * n); i++)
       C[i] = 0.0;
   }

   printf("\nObliczenia mnożenia macierzy z wykorzystaniem funkcji BLAS: dgemm\n");
   t = omp_get_wtime();
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
   t = omp_get_wtime() - t;
   printf("Czas obliczeń (1 watek BLAS) = %lf s\n\n", t);

   drukuj_macierz(m, k, A, & apos; A & apos;);
   drukuj_macierz(k, n, B, & apos; B & apos;);
   drukuj_macierz(m, n, C, & apos; C & apos;);

   #pragma omp parallel
   for private(i)
   for (i = 0; i < (m * n); i++)
     C[i] = 0.0;

   printf("\nObliczenia mnożenia macierzy z wykorzystaniem OpenMP i %d wątków\n", omp_get_num_procs());
   t = omp_get_wtime();
   #pragma omp parallel
   for private(i, j, l) num_threads(omp_get_num_procs())
   for (i = 0; i < m; i++)
     for (j = 0; j < k; j++)
       for (l = 0; l < n; l++)
         C[l + i * n] += A[j + i * k] * B[l + j * n];
   t = omp_get_wtime() - t;
   printf("Czas obliczeN (%d watków OpenMP) = %lfs\n\n", omp_get_num_procs(), t);

   drukuj_macierz(m, k, A, & apos; A & apos;);
   drukuj_macierz(k, n, B, & apos; B & apos;);
   drukuj_macierz(m, n, C, & apos; C & apos;);

   #pragma omp parallel
   for private(i)
   for (i = 0; i < (m * n); i++)
     C[i] = 0.0;

   printf("\nObliczenia mnożenia macierzy ze wzorów Cauchy&apos;ego - 1 wątek\n");
   t = omp_get_wtime();
   for (i = 0; i < m; i++)
     for (j = 0; j < k; j++)
       for (l = 0; l < n; l++)
         C[l + i * n] += A[j + i * k] * B[l + j * n];
   t = omp_get_wtime() - t;
   printf("Czas obliczeń (1 watek) = %lf s\n\n", t);

   drukuj_macierz(m, k, A, & apos; A & apos;);
   drukuj_macierz(k, n, B, & apos; B & apos;);
   drukuj_macierz(m, n, C, & apos; C & apos;);

   printf("\n Zwalniam pamięć\n");
   free(A);
   free(B);
   free(C);

   return 0;
 }