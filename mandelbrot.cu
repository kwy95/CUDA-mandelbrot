#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "lodepng.h"

#define M 256
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
#define DIE(...) { \
    fprintf(stderr, __VA_ARGS__); \
    exit(EXIT_FAILURE); \
}

void checkPtr(void* ptr);

struct complex {
    REALN real;
    REALN imag;
};
typedef struct complex* Complex;
enum tint {
    Red,
    Green,
    Blue
};
typedef enum tint color;

__host__ __device__
void cAdd(const Complex a, const Complex b, Complex c);
__host__ __device__
void cMult(const Complex a, const Complex b, Complex c);
__host__ __device__
REALN cAbsSqr(Complex z);

__host__ __device__
void nextMandelbrot(Complex z, struct complex c, int n);
__host__ __device__
int iterateMandelbrot(Complex z);

__host__ __device__
char getColor(int iterations, int rgb, int colorMode);

void calculateImage_cpu(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int threads, int offset, int nPixels, int colorMode);
void calculateImage_gpu(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int threads, int offset, int nPixels, int colorMode);
__global__
void calculateImageKernel(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int offset, int colorMode);

/**
 * Código para gerar PNG após o calculo dos valores dos pixels
 *
 * Fonte: arquivo example_encode.c disponivel em https://lodev.org/lodepng/
 */
void encodePNG(const char* filename, const unsigned char* image, unsigned width, unsigned height);

int main(int argc, char** argv) {
    if (argc != 10 && argc != 11) {
        printf("USAGE: %s <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA> [colorMode]\n", argv[0]);
        exit(1);
    }
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Complex z0       = (Complex) malloc(sizeof(struct complex)); checkPtr(z0);
    Complex z1       = (Complex) malloc(sizeof(struct complex)); checkPtr(z1);
            z0->real = atof(argv[1]); z0->imag = atof(argv[2]); // ponto inicial
            z1->real = atof(argv[3]); z1->imag = atof(argv[4]); // ponto final
    int     width    = atoi(argv[5]);
    int     height   = atoi(argv[6]);
    char*   option   = argv[7];
    int     threads  = atoi(argv[8]);
    char*   filename = argv[9];
    int colorMode    = 0;
    if (argc == 11) {
        colorMode    = atoi(argv[10]);
    }

    REALN dx      = ABS(z1->real - z0->real) / width;
    REALN dy      = ABS(z1->imag - z0->imag) / height;
    int   nPixels = (width * height) / world_size; // numero de pixels a ser calculado por cada processo
    int   offset  = world_rank * nPixels;          // local de inicio do calculo da imagem para cada processo
    int   err     = 0;

    unsigned char* image = (unsigned char*) malloc(width * height * 3);
    checkPtr(image);

    if (strcmp(option, "cpu") == 0) calculateImage_cpu(image, width, height, z0, dx, dy, threads, offset, nPixels, colorMode);
    if (strcmp(option, "gpu") == 0) calculateImage_gpu(image, width, height, z0, dx, dy, threads, offset, nPixels, colorMode);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            err |= MPI_Recv(&image[i * nPixels * 3], // Buffer to write to. You must ensure that the message fits here.
                    nPixels * 3,                     // How many elements.
                    MPI_CHAR,                        // Type.
                    i,                               // From which process
                    0,                               // Tag of message. Again, remember that TCP do not guarantee order.
                    MPI_COMM_WORLD,                  // In the context of the entire world.
                    MPI_STATUS_IGNORE                // Ignore the status return.
            );
        }
    } else {
        err |= MPI_Send(&image[offset * 3], // Buffer to send
            nPixels * 3,                    // How many elements.
            MPI_CHAR,                       // Type of element.
            0,                              // For which process
            0,                              // Tag of the message. Remember that TCP do not guarantee order.
            MPI_COMM_WORLD                  // In the context of the entire world.
        );
    }

    if (err)
        DIE("There was an MPI error.\n");

    if (world_rank == 0)
        encodePNG(filename, image, width, height);

    free(image);
    free(z0);
    free(z1);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}

void checkPtr(void* ptr) {
    if (ptr == NULL) {
        printf("Pointer error!\n");
        exit(1);
    }
}

__host__ __device__
void cAdd(const Complex a, const Complex b, Complex c) {
    REALN real = a->real + b->real;
    REALN imag = a->imag + b->imag;
    c->real = real;
    c->imag = imag;
}

__host__ __device__
void cMult(const Complex a, const Complex b, Complex c) {
    REALN real = a->real*b->real - a->imag*b->imag;
    REALN imag = a->real*b->imag + a->imag*b->real;
    c->real = real;
    c->imag = imag;
}

__host__ __device__
REALN cAbsSqr(Complex z) {
    return z->real*z->real + z->imag*z->imag;
}

__host__ __device__
void nextMandelbrot(Complex z, struct complex c, int n) {
    if (n == 0) {
        z->real = 0;
        z->imag = 0;
        return;
    }
    cMult(z, z, z);
    cAdd(z, &c, z);
}
__host__ __device__
int iterateMandelbrot(Complex z) {
    int count    = 0;
    struct complex c = { z->real, z->imag };

    while (count < M) {
        nextMandelbrot(z, c, count);
        if (cAbsSqr(z) > 4) {
            return count;
        }
        count++;
    }

    return -1;
}

__host__ __device__
char getColor(int iterations, color rgb, int colorMode) {
    if (iterations == -1) return 0;
    int m = 1530;
    float hue = (float) iterations / (float) M;
    if (colorMode == 2)
        return iterations % 2 == 0 ? (char) 0 : (char) 255;

    if (rgb == Red) {
        if (colorMode == 0) {//#FD453B #FFA300
            return iterations % 2 == 0 ? (char) 253 : (char) 255;
            return (iterations * 5) % 256;
        }
        if (hue < 1./6) return (char) 255;
        if (hue < 1./3) return (-1)*m*(hue - (1./6)) + 255;
        if (hue < 2./3) return 0;
        if (hue < 5./6) return m*(hue - (2./3));
        return (char) 255;
    } else if (rgb == Green) {
        if (colorMode == 0) {
            return iterations % 2 == 0 ? (char) 69 : (char) 163;
            return (iterations * iterations) % 256;
        }
        if (hue < 1./6) return m*(hue - (0));
        if (hue < 1./2) return (char) 255;
        if (hue < 2./3) return (-1)*m*(hue - (1./2)) + 255;
        return 0;
    } else { // rgb == Blue
        if (colorMode == 0) {
            return iterations % 2 == 0 ? (char) 59 : (char) 0;
            return (iterations * 1337) % 256;
        }
        if (hue < 1./3) return 0;
        if (hue < 1./2) return m*(hue - (1./3));
        if (hue < 5./6) return (char) 255;
        return (-1)*m*(hue - (5./6)) + 255;
    }
}

void calculateImage_cpu(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int threads, int offset, int nPixels, int colorMode) {
    int i;

    #pragma omp parallel for num_threads(threads)
    for(i = offset; i < nPixels + offset; i++) {
        if (i < width * height) {
            int x = i % width;
            int y = (i - x) / width;

            Complex z = (Complex) malloc(sizeof(struct complex));
            z->real = start->real + (dx*x);
            z->imag = start->imag - (dy*y);

            int iterations = iterateMandelbrot(z);
            image[3 * width * y + 3 * x + 0] = getColor(iterations, Red,   colorMode);
            image[3 * width * y + 3 * x + 1] = getColor(iterations, Green, colorMode);
            image[3 * width * y + 3 * x + 2] = getColor(iterations, Blue,  colorMode);
        }// else { continue; }
    }
}

void calculateImage_gpu(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int threads, int offset, int nPixels, int colorMode) {
    unsigned char* d_image;
    Complex d_start;
    cudaMalloc((void**)&d_image, sizeof(unsigned char) * width * height * 3);
    cudaMalloc((void**)&d_start, sizeof(struct complex));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_image, image, 3 * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start, sizeof(struct complex), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    int blocks = threads * (1 + (nPixels) / threads);
    calculateImageKernel<<<blocks,threads>>>(d_image, width, height, d_start, dx, dy, offset, colorMode);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel execution failure");

    cudaMemcpy(image, d_image, 3 * width * height, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    cudaFree(d_image);
    cudaFree(d_start);
}

__global__
void calculateImageKernel(unsigned char* image, int width, int height, Complex start, REALN dx, REALN dy, int offset, int colorMode) {
    int i = blockDim.x*blockIdx.x + threadIdx.x + offset;

    if(i < width * height) {
        int x = i % width;
        int y = (i - x) / width;
        struct complex zEnd = { start->real + (dx*x), start->imag - (dy*y) };
        Complex z = &zEnd;

        int iterations = iterateMandelbrot(z);

        image[3 * width * y + 3 * x + 0] = getColor(iterations, Red,   colorMode);
        image[3 * width * y + 3 * x + 1] = getColor(iterations, Green, colorMode);
        image[3 * width * y + 3 * x + 2] = getColor(iterations, Blue,  colorMode);
    }
}

/**
 * Código para gerar PNG após o calculo dos valores dos pixels
 *
 * Fonte: arquivo example_encode.c disponivel em https://lodev.org/lodepng/
 */
void encodePNG(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    /*Encode the image*/
    unsigned error = lodepng_encode_file(filename, image, width, height, LCT_RGB, 8);

    /*if there's an error, display it*/
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}
