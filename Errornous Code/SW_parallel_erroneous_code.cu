//This is erroneous parallel implementation code  that we tried initially
//It's giving correct output for 2*2, wrong output for 4*4 and crash for 8*8 matrix

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include<stdlib.h>
#include<math.h>
#include <stdio.h>
#define BLOCK_SIZE  4
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); getchar(); }
    }
}

__device__ double* createZeroMatrix(int);
__device__ double* strassensMultRec(double *, double*, double*,int );
__device__ double* divide(double * matrixA,int n, int row,int col);
__device__ double * addMatrix(double*,double*,int);
__device__ double* subMatrix(double*,double*,int);
__device__ void compose(double*,double*,int,int,int,int );
__global__ void compute(double*, double*,double*,double* ,int ,int );

// strassen_recursive_algorithm using D & C
// A = B * C
__device__ double* strassensMultRec(double* matrixA, double* matrixB, double* matrixT,int n){

    
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
if(i*n+j==0){
    
  //probably we can use matrixA instead of result 
	double * result = createZeroMatrix(n);

	if(n>1) {
		//Divide the matrix suppos n=4(4*4 matrix)
		
		/*      0 1 2 3 
		    ---|---------
			  0|2 3 4 5
			  1|3 5 6 7 
			  2|4 4 4 4
			  3|4 2 4 5
			   
		*/
   
		double * a11 = divide(matrixA, n, 0, 0); // it'll return half sized matrix of whole matrix
    
		double * a12 = divide(matrixA, n, 0, (n/2)); //passing 2 as 2nd parameter beacuse a12 starts from column 2 
    
		double * a21 = divide(matrixA, n, (n/2), 0); 
    
		double * a22 = divide(matrixA, n, (n/2), (n/2)); 
    
    //printf("\n data after divided: %1.lf %1.lf %1.lf %1.lf \n",a11[0],a12[0],a21[0],a22[0]);
/*
    matrixT[i*n+j] = a11[i*n+j];
    matrixT[i*n+j] = a12[i*n+j];
    matrixT[i*n+j] = a21[i*n+j];
    matrixT[i*n+j] = a22[i*n+j];*/
    
    
		double * b11 = divide(matrixB, n, 0, 0);
		double * b12 = divide(matrixB, n, 0, n/2);
		double * b21 = divide(matrixB, n, n/2, 0);
		double * b22 = divide(matrixB, n, n/2, n/2);
		
		//Recursive call for Divide and Conquer

		double *s1=addMatrix(a21,a22,n/2);
		double *s2=subMatrix(s1,a11,n/2);
		double *s3=subMatrix(a11,a21,n/2);
		double *s4=subMatrix(a12,s2,n/2);
		double *s5=subMatrix(b12,b11,n/2);
		double *s6=subMatrix(b22,s5,n/2);
		double *s7=subMatrix(b22,b12,n/2);
		double *s8=subMatrix(s6,b21,n/2);
		
		double* m1= strassensMultRec(s2,s6,matrixT,n/2);
		double* m2= strassensMultRec(a11,b11,matrixT,n/2);
		double* m3= strassensMultRec(a12,b21,matrixT,n/2);
		double* m4= strassensMultRec(s3,s7,matrixT,n/2);
		double* m5= strassensMultRec(s1,s5,matrixT,n/2);
		double* m6= strassensMultRec(s4,b22,matrixT,n/2);
		double* m7= strassensMultRec(a22,s8,matrixT,n/2);

		double *v1=addMatrix(m1,m2,n/2);
		double *v2=addMatrix(v1,m4,n/2);

		double* c11 = addMatrix(m2,m3,n/2);
		double* c12 = addMatrix(addMatrix(v1,m5,n/2),m6,n/2);
		double* c21 = subMatrix(v2,m7,n/2);
		double* c22 = addMatrix(v2,m5,n/2);

		//Compose the matrix
		compose(c11,result,0,0,n/2,n);
		compose(c12,result,0,n/2,n/2,n);
		compose(c21,result,n/2,0,n/2,n);
		compose(c22,result,n/2,n/2,n/2,n);
	} 
	else {
		//This is the terminating condition for recurssion.
		result[0*n+0]=matrixA[0*n+0] * matrixB[0*n+0];
	}

  
  
  
      //printf("\n FINALLY: (%d *%d)+ %d = %d |||| %1.lf %1.lf %1.lf %1.lf ",i,n,j,i*n+j,result[0],result[1],result[2],result[3]);
      for(int x=0;x<(n*n);x++){
          matrixT[x]=result[x];
      
       //printf("\n FINALLY: (%d *%d)+ %d = %d |||| %1.lf %1.lf %1.lf %1.lf ",i,n,j,i*n+j,result[0],result[1],result[2],result[3]);
  }

	return result;
}
}

//divide function
__device__ double* divide(double * matrix,int n, int row,int col) {
	  int n_new=n/2; // suppose n=8 then will divide it into half n_new=4	

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	  double * array = createZeroMatrix(n_new);	
	  int r_base=row,c_base=col;
	
    //array[i*n_new+j] = matrix[r_base+i][c_base+j]; // we need to make formula for right side
    
    if((i * n_new+j)<(n_new * n_new)){
    array[i * n_new+j] = matrix[(r_base+i)  * n + (c_base+j)];
    printf("\n divide: %d %d %d %d %1.lf",n_new,n,i*n_new+j,(r_base+i)  * n + (c_base+j),matrix[(r_base+i)  * n + (c_base+j)]);
    }

	return array;
}

//add matrix function
__device__  double* addMatrix(double* B,double* C,int n){
	  double* res = createZeroMatrix(n);
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

     if (i < n && j < n)
    {
      res[i* n +j]=B[i* n +j]+C[i* n +j];
      printf("\n addMatrix: %d %d : %1.lf = %1.lf + %1.lf",n, i*n + j, res[i*n + j], B[i*n + j], C[i*n + j]);
    }

    return res;
}

// compose function
__device__ void compose(double* m1,double* result,int row,int col,int n_m1,int n_result){
	int r_base=row,c_base=col;  
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	
    //result[r_base+i][c_base+j]=matrix[i*n+j]; // left side need to be adjusted

    //probably we can use matrixA instead of result 

    if (i < n_m1 && j < n_m1)
    {
    result[(r_base+i)  * n_result + (c_base+j)] = m1[i*n_m1+j];
    printf("\n compose:  ::(%d+%d)*2 + (%d+%d) = %d :::: (%d * %d) + %d = %d : %1.lf = %1.lf ",r_base,n_result,c_base,j,((r_base+i)  * n_result + (c_base+j)),i,n_m1,j, i*n_m1 + j, result[(r_base+i)  * n_result + (c_base+j)], m1[i*n_m1 + j]);
    }
}

__device__  double* createZeroMatrix(int n){
	double* array = (double*)malloc(n*n*sizeof(double));	

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
     if (i < n && j < n){
        array[i*n + j] = 9.0;
     }
     
	  return array;
} 

__device__  double* subMatrix(double* B,double* C,int n){
	  double* res = createZeroMatrix(n);
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

     if (i < n && j < n)
    {
      res[i* n +j]=B[i* n +j]-C[i* n +j];
      printf("\n subMatrix: %d %d : %1.lf = %1.lf - %1.lf",n,i*n + j, res[i*n + j], B[i*n + j], C[i*n + j]);
    }
    return res;
}

void initialize(double *b,double *c,double *t,int size){
    int i,j;
    for (i=0;i<size;i++)
    {
        for(j=0;j<size;j++)
        {
          b[i*size+j]=3.0;
          c[i*size+j]=3.0;
          t[i*size+j]=3.0;
        }
    }
}


__global__ void compute(double *a,double *b,double *c,double *t,int size){
    //function for mult/add
  int n=size;

  //rather temp you can call this only one time by : if (i==0) OR i(i*n+j==0)

  double *tmp=strassensMultRec(b,c,t,size);

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  a[i*n + j]=tmp[i*n + j]; 
  //printf("\n MAIN_F: %1.lf : %1.lf : %1.lf : %1.lf",matrixB[0],matrixB[1],matrixB[2],matrixB[3]);
}

int main() {
    
    double* hostA; // The A matrix
    double* hostB; // The B matrix
    double* hostC; // The output C matrix
    double* deviceA;
    double* deviceB;
    double* deviceC;

    double* hostT;
    double* deviceT;

    int minSize=pow(2,1);
    int maxSize=pow(2,1);
    int size,i,runs,k=1;


    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

 
    for(size=minSize;size<=maxSize;size*=2,k++){

      int tot_allocation_size = size * size * sizeof(double);
      //Dynamically Allocating Memory

      hostA = (double*)malloc(tot_allocation_size); 
      hostB = (double*)malloc(tot_allocation_size); 
      hostC = (double*)malloc(tot_allocation_size); 
      hostT = (double*)malloc(tot_allocation_size); 

      initialize(hostB,hostC,hostT,size);
          
      cudaMalloc((void**)&deviceA, tot_allocation_size);
      cudaMalloc((void**)&deviceB, tot_allocation_size);
      cudaMalloc((void**)&deviceC, tot_allocation_size);
      cudaMalloc((void**)&deviceT, tot_allocation_size);

      cudaEventRecord(start);

      //@@ Copy memory to the GPU here
          
      cudaMemcpy(deviceB, hostB, tot_allocation_size, cudaMemcpyHostToDevice);
      cudaMemcpy(deviceC, hostC, tot_allocation_size, cudaMemcpyHostToDevice);


      //@@ Initialize the grid  and block dimensions here
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
      dim3 dimGrid(ceilf(size/ (double)BLOCK_SIZE), ceilf(size / (double)BLOCK_SIZE));
    
      compute <<< dimGrid, dimBlock >>> (deviceA, deviceB, deviceC,deviceT, size);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gpuErrchk(cudaMemcpy(hostA, deviceA, tot_allocation_size, cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(hostT, deviceT, tot_allocation_size, cudaMemcpyDeviceToHost));

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&milliseconds, start, stop);


      
      printf("********TIME: %lf\n", milliseconds /1000);
      
      printf("\n---------A----------\n");
      for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
          printf("%1.lf  ",hostA[i*size+j]);
        }  
        printf("\n");
      }

      printf("\n---------T----------\n");
      for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
          printf("%1.lf  ",hostT[i*size+j]);
        }  
        printf("\n");
      }
      
      //@@ Free the GPU memory here
      cudaFree(deviceA);
      cudaFree(deviceB);
      cudaFree(deviceC);
      free(hostA);
      free(hostB);
      free(hostC);
    }
    return 0;
}