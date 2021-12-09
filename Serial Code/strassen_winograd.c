//%%writefile Serial.c
//%%writefile Serial.c

#include<stdio.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>

double** createZeroMatrix(int);
double** strassensMultRec(double **, double**,int n);
double** divide(double ** matrixA,int n, int row,int col);
double ** addMatrix(double**,double**,int);
double** subMatrix(double**,double**,int);
void compose(double**,double**,int,int,int);
void printMatrix(double **, int);


// strassen_recursive_algorithm using D & C

double** strassensMultRec(double ** matrixA, double** matrixB,int n){
	double ** result = createZeroMatrix(n);
	if(n>1) {
		//Divide the matrix suppos n=4(4*4 matrix)
		
		/*      0 1 2 3 
		    ---|---------
			  0|2 3 4 5
			  1|3 5 6 7 
			  2|4 4 4 4
			  3|4 2 4 5
			   
		*/
		double ** a11 = divide(matrixA, n, 0, 0); // it'll return half sized matrix of whole matrix
		double ** a12 = divide(matrixA, n, 0, (n/2)); //passing 2 as 2nd parameter beacuse a12 starts from column 2 
		double ** a21 = divide(matrixA, n, (n/2), 0); 
		double ** a22 = divide(matrixA, n, (n/2), (n/2)); 
			
		double ** b11 = divide(matrixB, n, 0, 0);
		double ** b12 = divide(matrixB, n, 0, n/2);
		double ** b21 = divide(matrixB, n, n/2, 0);
		double ** b22 = divide(matrixB, n, n/2, n/2);
		
		//Recursive call for Divide and Conquer

		double **s1=addMatrix(a21,a22,n/2);
		double **s2=subMatrix(s1,a11,n/2);
		double **s3=subMatrix(a11,a21,n/2);
		double **s4=subMatrix(a12,s2,n/2);
		double **s5=subMatrix(b12,b11,n/2);
		double **s6=subMatrix(b22,s5,n/2);
		double **s7=subMatrix(b22,b12,n/2);
		double **s8=subMatrix(s6,b21,n/2);
		
		double** m1= strassensMultRec(s2,s6,n/2);
		double** m2= strassensMultRec(a11,b11,n/2);
		double** m3= strassensMultRec(a12,b21,n/2);
		double** m4= strassensMultRec(s3,s7,n/2);
		double** m5= strassensMultRec(s1,s5,n/2);
		double** m6= strassensMultRec(s4,b22,n/2);
		double** m7= strassensMultRec(a22,s8,n/2);

		double **v1=addMatrix(m1,m2,n/2);
		double **v2=addMatrix(v1,m4,n/2);

		double** c11 = addMatrix(m2,m3,n/2);
		double** c12 = addMatrix(addMatrix(v1,m5,n/2),m6,n/2);
		double** c21 = subMatrix(v2,m7,n/2);
		double** c22 = addMatrix(v2,m5,n/2);

		//Compose the matrix
		compose(c11,result,0,0,n/2);
		compose(c12,result,0,n/2,n/2);
		compose(c21,result,n/2,0,n/2);
		compose(c22,result,n/2,n/2,n/2);
	} 
	else {
		//This is the terminating condition for recurssion.
		result[0][0]=matrixA[0][0]*matrixB[0][0];
	}
	return result;
}

//divide function

double** divide(double ** matrix,int n, int row,int col) {
	int n_new=n/2; // suppose n=8 then will divide it into half n_new=4	
	
	double ** array = createZeroMatrix(n_new);	
	int i,j,r=row,c=col;
	for(i = 0;i < n_new; i++) {
		c=col;
   	 	for(j = 0; j < n_new; j++) {
        		array[i][j] = matrix[r][c];
			c++;
    		}
		r++;
	}
	return array;
}

//add matrix function

double** addMatrix(double** matrixA,double** matrixB,int n){
	double ** res = createZeroMatrix(n);
	int i,j;	
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			res[i][j]=matrixA[i][j]+matrixB[i][j];
	
	return res;
}


// compose function

void compose(double** matrix,double** result,int row,int col,int n){
	int i,j,r=row,c=col;
	for(i=0;i<n;i++){
		c=col;
		for(j=0;j<n;j++){
			result[r][c]=matrix[i][j];	
			c++;	
		}
		r++;
	}
}

double ** createZeroMatrix(int n){
	double ** array = (double**)malloc(n*sizeof(double *));	
	int i,j;
	for(i = 0;i < n; i++) {
	    	array[i] = (double*)malloc(n*sizeof(double));
   	 	for(j = 0; j < n; j++) {
	        	array[i][j] = 0.0;
	    	}
	}
	return array;
} 

double** subMatrix(double** matrixA,double** matrixB,int n){
	double ** res = createZeroMatrix(n);
	int i,j;	
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			res[i][j]=matrixA[i][j]-matrixB[i][j];
	
	return res;
}

void initialize(double **b,double **c,int size){
    //initialise arrays
    int i,j;
    for (i=0;i<size;i++)
    {
        for(j=0;j<size;j++)
        {
            b[i][j]=3;
            c[i][j]=2;
        }
    }
    
}

// Here in compute function we'll call strassen divide and conquer approach

void compute(double **a,double **b,double **c,int runs,int size){
    //function for mult/add

   int rn,i,k,j,l;
    //for(rn=0;rn<runs;rn++)
    //{
      	a = strassensMultRec(b,c,size);
		//printMatrix(a,size);
	        
    //}
    if(1==2)
        printf("dummy code");
}
/*
void printMatrix(double ** matrix,int n) {
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			printf("   %.2f   ",matrix[i][j]);
		}
		printf("\n");
	}
}
*/
int main()
{
int minSize=pow(2,3);
int maxSize=pow(2,8);
int size,i,j,runs,k=3;
clock_t start,end;
double *a,*b,*c,walltime,throughput;

for(size=minSize;size<=maxSize;size*=2,k++){

        // since we are using double pointer in strassen algorithm of github we've allocated dynemic memory using double pointers

        double** a = (double**)malloc(size * sizeof(double*)); // a[4][4]
        for (i = 0; i < size; i++)
            a[i] = (double*)malloc(size * sizeof(double));

        double** b = (double**)malloc(size * sizeof(double*));
        for (i = 0; i < size; i++)
            b[i] = (double*)malloc(size * sizeof(double));

        double** c = (double**)malloc(size * sizeof(double*));
        for (i = 0; i < size; i++)
            c[i] = (double*)malloc(size * sizeof(double));

        

        initialize(b,c,size);
 
        runs=maxSize/size;

        start=clock();
        	compute(a,b,c,runs,size);
        end=clock();

        walltime=(end-start)/(double)CLOCKS_PER_SEC;
        throughput = (maxSize*sizeof(double))/walltime;
        printf("%d,%lf,%lf\n",k,walltime,throughput/pow(10,9));
        free(a);
        free(b);
        free(c);
}
return 0;
}
