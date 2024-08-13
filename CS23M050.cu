#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void initializeArray(int* array, int value){
    array[threadIdx.x] = value;
}

__device__ long  getDistance(int x1, int y1, int x2, int y2) {
    long  deltaX = x2 - x1;
    long  deltaY = y2 - y1;
    return deltaX * deltaX + deltaY * deltaY;
}

__device__ int canHit(int attacker, int target, int T, int* dxcoord, int* dycoord, int* dhp) {
    long  deltaX = dxcoord[target] - dxcoord[attacker];  
    long  deltaY = dycoord[target] - dycoord[attacker];
    long  currDist = getDistance(dxcoord[attacker], dycoord[attacker], dxcoord[target], dycoord[target]);

    if(dhp[target] <= 0) currDist = LLONG_MAX;

    int newTarget = target;

    for (int i = 0; i < T; i++) {
        if(i == attacker || i == target) continue;
        int crossProduct = deltaX * (dycoord[i] - dycoord[attacker]) - deltaY * (dxcoord[i] - dxcoord[attacker]);
        if (crossProduct == 0 && dhp[i] > 0) {
            if (dxcoord[target] < dxcoord[attacker] && dxcoord[i] < dxcoord[attacker]){
                long newDist = getDistance(dxcoord[attacker], dycoord[attacker], dxcoord[i], dycoord[i]);
                if(newDist < currDist){
                    currDist = newDist;
                    newTarget = i;
                }
            } else if(dxcoord[target] > dxcoord[attacker] && dxcoord[i] > dxcoord[attacker]){
                long newDist = getDistance(dxcoord[attacker], dycoord[attacker], dxcoord[i], dycoord[i]);
                if(newDist < currDist){
                    currDist = newDist;
                    newTarget = i;
                }
            } else if(dxcoord[target] == dxcoord[attacker] && dxcoord[i] == dxcoord[attacker]){
                if(dycoord[target] > dycoord[attacker] && dycoord[i] > dycoord[attacker]){
                   long newDist = getDistance(dxcoord[attacker], dycoord[attacker], dxcoord[i], dycoord[i]);
                   if(newDist < currDist){
                        currDist = newDist;
                        newTarget = i;
                   }
                } else if (dycoord[target] < dycoord[attacker] && dycoord[i] < dycoord[attacker]){
                   long newDist = getDistance(dxcoord[attacker], dycoord[attacker], dxcoord[i], dycoord[i]);
                   if(newDist < currDist){
                      currDist = newDist;
                      newTarget = i;
                   }
                }
            }
        }   
    }
    return newTarget;
}

__global__ void simulateGame(int k, int T, int* dxcoord, int* dycoord, int* dhp, int* dScore, int* gameRunning) {
    if(k % T == 0) return;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int target = -1;
    if (dhp[idx] > 0) {
        target = (idx + k) % T;
        target = canHit(idx, target, T, dxcoord, dycoord, dhp);
        if(dhp[target] <= 0) target = -1;
    }
    __syncthreads();

    if(target != -1){
        atomicAdd(&dScore[idx], 1);
        int ret_val = atomicAdd(&dhp[target], -1);
        if(ret_val==1){
            atomicAdd(gameRunning,-1);
        }
    }    
}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *dxcoord;
    int *dycoord;
    int *dhp;
    int *dScore;

    cudaMalloc(&dxcoord, T*sizeof(int));
    cudaMalloc(&dycoord, T*sizeof(int));
    cudaMalloc(&dhp, sizeof(int) * T);
    cudaMalloc(&dScore, sizeof(int) * T);

    cudaMemcpy(dxcoord, xcoord, T*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dycoord, ycoord, T*sizeof(int), cudaMemcpyHostToDevice);
    initializeArray<<<1, T>>>(dhp, H);
    initializeArray<<<1, T>>>(dScore, 0);

    int *gameRunning;
    cudaHostAlloc((void**)&gameRunning, sizeof(int), cudaHostAllocDefault);
    *gameRunning = T; 

    int round = 1;
    while(*gameRunning > 1){
        simulateGame<<<1, T>>>(round, T, dxcoord, dycoord, dhp, dScore, gameRunning);
        cudaDeviceSynchronize();
        round++;
    }

    cudaMemcpy(score, dScore, T*sizeof(int), cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}