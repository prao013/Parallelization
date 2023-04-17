#include <algorithm>
#include <queue>
#include <set>
#include <cmath>
#include "parallel.h"
#include <cstring>
#include <string.h>

using namespace parlay;
using namespace std;
#define THRESHOLD 10000
#define SIZES_TH 100
#define OLD_TH 1000
#define FILTER_TH 10000
#define FRONTIER_TH 100000

inline bool CAS(int* a, int oldval, int newval) {
    static_assert(sizeof(int) <= 8, "Bad CAS length");
    if (sizeof(int) == 1) {
        uint8_t r_oval, r_nval;
        std::memcpy(&r_oval, &oldval, sizeof(int));
        std::memcpy(&r_nval, &newval, sizeof(int));
        return __sync_bool_compare_and_swap(reinterpret_cast<uint8_t*>(a), r_oval, r_nval);
    } else if (sizeof(int) == 4) {
        uint32_t r_oval, r_nval;
        std::memcpy(&r_oval, &oldval, sizeof(int));
        std::memcpy(&r_nval, &newval, sizeof(int));
        return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>(a), r_oval, r_nval);
    } else { 
        uint64_t r_oval, r_nval;
        std::memcpy(&r_oval, &oldval, sizeof(int));
        std::memcpy(&r_nval, &newval, sizeof(int));
        return __sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>(a), r_oval, r_nval);
    } 
}

void ip_scan(int* In, int n) {
    if (n == 0) return;
    if (n <= THRESHOLD) {
        for (int i = 1; i < n; i++) In[i] += In[i-1];
        return;
    }
    int blocks = ceil(sqrt(n));
    int blockSize = n / blocks;
    if (n % blocks != 0) blockSize++;
    int* offset = new int[blocks];
    offset[0] = 0;
    //cilk_for (size_t i = 1; i < blocks; i++) 
    parallel_for(1,blocks,[&](int i){
        offset[i] = 0;
        for (int j = (i-1)*blockSize; j < i*blockSize; j++) { 
            if (j >= n) break; 
            offset[i] += In[j];
        }
    });
    for (int i = 1; i < blocks; i++) offset[i] += offset[i-1];
    //cilk_for (size_t i = 0; i < blocks; i++)
    parallel_for(0,blocks,[&](int i){
        for (int j = 0; j < blockSize; j++) {
            int index = j+blockSize*i;
            if (index < n) {
                if (j == 0) {
                    In[index] += offset[i];
                }
                else {
                    In[index] += In[index-1];
                }
            }
        }
    });
    delete[] offset;
}


int* filter(int* offset, int* E, int* flag, int curr, int &size) {
    int neigh_size = offset[curr+1] - offset[curr];
    ip_scan(flag, neigh_size);
    size = flag[neigh_size-1];
    int* B = new int[size];
    if (flag[0] == 1) B[0] = E[offset[curr]];
    int fix = offset[curr];
    if (neigh_size < FILTER_TH) { 
        for (int i = offset[curr]; i < offset[curr] + neigh_size; i++) {
            if (flag[i-fix] != flag[i-1-fix])
	    {B[flag[i-fix] - 1] = E[i];}
        }
    }
    else {
       // cilk_for (int i = offset[curr]; i < offset[curr] + neigh_size; i++) 
	parallel_for(offset[curr],offset[curr] + neigh_size,[&](int i){
            if (flag[i-fix] != flag[i-1-fix])
	    { B[flag[i-fix] - 1] = E[i];}
        });
    }

    return B;
}


int* flatten(int** nghs, int* sizes, int &oldSize) {
    int* offset = new int[oldSize];
    //cilk_for(size_t i = 0; i < oldSize; i++) 
    parallel_for(0,oldSize,[&](size_t i){
        offset[i] = sizes[i];
    });
    ip_scan(offset, oldSize);
    int newSize = offset[oldSize-1];
    int* B = new int[newSize];
    if (oldSize < OLD_TH) {
        for(int i = 0; i < oldSize; i++) {
            int ofs;
            if (i == 0) ofs = 0;
            else ofs = offset[i-1];
            if (sizes[i] < SIZES_TH) {
                for(int j = 0; j < sizes[i]; j++) {
                    B[ofs+j] = nghs[i][j];
                }
            }
            else {
               // cilk_for(int j = 0; j < sizes[i]; j++) 
                parallel_for(0,sizes[i],[&](size_t j){
                    B[ofs+j] = nghs[i][j];
                });
            }

            delete[] nghs[i];
        }
    }
    else {
       // cilk_for(int i = 0; i < oldSize; i++) 
        parallel_for(0,oldSize,[&](size_t i){
            int ofs;
            if (i == 0) ofs = 0;
            else ofs = offset[i-1];

            if (sizes[i] < SIZES_TH) {
                for(int j = 0; j < sizes[i]; j++) {
                    B[ofs+j] = nghs[i][j];
                }
            }
            else {
               // cilk_for(int j = 0; j < sizes[i]; j++) 
                parallel_for(0,sizes[i],[&](size_t j){
                    B[ofs+j] = nghs[i][j];
                });
            }

            delete[] nghs[i];
        });
    }

    delete[] nghs;
    delete[] sizes;
    delete[] offset;
    oldSize = newSize;
    return B;
}

int* convertSparseToDense(int n, int* frontier, int fsize) {
    int* new_frontier = new int[n];
   // cilk_for(int i = 0; i < n; i++) 
    parallel_for(0,n,[&](int i){
        new_frontier[i] = 0;
    });
    if (fsize < 1000) { // Granularity control
        for(int i = 0; i < fsize; i++) {
            new_frontier[frontier[i]] = 1;
        }
    }
    else {
       // cilk_for(int i = 0; i < fsize; i++) 
        parallel_for(0,fsize,[&](int i){
            new_frontier[frontier[i]] = 1;
        });
    }

    return new_frontier;
}
int* convertDenseToSparse(int n, int* frontier, int &fsize) {
    ip_scan(frontier, n);
    fsize = frontier[n-1];

    int* new_frontier = new int[fsize];

    if (frontier[0] == 1){ new_frontier[0] = 0;}
   // cilk_for (int i = 0; i < n; i++) 
    	parallel_for(0,n,[&](int i){
        if (frontier[i-1] != frontier[i])
	{ new_frontier[frontier[i] - 1] = i;}
    });
    return new_frontier;
}
int* convert(int n, int* frontier, int &fsize, string currMode) {
    if (currMode == "dense") {
        return convertSparseToDense(n, frontier, fsize);
    }
    else {
        return convertDenseToSparse(n, frontier, fsize);
    }
}

int* edgeMapSparse(int* offset, int* E, int* dist, int* frontier, int &fsize) {
    int** nghs = new int*[fsize];
    int* sizes = new int[fsize];
    //cilk_for(int i = 0; i < fsize; i++) 
    parallel_for(0,fsize,[&](int i){
        int curr = frontier[i];
        int ofs = offset[curr];
        int outSize = offset[curr+1] - offset[curr];
        int* out_flag = new int[outSize];
        for(int j = 0; j < outSize; j++) out_flag[j] = 0;
        if (outSize < 1000) {
            for(int j = offset[curr]; j < offset[curr+1]; j++) {
                int ngh = E[j];
                if (dist[ngh] == -1 && CAS(&dist[ngh], -1, (dist[curr] + 1))) {
                    out_flag[j-ofs] = 1;
                }
            }
        }
        else {
           // cilk_for(int j = offset[curr]; j < offset[curr+1]; j++) 
	    parallel_for(offset[curr],offset[curr+1],[&](int j){
                int ngh = E[j];
                if (dist[ngh] == -1 && CAS(&dist[ngh], -1, (dist[curr] + 1))) {
                    out_flag[j-ofs] = 1;
                }
            });
        }
        int nghs_size = 0;
        nghs[i] = filter(offset, E, out_flag, curr, nghs_size);
        sizes[i] = nghs_size;
        delete[] out_flag;
    });
    return flatten(nghs, sizes, fsize);
}

int* edgeMapDense(int n, int* offset, int* E, int* dist, int* frontier, int &fsize) {
    int* newf_bit = new int[n];
    int* size = new int[n];
   // cilk_for(int i = 0; i < n; i++) 
    parallel_for(0,n,[&](int i){
        newf_bit[i] = 0;
        size[i] = 0;
    });
    // cilk_for(int i = 0; i < n; i++) 
	parallel_for(0,n,[&](int i){
        if (dist[i] == -1) {
            for (int j = offset[i]; j < offset[i+1]; j++) {
                int ngh = E[j];
                if (frontier[ngh]) {
                    dist[i] = dist[ngh] + 1;
                    newf_bit[i] = 1;
                    size[i] = 1;
                    break;
                }
            }
        }
    });
    ip_scan(size, n);
    fsize = size[n-1];
    delete[] size;
    return newf_bit;
}
int* edgeMap(int n, int* offset, int* E, int* dist, int* frontier, int &fsize, string prevMode, string currMode) {
    if (prevMode != currMode) {
        frontier = convert(n, frontier, fsize, currMode);
    }

    if (currMode == "dense") {
        frontier = edgeMapDense(n, offset, E, dist, frontier, fsize);
    }
    else if (currMode == "sparse") {
        frontier = edgeMapSparse(offset, E, dist, frontier, fsize);
    }

    return frontier;
}









void BFS(uint64_t* offsets, uint32_t* edges, uint32_t* dist, size_t n, size_t m,
         uint32_t s) {
	if(n>100000){
		 dist[s] = 0;
  queue<uint32_t> q;
  q.push(s);
  while (!q.empty()) {
    uint32_t u = q.front();
    q.pop();
    for (size_t i = offsets[u]; i < offsets[u + 1]; i++) {
      uint32_t v = edges[i];
      if (dist[v] > dist[u] + 1) {
        dist[v] = dist[u] + 1;
        q.push(v);
      }
    }
  }
	}
	else{
    //cilk_for(int i = 0; i < n; i++) 
	int* dister=new int[n];
	parallel_for(0,n,[&](int i){
	    dister[i] = -1;
    });
	int* offset=new int[n];
	int* e=new int[n];
parallel_for(0,n,[&](int i){
	    offset[i] = int(offsets[i]);
    });
	parallel_for(0,n,[&](int i){
	    e[i]=int(edges[i]);;
    });
    dist[s] = 0;
    int fsize = 1;
    int* frontier = new int[fsize];
    frontier[0] = s;

    string prevMode = "sparse";
    string currMode = "sparse";
    while (fsize > 0) {
        currMode = fsize > FRONTIER_TH ? "dense":"sparse";
        frontier = edgeMap(n, offset, e, dister, frontier, fsize, prevMode, currMode);
        prevMode = currMode;
    }
	
parallel_for(0,n,[&](int i){
	    dist[i] = uint32_t(dister[i]);
    });
    delete[] frontier;}
}
