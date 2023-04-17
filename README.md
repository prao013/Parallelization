# project2-BFS  

## Compiling the code  
A Makefile is given in the directory, simply use ``make`` to compile the code. If you're compiling the code on a M-series Mac, add the ``MXMAC=1`` option:  
```bash
make MXMAC=1  
```
## Running the code  
Enter the course environment by typing ``cs214`` in command line. Then you can run the code with:  
```bash
numactl -i all ./BFS [graph_name] [num_rounds]  
```
``numactl`` can help distribute memory pages evenly on multiple sockets. More information can be found in [Linux man page](https://linux.die.net/man/8/numactl). If not specified, the default values are ``graph_name=com-orkut_sym.bin``, ``num_rounds=3``. All tested graphs include:  
+ com-orkut_sym.bin  
+ soc-LiveJournal1_sym.bin  
+ RoadUSA_sym.bin  
We also include an example graph to help debug. The name is ``sample_sym.bin``. You can uncomment the following block in ``BFS.cpp`` to output its edge list.  
```C++
printf("n: %zu, m: %zu\n", n, m);
for (size_t i = 0; i < n; i++) {
  for (size_t j = offsets[i]; j < offsets[i + 1]; j++) {
    printf("(%zu, %u)\n", i, edges[j]);
  }
  puts("");
}
```


## Getting started  
Please implement your parallel quicksort in the file ``BFS.h``. All other files will be replaced in our testing.  

## Optimization hints  
+ Pre-allocate memory. Memory allocation can not be done in parallel.  
+ Optimize your **reduce/scan/filter** code first because it's an important building block in this algorithm. You can even have a separate unit test for it.  
+ More optimization hints can be found in the project instructions.

