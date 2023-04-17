ifdef CLANG
CC = clang++
else
CC = g++
endif

CPPFLAGS = -std=c++17 -Wall -Werror

ifdef CILKPLUS
CPPFLAGS += -DPARLAY_CILKPLUS -DCILK -fcilkplus
else ifdef SERIAL
CPPFLAGS += -DPARLAY_SEQUENTIAL
else
CPPFLAGS += -pthread
endif

ifdef DEBUG
CPPFLAGS += -Og -mcx16 -DDEBUG
else ifdef PERF
CPPFLAGS += -Og -mcx16 -march=native -g
else ifdef MEMCHECK
CPPFLAGS += -Og -mcx16 -DPARLAY_SEQUENTIAL
else ifdef MXMAC
CPPFLAGS += -O3 -mcpu=apple-m1
else
CPPFLAGS += -O3 -mcx16 -march=native
endif

all: BFS

BFS: BFS.h BFS.cpp
	$(CC) $(CPPFLAGS) BFS.cpp -o BFS

clean:
	rm BFS
