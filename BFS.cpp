#include "BFS.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>

#include "get_time.h"

using Type = long long;
using namespace std;
using namespace parlay;

void read_graph(const string filename, uint64_t*& offsets, uint32_t*& edges,
                size_t& n, size_t& m) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "Error: Cannot open file " << filename << '\n';
    abort();
  }
  size_t sizes;
  ifs.read(reinterpret_cast<char*>(&n), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&m), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&sizes), sizeof(size_t));
  assert(sizes == (n + 1) * 8 + m * 4 + 3 * 8);

  offsets = (uint64_t*)malloc((n + 1) * sizeof(uint64_t));
  edges = (uint32_t*)malloc(m * sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(offsets), (n + 1) * sizeof(uint64_t));
  ifs.read(reinterpret_cast<char*>(edges), m * sizeof(uint32_t));
  if (ifs.peek() != EOF) {
    cerr << "Error: Bad data\n";
    abort();
  }
  ifs.close();
}

void sequential_BFS(uint64_t* offsets, uint32_t* edges, uint32_t* dist,
                    size_t n, size_t m, uint32_t s) {
  dist[s] = 0;
  queue<int> q;
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

int main(int argc, char* argv[]) {
  string graph = "com-orkut_sym.bin";
  int num_rounds = 3;
  if (argc >= 2) {
    graph = string(argv[1]);
  }
  if (argc >= 3) {
    num_rounds = atoi(argv[2]);
  }
  string input_path = "/usr/local/cs214/graph/" + graph;

  size_t n, m;
  uint64_t* offsets;
  uint32_t* edges;
  read_graph(input_path, offsets, edges, n, m);

  // printf("n: %zu, m: %zu\n", n, m);
  // for (size_t i = 0; i < n; i++) {
  // for (size_t j = offsets[i]; j < offsets[i + 1]; j++) {
  // printf("(%zu, %u)\n", i, edges[j]);
  //}
  // puts("");
  //}

  double total_time = 0;
  uint32_t* dist = (uint32_t*)malloc(n * sizeof(uint32_t));
  uint32_t* v_dist = (uint32_t*)malloc(n * sizeof(uint32_t));

  uint32_t sources[] = {1, 2, 3, 4, 5, 6, 7, 8};
  size_t num_sources = sizeof(sources) / sizeof(sources[0]);

  for (int i = 0; i <= num_rounds; i++) {
    parallel_for(0, n, [&](size_t j) { dist[j] = v_dist[j] = INT_MAX; });

    parlay::timer t;
    BFS(offsets, edges, dist, n, m, sources[i % num_sources]);
    t.stop();

    sequential_BFS(offsets, edges, v_dist, n, m, sources[i % num_sources]);
    parallel_for(0, n, [&](size_t j) {
      if (dist[j] != v_dist[j]) {
        std::cout << "The output is not correct\n";
        exit(0);
      }
    });

    if (i == 0) {
      std::cout << "Warmup round running time: " << t.total_time() << std::endl;
    } else {
      std::cout << "Round " << i << " running time: " << t.total_time()
                << std::endl;
      total_time += t.total_time();
    }
  }
  std::cout << "Average running time: " << total_time / num_rounds << std::endl;

  free(offsets);
  free(edges);
  free(dist);
  free(v_dist);
  return 0;
}
