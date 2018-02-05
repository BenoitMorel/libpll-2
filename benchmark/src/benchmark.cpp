
#include "pll.h"
#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "constants.h"
#include "kernels.h"
#include <iostream>
#include <chrono>

using namespace std;
using milli = std::chrono::milliseconds;

void bench_partials(std::shared_ptr<Dataset> dataset, unsigned int iterations)
{
  auto start = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < iterations; ++i) {
    update_all_partials(dataset);  
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double>  elapsed_seconds = end - start;
  std::cout << elapsed_seconds.count() << "s " << std::endl;
}

void bench_likelihood(std::shared_ptr<Dataset> dataset)
{
  std::cout << compute_likelihood(dataset) << std::endl;  
}

int main()
{
  char name[] = "/tmp/fileXXXXXX";
  int fd = mkstemp(name);
  if (!fd) {
    exit(1);
  }
  printf("Benchmark results will be stored in %s\n", name);
  
  std::shared_ptr<Dataset> hbg011004 = loadDataset("data/HBG011004.raxml.bestTree",
      "data/HBG011004.fasta",
      PLL_ATTRIB_ARCH_AVX,
      AF_FASTA,
      AT_DNA);
  bench_partials(hbg011004, 1000);
  //bench_likelihood(hbg011004);

  return 0;
}



