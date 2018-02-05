
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
  
  unsigned int attribute = PLL_ATTRIB_ARCH_AVX | PLL_ATTRIB_TEMPLATES;

  std::shared_ptr<Dataset> hbg011004 = loadDataset("data/HBG011004.raxml.bestTree",
      "data/HBG011004.fasta",
      attribute,
      AF_FASTA,
      AT_DNA);
  
  std::shared_ptr<Dataset> family_149 = loadDataset("data/family_149.newick",
      "data/family_149.fasta",
      attribute,
      AF_FASTA,
      AT_PROT);
  
  
  std::cout << "bench DNA " << std::endl;
  bench_partials(hbg011004, 300);
  bench_likelihood(hbg011004);

  std::cout << "bench PROT " << std::endl;
  bench_partials(family_149, 10);
  bench_likelihood(family_149);

  return 0;
}



