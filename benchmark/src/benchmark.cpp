
#include "pll.h"
#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "constants.h"
#include "kernels.h"
#include <iostream>

void bench_partials(std::shared_ptr<Dataset> dataset)
{
  update_all_partials(dataset);  
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
  bench_partials(hbg011004);
  bench_likelihood(hbg011004);

  return 0;
}



