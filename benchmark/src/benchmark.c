
#include "pll.h"
#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "constants.h"


void bench_partials(Dataset *dataset)
{

}

int main()
{
  char name[] = "/tmp/fileXXXXXX";
  int fd = mkstemp(name);
  if (!fd) {
    exit(1);
  }
  printf("Benchmark results will be stored in %s\n", name);
  
  Dataset *hbg011004 = load_dataset("data/HBG011004.raxml.bestTree",
      "data/HBG011004.fasta",
      PLL_ATTRIB_ARCH_AVX,
      AT_DNA,
      AF_FASTA);




  destroy_dataset(hbg011004);
  
  return 0;
}
