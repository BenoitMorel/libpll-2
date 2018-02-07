
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

double bench_likelihood(std::shared_ptr<Dataset> dataset)
{
  double ll = compute_likelihood(dataset);
  return ll;
}

void check_dna(std::shared_ptr<Dataset> dataset) 
{
  double llDNA = bench_likelihood(dataset);
  if (fabs(llDNA + 55847.7) > 1.0) {
    std::cerr << "Error: wrong dna likelihood" << std::endl;
  }
}

void check_prot(std::shared_ptr<Dataset> dataset) 
{
  double llPROT = bench_likelihood(dataset);
  if (fabs(llPROT + 106350) > 1.0) {
    std::cerr << "Error: wrong prot likelihood" << std::endl;
  }
}


int main(int argc, char **argv)
{
  if (argc != 2) {
    std::cerr << "Invalid syntax" << std::endl;
  }
  bool useTemplates = atoi(argv[1]);

  unsigned int attribute_avx = PLL_ATTRIB_ARCH_AVX;
  unsigned int attribute_sse = PLL_ATTRIB_ARCH_SSE;
  if (useTemplates) { 
    attribute_avx |= PLL_ATTRIB_TEMPLATES;
    attribute_sse |= PLL_ATTRIB_TEMPLATES;
    std::cout << "Templates implementation" << std::endl;
  }

  std::shared_ptr<Dataset> hbg011004_avx = loadDataset("data/HBG011004.raxml.bestTree",
      "data/HBG011004.fasta",
      attribute_avx,
      AF_FASTA,
      AT_DNA);
  std::shared_ptr<Dataset> family_149_avx = loadDataset("data/family_149.newick",
      "data/family_149.fasta",
      attribute_avx,
      AF_FASTA,
      AT_PROT);
  std::shared_ptr<Dataset> hbg011004_sse = loadDataset("data/HBG011004.raxml.bestTree",
      "data/HBG011004.fasta",
      attribute_sse,
      AF_FASTA,
      AT_DNA);
  std::shared_ptr<Dataset> family_149_sse = loadDataset("data/family_149.newick",
      "data/family_149.fasta",
      attribute_sse,
      AF_FASTA,
      AT_PROT);
  /*
  std::cout << "bench DNA avx: ";
  bench_partials(hbg011004_avx, 300);
  check_dna(hbg011004_avx);

  std::cout << "bench PROT avx: ";
  bench_partials(family_149_avx, 10);
  check_prot(family_149_avx);
*/
  std::cout << "bench DNA sse: ";
  bench_partials(hbg011004_sse, 300);
  check_dna(hbg011004_sse);

  std::cout << "bench PROT sse: ";
  bench_partials(family_149_sse, 10);
  check_prot(family_149_sse);


  return 0;
}



