
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
  std::cout << ll << std::endl;  
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

  unsigned int attribute = PLL_ATTRIB_ARCH_AVX;
  if (useTemplates) { 
    attribute |= PLL_ATTRIB_TEMPLATES;
    std::cout << "Templates implementation" << std::endl;
  }

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
  check_dna(hbg011004);
  

  std::cout << "bench PROT " << std::endl;
  bench_partials(family_149, 10);
  check_prot(family_149);


  return 0;
}



