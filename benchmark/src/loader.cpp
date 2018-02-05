#include "loader.h"
#include <map>
#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>
const unsigned int DEFAULT_BL = 0.000001;
const unsigned int RATE_CATS = 4;

class LibpllException: public std::exception {
public:
  LibpllException(const std::string &s): msg_(s) {}
  LibpllException(const std::string &s1, 
      const std::string s2): msg_(s1 + s2) {}
  virtual const char* what() const throw() { return msg_.c_str(); }
  void append(const std::string &str) {msg_ += str;}

private:
  std::string msg_;
};

struct pll_sequence {
  pll_sequence(char *label, char *seq, unsigned int len):
    label(label),
    seq(seq),
    len(len) {}
  char *label;
  char *seq;
  unsigned int len;
  ~pll_sequence() {
    free(label);
    free(seq);
  }
};

using pll_sequence_ptr = std::shared_ptr<pll_sequence>;
using pll_sequences = std::vector<pll_sequence_ptr>;

static int cb_full_traversal(pll_unode_t * node)
{
    return 1;
}

void setMissingBL(pll_utree_t * tree, 
    double length)
{
  for (unsigned int i = 0; i < tree->tip_count; ++i)
    if (!tree->nodes[i]->length)
      tree->nodes[i]->length = length;
  for (unsigned int i = tree->tip_count; i < tree->tip_count + tree->inner_count; ++i) {
    if (!tree->nodes[i]->length)
      tree->nodes[i]->length = length;
    if (!tree->nodes[i]->next->length)
      tree->nodes[i]->next->length = length;
    if (!tree->nodes[i]->next->next->length)
      tree->nodes[i]->next->next->length = length;
  }  
}

void parseFasta(const char *fastaFile, 
    const pll_state_t *map,
    pll_sequences &sequences,
    unsigned int *&weights)
{
  auto reader = pll_fasta_open(fastaFile, pll_map_fasta);
  if (!reader) {
    throw LibpllException("Cannot parse fasta file ", fastaFile);
  }
  char * head;
  long head_len;
  char *seq;
  long seq_len;
  long seqno;
  int length;
  while (pll_fasta_getnext(reader, &head, &head_len, &seq, &seq_len, &seqno)) {
    sequences.push_back(pll_sequence_ptr(new pll_sequence(head, seq, seq_len)));
    length = seq_len;
  }
  int count = sequences.size();;
  char** buffer = (char**)malloc(count * sizeof(char *));
  for (int i = 0; i < count; ++i) {
    buffer[i] = sequences[i]->seq;
  }
  weights = pll_compress_site_patterns(buffer, map, count, &length);
  if (!weights) 
    throw LibpllException("Error while parsing fasta: cannot compress sites");
  for (int i = 0; i < count; ++i) {
    sequences[i]->len = length;
  }
  free(buffer);
  pll_fasta_close(reader);
}
  
void parsePhylip(const char *phylipFile, 
    const pll_state_t *map,
    pll_sequences &sequences,
    bool interleaved,
    unsigned int *&weights)
{
  auto reader = pll_phylip_open(phylipFile, pll_map_phylip);
  if (!reader) {
    throw LibpllException("Error while opening phylip file ", phylipFile);
  }
  auto parse = interleaved ? pll_phylip_parse_interleaved : pll_phylip_parse_sequential;
  auto msa = parse(reader);
  if (!msa) 
    throw LibpllException("Error while parsing phylip file ", phylipFile);
  weights = pll_compress_site_patterns(msa->sequence, map, msa->count, &msa->length);
  if (!weights) 
    throw LibpllException("Error while parsing fasta: cannot compress sites");
  pll_phylip_close(reader);
  for (auto i = 0; i < msa->count; ++i) {
    pll_sequence_ptr seq(new pll_sequence(msa->label[i], msa->sequence[i], msa->count));
    sequences.push_back(seq);
    // avoid freeing these buffers with pll_msa_destroy
    msa->label[i] = 0;
    msa->sequence[i] = 0;
  }
  pll_msa_destroy(msa);
}






std::shared_ptr<Dataset> loadDataset(const std::string &newickFilename,
    const std::string &alignmentFilename,
    unsigned int attribute,
    AlignmentFormat format,
    AlphabetType alphabet)
{
  // sequences 
  pll_sequences sequences;
  unsigned int *patternWeights = 0;
  const pll_state_t *charmap = (alphabet == AT_DNA) ? pll_map_nt : pll_map_aa; 
  if (format == AF_FASTA) 
    parseFasta(alignmentFilename.c_str(), charmap, sequences, patternWeights);
  else if (format == AF_PHYLIP_INTERLEAVED or format == AF_PHYLIP_SEQUENTIAL)
    parsePhylip(alignmentFilename.c_str(), charmap, sequences, format == AF_PHYLIP_INTERLEAVED ,patternWeights);
  else
    throw LibpllException("Invalid alignment format");
  
  // partition
  unsigned int tipNumber = sequences.size();
  unsigned int innerNumber = tipNumber -1;
  unsigned int edgesNumber = 2 * tipNumber - 1;
  unsigned int sitesNumber = sequences[0]->len;
  unsigned int statesNumber = (alphabet == AT_DNA) ? 4 : 20;
  unsigned int ratesMatrices = 1;
  unsigned int categories = 4;
  pll_partition_t *partition = pll_partition_create(tipNumber,
      innerNumber,
      statesNumber,
      sitesNumber, 
      ratesMatrices, 
      edgesNumber,// prob_matrices
      categories,  
      edgesNumber,// scalers
      attribute);  
  if (!partition) 
    throw LibpllException("Could not create libpll partition");
  pll_set_pattern_weights(partition, patternWeights);
  free(patternWeights);

  // fill partition
  std::map<std::string, int> tipsLabelling;
  unsigned int labelIndex = 0;
  for (auto seq: sequences) {
    tipsLabelling[seq->label] = labelIndex;
    pll_set_tip_states(partition, labelIndex, charmap, seq->seq);
    labelIndex++;
  }
  sequences.clear();
  // todobenoit do not use hardcoded values
  double gammaRates[4] = {0.136954, 0.476752, 1, 2.38629};
  pll_set_category_rates(partition, gammaRates);
  
  if (statesNumber == 4) {
    double subst[6] = {1, 1, 1, 1, 1, 1};
    double frequencies[4] = {0.25, 0.25, 0.25, 0.25}; 
    pll_set_frequencies(partition, 0, frequencies);
    pll_set_subst_params(partition, 0, subst);
  } else {
    pll_set_frequencies(partition, 0, pll_aa_freqs_dayhoff);
    pll_set_subst_params(partition, 0, pll_aa_rates_dayhoff);
  }

  // tree
  std::ifstream t(newickFilename);
  if (!t)
    throw LibpllException("Could not load open newick file ", newickFilename);
  std::string newickString((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
  pll_utree_t * utree = pll_utree_parse_newick_string(newickString.c_str());
  if (!utree) 
    throw LibpllException("Error in pll_utree_parse_newick_string on ", newickString);
  setMissingBL(utree, DEFAULT_BL);
  
  // map tree to partition
  for (unsigned int i = 0; i < utree->inner_count + utree->tip_count; ++i) {
    auto node = utree->nodes[i];
    if (!node->next) { // tip!
      node->clv_index = tipsLabelling[node->label];
    }
  }

  // operations
  unsigned int params_indices[RATE_CATS] = {0,0,0,0};
  unsigned int nodes_count = utree->tip_count + utree->inner_count;
  unsigned int branch_count = nodes_count - 1;
  pll_unode_t * root = utree->nodes[nodes_count - 1];
  unsigned int traversal_size;
  unsigned int matrix_count;
  unsigned int ops_count;
  pll_unode_t **travbuffer = (pll_unode_t **)malloc(nodes_count * sizeof(pll_unode_t *));
  double *branch_lengths = (double *)malloc(branch_count * sizeof(double));
  unsigned int *matrix_indices = (unsigned int *)malloc(branch_count * sizeof(unsigned int));
  pll_operation_t *operations = (pll_operation_t *)malloc(utree->inner_count * sizeof(pll_operation_t));
  pll_utree_traverse(root,
        PLL_TREE_TRAVERSE_POSTORDER,
        cb_full_traversal,
        travbuffer,
        &traversal_size);
  pll_utree_create_operations(travbuffer,
      traversal_size,
      branch_lengths,
      matrix_indices,
      operations,
      &matrix_count,
      &ops_count);
  pll_update_prob_matrices(partition,
      params_indices,
      matrix_indices,
      branch_lengths,
      matrix_count);


  std::shared_ptr<Dataset> dataset(new Dataset());
  dataset->name = "dataset"; //todobenoit
  dataset->partition = partition;
  dataset->tree = utree;
  dataset->operations = operations;
  dataset->ops_count = ops_count;
  dataset->root = root;
  return dataset;
}
  
