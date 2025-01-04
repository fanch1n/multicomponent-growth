#ifndef FIELD_HPP_INCLUEDED
#define FIELD_HPP_INCLUEDED 1
# include<vector>
# include<list>
# include<iostream>
# include <string>
# include <random>
//  #define ARMA_DONT_USE_CXX11
# include <armadillo> 
using namespace std; 
using namespace arma;
class Field{
  public:
    int w_; 
    int Li_; 
    int Lj_; 
    int N_; 
    double energy_; 
    double bond_; 
    double mu_; 
    int Npart_; 
    vector<int> phi_; 
    vector<mat> cell_;
    vector<int> init_seed_; 
/*---------------*/
    Field(); 
    Field(int Li, int Lj, double bond, double mu, unsigned prng_seed, int w, vector<mat> cell, vector<int> init_seed); 
    double get_energy(); 
    double get_local_energy(int k); 
    double get_bond_energy(int k1, int y1, int k2, int y2); 
    vector<int> get_nbindex(int k); 
    int count_nnbonds(int k);
    int count_neighbors(int k);
    void print_config(); 
    vector<double> get_bond_energy_withtype(int k1, int y1, int k2, int y2); 
    std::mt19937_64 prng_; 
    std::uniform_real_distribution<double> udist_; 
  private:
};
  int x_mod(int i, int L);
  int to_state(int sp, int dir); 
#endif

