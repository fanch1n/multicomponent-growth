#ifndef MC_HPP_INCLUEDED
#define MC_HPP_INCLUEDED 1
# include<vector>
#include <list>
#include <iostream>
#include <string>
#include <random>
#include <map>
#include "field.hpp"
#include "rate_table.hpp"
using namespace std; 

class MC_Ising{

  public: 
    int Li_; 
    int Lj_; 
    int N_; 
    int largest_nucleus;
    double resolution; // resolution for converting float to int
    double scale;
    map<long long int, long long int> reaction_map;
    Field field_; // lattice configuration
    RateTable RT_; // rable table (float > integer conversion) 
    long int mc_step_; 
    long double time_;
    ofstream out_prop_; 
   
    /*----window----*/
    int window_center;
    int window_start;
    int window_end;
    int prev_window_end;
    int window_width;
    int bulk_id;
    /*---------------*/
    void set_mc_args(int w,int Li, int Lj, double bond, double mu, unsigned prng_seed, vector<mat> cell, vector<int> init_seed); 
    int get_reaction_id(int site); // find out which reaction to happen
    int get_largestCluster(); 
    int find_windowCenter();
    bool update_window();
    bool KMC_step(); //sample rate table; modify lattice; update rate table. 
    bool GCMC_sweep(int left, int right);
    //void reconstruct(); // loop over the lattice and set rates and propagte the rate table.
    void to_file();
    void write_snapshot(std::ofstream& out_fig);
  private:
    int w_; 
    std::mt19937_64 prng_; 
    std::uniform_real_distribution<double> udist_; 
}; 

#endif

