#ifndef RATETABLE_HPP_INCLUEDED
#define RATETABLE_HPP_INCLUEDED 1
#include <vector>
#include <list>
#include<iostream>
#include <string>
#include <random>
#include <map>
//  #define ARMA_DONT_USE_CXX11
using namespace std; 
class RateTable{
  public:
    int Nsite_; 
    vector<long long int> rates_; // reaction rate for each individual sites
    vector<vector<long long int>> table_; // full rate look up table 
    map<long long int, long long int> reaction_map_;
/*---------------*/
    RateTable();
    RateTable(int Nsite, map<long long int, long long int> reaction_map, unsigned prng_seed); 
  
    double R;
    long long int rl;   
    int sample_site(); // sample reaction and time
    double sample_time();
    void update(int site, int reaction_number); // propagate through the window
    void set_rate(int site, int reaction_number); // set rate for site, do not propagate
    void recompute(); //propagate using the rates for individal sites
    void print_table(std::ofstream& table_out);
    std::mt19937_64 prng_; 
    std::uniform_real_distribution<double> udist_; 
  private:
};
  bool check_overflow(long long int i);
#endif
