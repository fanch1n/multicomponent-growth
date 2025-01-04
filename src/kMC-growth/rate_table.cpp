#include <cmath>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "rate_table.hpp"
#include <bits/stdc++.h>
using namespace std;
extern int Q;
/*-----------------------------------------------*/
RateTable::RateTable()
{
}

RateTable::RateTable(int Nsite, map<long long int, long long int> reaction_map, unsigned prng_seed) 
{
  prng_ = std::mt19937_64(prng_seed);  
  udist_ = std::uniform_real_distribution<double>(0.0, 1.0); 
  Nsite_ = Nsite;
  /*---- initialize table ----*/
  R = 0; 
  rl = 0; 
  rates_ = vector<long long int>(Nsite, 0);
  table_.push_back(rates_); 
  reaction_map_ = reaction_map; 
  int N = Nsite_ >> 1;
  vector<long long int> r_temp;
  r_temp = rates_;
  while(N >= 1){
    vector<long long int> arr;
    for(int i = 0; i < N; i++){
      if(2*i + 1 < r_temp.size()){
        arr.push_back(r_temp[2*i] + r_temp[2*i+1]);
      }
      else{
        arr.push_back(r_temp[3*i]);
      }
    }
    cerr << N << "\t" << arr.size() << endl;
    table_.push_back(arr);
    r_temp = arr;  
    if(N == 1 or (N & 1) == 0) N = N >> 1; // N = 1 terminate or N is even
    else N = (N >> 1) + 1;  //  N is odd
  }
  cerr << "RT: end table\t" << table_.size() << "\t" << table_[0].size() << endl;

  return;
}
/*-----------------------------------------------*/
void RateTable::update(int site, int reaction_number){
  // propagate through the window
  rates_[site] = reaction_map_[(long long int) reaction_number];
  int id = site;
  long long int old_rate = table_[0][site];
  long long int new_rate = reaction_map_[(long long int) reaction_number];
  long long int delta = new_rate - old_rate;
  for(int j = 0; j < table_.size(); j++){
    // check for overflow
    if (delta > 0 and table_[j][id] > LLONG_MAX - delta){
      cerr << "during update rate overflow" << endl;
      cerr << table_[j][id] << "\t" << delta << endl;
      exit(1);
    } 
    table_[j][id] += delta;
    id = id >> 1;
  } 
  //print_table();
  return;
}
/*-----------------------------------------------*/
void RateTable::set_rate(int site, int reaction_number){
  // set rate for site, do not propagate
  rates_[site] = reaction_map_[(long long int) reaction_number];
  table_[0][site] = reaction_map_[(long long int) reaction_number];
  return;
}
/*-----------------------------------------------*/
void RateTable::recompute(){
// propagate using the rates for individual sites; Notice this should be called 
// after individual rates are updated!
  int N = Nsite_ >> 1;
  int row = 1;
  while(N >= 1){
    for(int i = 0; i < N; i++){
      if(2*i+1 < table_[row-1].size()){
        // check for overflow
        if (table_[row-1][2*i] > LLONG_MAX - table_[row-1][2*i+1]){
          cerr << "duing propagation rate overflow" << endl;
          cerr << row << "\t" <<  table_[row-1][2*i] << endl;
          exit(1);
        } 
        table_[row][i] = table_[row-1][2*i] + table_[row-1][2*i+1];
      }
      else table_[row][i] = table_[row-1][2*i];
    }
    if(N == 1 or (N & 1) == 0) N = N >> 1; // N = 1 terminate or N is even
    else N = (N >> 1) + 1;  //  N is odd
    row ++;
  }
  return; 
}
/*-----------------------------------------------*/
double RateTable::sample_time()
{
  double rt = udist_(prng_);
  double tau = log(1/rt) / table_[table_.size()-1][0]; // need unit conversion in mc_ising.cpp 
  return tau; 
}
/*----------------------------------------------*/
int RateTable::sample_site()
{
  double rand = udist_(prng_);
  while(rand == 1.){
    rand = udist_(prng_);
  }  
  double target = rand * table_[table_.size()-1][0]; 
  R = target;
  int k = 0;
  rl = 0; 
  long long int rr = table_[table_.size()-1][0];
  long long int rm;
  for(int j = table_.size() - 2; j > -1; j--){
    rm = rl + table_[j][k * 2];
    if(target < rm){
      k = k << 1;
      rr = rm;
    }
    else{
      k = (k << 1) + 1;
      rl = rm;
    }
  }
  // check with brute force search 
  //int p = 0;
  //double flag1 = 0;
  //double flag2 = table_[0][0];
  //while(!(flag1 < target && flag2 >= target))
  //{
  //  p++;
  //  flag2 += table_[0][p];
  //  flag1 += table_[0][p-1];
  //}
  //if(p != k){
  //  cerr << "!!!incorrect binary search" << endl;
  //  cerr << "rand r: " << rand << endl;
  //  cerr << p << "\t" << k << endl;
  //  cerr << "k: " << k << "\t" << table_[0][k] << "\ttarget: " << target << endl; 
  //  cerr << "p: " << p << "\tr_p-1, p: " << table_[0][p-1] << "\t" << table_[0][p] << endl; 
  //  exit(1);
  //}
  
  return k;
}
/*----------------------------------------------*/
void RateTable::print_table(std::ofstream& table_out){
  table_out << "#" << endl;
  for(int j = 0; j < table_.size(); j++){
    for(int i = 0; i < table_[j].size(); i++){
      table_out << table_[j][i] << "  ";
    }
    table_out << endl;
  } 
  table_out << "#"  << endl;
  return;
}
/*----------------------------------------------*/
bool check_overflow(long long int i){
  bool re = false;
  return re;
}
