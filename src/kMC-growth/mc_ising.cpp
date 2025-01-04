#include <cmath>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <stack>
#include "mc_ising.hpp"
#include <bits/stdc++.h>
using namespace std;
extern double frac; 
extern int mc_step, Q, seed_col, root_seed;

/*----------------------------------------------*/
void MC_Ising::set_mc_args(int w, int Li, int Lj, double bond, double mu, unsigned prng_seed, vector<mat> cell, vector<int> init_seed)
{
  w_ = w;
  Li_ = Li; 
  Lj_ = Lj; 
  N_ = Li_ * Lj_; 
  mc_step_ = mc_step;
  time_ = 0.0;
  prng_ = std::mt19937_64(prng_seed);  
  udist_ = std::uniform_real_distribution<double>(0.0, 1.0); 
  std::uniform_int_distribution<unsigned> seed_dist(1, std::numeric_limits<unsigned>::max()); 
  std::uniform_int_distribution<unsigned> rate_dist(1, std::numeric_limits<unsigned>::max()); 
  out_prop_.close(); 
  out_prop_.open(to_string(w_) + "_" + to_string(mc_step_) + "_properties.out"); 
  
  bulk_id = 0; // get update at step 0 to correct values
  window_center = seed_col * Lj_;
  window_width = 100;
  window_start = window_center - window_width * Lj_;
  window_end = window_center + window_width * Lj_;
  prev_window_end = window_end;
  // const char *Rtypes[6] = { "Null", "Insertion", "Removal-0", "Removal-1", "Removal-2", "Removal-3"}; 
  map<int, double> remap; 
  remap[0] = (Q-1) * exp(mu);   // insertion
  remap[1] = 1.;               // removal of particle making 0 bond with its nb
  remap[2] = exp(-bond);      // removal ... 1 bond ...
  remap[3] = exp(-2 * bond); // removal ... 2 bonds ...
  remap[4] = exp(-3 * bond);// removal ... 3 bonds ...
  remap[-1] = 0.;          // impossible reaction
  resolution = 1.;
   scale = 10000000000000.0;
  // scale = 1844674407370955.2; // scale = remap[4] * resolution; 
  cerr << setprecision(12) << "scale for the float point number conversion: " << scale << endl;
  cerr << "original rate\t scaled rate" << endl;
  for(int j = -1; j < 5; j++){
    reaction_map[(long long int) j] = (long long int) (remap[j] * scale); // insertion
    cerr << "\t" << remap[j] << "\t" << reaction_map[(long long int) j] << endl;
  } 
  cerr << "----------------------" << endl;
  /*---initialize---*/ 
  field_ = Field(Li_, Lj_, bond, mu, seed_dist(prng_), w_, cell, init_seed); 
  RT_ = RateTable(2 * window_width * Lj_, reaction_map, rate_dist(prng_)); 
  int react_id;
  for(int i = 0; i < 2 * window_width * Lj_; i++){
      react_id = get_reaction_id(window_start + i);  
      RT_.set_rate(i, react_id);
  } 
  RT_.recompute();
  cerr << "total rates within the window: " << RT_.table_[RT_.table_.size()-1][0] << endl; 
  //RT_.print_table();
  return;
}
/*----------------------------------------------*/
int MC_Ising::get_largestCluster(){
  vector<bool> visited(N_, false);
  int max_size = -1;
  stack<int> stack;
  for(int i = 0; i < N_; i++){
    if (field_.phi_[i] > 0 && !visited[i]){
      stack.push(i);
      visited[i] = true;
      int size = 1;
      while(!stack.empty()){
        int node = stack.top();
        stack.pop();
        vector<int> nb_idx;
        nb_idx = field_.get_nbindex(node);
        for(int m = 0; m < 4; m++){
          int nxt = nb_idx[m];
          if(field_.phi_[nxt] > 0 && !visited[nxt]){
            stack.push(nxt);
            visited[nxt] = true;
            size += 1;
          }
        }
      }
    max_size = max(max_size, size);
    }
  }
  return max_size;
}
/*----------------------------------------------*/
int MC_Ising::find_windowCenter(){
  int rightmost_index = -1;
  vector<bool> visited(window_end - bulk_id);  // start the dfs from bulk_id before update it
  vector<int> cluster_label(window_end - bulk_id, -1); // (index by true index - bulk_id);
  vector<int> rightmost_bondx(Lj_, -1); 
  int count = 0;
  stack<int> stack;
  for(int i = 0; i < Lj_; ++i){
    if (field_.phi_[i + bulk_id] > 0 && !visited[i]){  
      stack.push(i);
      visited[i] = true;
      cluster_label[i] = count;
      while(!stack.empty()){
        int node = stack.top();
        stack.pop();        
        rightmost_index = max(rightmost_index, node);
        int j = node % Lj_;
        rightmost_bondx[j] = max(node, rightmost_bondx[j]);
        vector<int> nb_idx;
        nb_idx = field_.get_nbindex(bulk_id + node);
        for(int m = 0; m < 4; m++){
          int nxt = nb_idx[m];
          if(nxt >= bulk_id && nxt < window_end){  
            if(field_.phi_[nxt] > 0 && !visited[nxt - bulk_id]){
              stack.push(nxt - bulk_id);
              visited[nxt - bulk_id] = true;
              cluster_label[nxt - bulk_id] = count;
            }
          }
        }
      }
      count ++;
    }
  }
  // get the right most occupied site from the all cluster member and  
  // check whether they all belong to the same cluster connected to the bulk
  vector<int> rightmost_labels; 
  for(int j = 0; j < Lj_; j++){
    rightmost_labels.push_back(cluster_label[rightmost_bondx[j]]);
  }
  int unique_label = 0;
  bool check1 = true;
  sort(rightmost_labels.begin(), rightmost_labels.end());
  for(int j = 0; j < Lj_; j++){
    if(rightmost_labels[j] == -1){
      check1 = false;
      break;
    }
    while (j < rightmost_labels.size()-1 && rightmost_labels[j] == rightmost_labels[j+1]){
      j++;
    }
    unique_label++;
  }
  if(unique_label > 1) check1 = false;
  if(check1 == false){
    cerr << "failed check1! interface not fully connected!" << endl;
    exit(1);
  }

  int frt_index = rightmost_index - (rightmost_index % Lj_) + Lj_ + bulk_id; 
  return frt_index;
}
/*----------------------------------------------*/
bool MC_Ising::update_window(){
  int curr_wcenter = window_center;
  window_center = find_windowCenter(); 
  // check if the new frt is too close to the previous window end
  if(window_center > window_end - 2 * Lj_){
    cerr << "failed check2!" << endl;
    exit(1);
  }
  
  prev_window_end = window_end;
  window_start = window_center - window_width * Lj_;
  window_end = window_center + window_width * Lj_;
  
  int react_id;
  for(int i = 0; i < 2 * window_width * Lj_; i ++){
      react_id = get_reaction_id(window_start + i);  
      RT_.set_rate(i, react_id);
  } 
  RT_.recompute();
  bulk_id = window_start - 2 * Lj_;
  return true;
}
/*----------------------------------------------*/
bool MC_Ising::KMC_step()
{
  int r_site = window_start + RT_.sample_site();
  double tau = RT_.sample_time();

  if(field_.phi_[r_site] == 0){ // insertion 
    int q1 = 1;
    int q2 = Q-1; 
    int q = -1;
    int mid;
    while(q1 <= q2){
      mid = (q2+q1)/2;
      if(RT_.rl + mid * exp(field_.mu_) * scale < RT_.R){ //normalize with scale/resolution
        q1 = mid + 1;
      } 
      else{
        q = mid;
        q2 = mid - 1;
      }
    }
    if(mid == Q-1) q = mid; // edge case when the chose insertion is the last one
    field_.phi_[r_site] = q;
    field_.Npart_ ++;
  }
  else{
    field_.phi_[r_site] = 0;
    field_.Npart_ --;
  }
  int ki = get_reaction_id(r_site);
  RT_.update(r_site - window_start, ki); // update r and r's nb within the window 
  vector<int> rnbs = field_.get_nbindex(r_site);
  for(int m = 0; m < rnbs.size(); m++){
    int ns = rnbs[m];
    if(ns >= window_start && ns < window_end){
      int kns = get_reaction_id(ns);
      RT_.update(ns - window_start, kns);
    } 
  }   
  time_ = time_ + tau * scale;
  
  return false; 
}
/*----------------------------------------------*/
bool MC_Ising::GCMC_sweep(int left, int right){
  // left: prev_window_end; right: window_end
  // do sweeps only in the newly added region  when window moving forward (to the right), right > left
  // if move backwards do nothing
  if(right < left){ //cerr << "backward window!" << endl;
    return false;
  }
  for(int i = left; i < right; i++){
    int oldq = field_.phi_[i];
    int old_Npart = field_.Npart_;
    double old_energy = field_.get_local_energy(i);  
    int newq = int((Q-1) * udist_(prng_)); 
    field_.phi_[i] = newq;
    double new_energy = field_.get_local_energy(i);
    double dE = new_energy - old_energy;
    if(newq > 0 && oldq == 0){
      field_.Npart_++;
      dE -= field_.mu_;
    }
    else if(newq == 0 && oldq > 0){
      field_.Npart_--;
      dE += field_.mu_;
    }
    // newq > 0 && oldq > 0; do nothing    
    if(udist_(prng_) < exp(-dE)){
      field_.energy_ += dE;
    }
    else{
      field_.phi_[i] = oldq;
      field_.Npart_ = old_Npart;
    }
  }
  return false;
}
/*----------------------------------------------*/
int MC_Ising::get_reaction_id(int site){
  int id = -1; // impossible move
  int nb_num = field_.count_neighbors(site); // count number of occupied nn neighbors
  if(nb_num < 4){
    if(field_.phi_[site] == 0){ // insertion
      return 0;
    }
    else{
      int bond_num = field_.count_nnbonds(site); // removal of 0, 1, 2, 3 bonds >> reaction 1, 2, 3, 4
      return bond_num + 1;
    } 
  }
  return id;
}
/*----------------------------------------------*/
void MC_Ising::to_file()
{ 
  out_prop_ << mc_step_ << "\t" << time_ << "\t" << window_center << "\t" << field_.Npart_ << endl;
  return;
}
void MC_Ising::write_snapshot(std::ofstream& out_fig)
{
  out_fig << "# " << mc_step_ << " " << time_ << " " << window_center/Lj_ << " " << root_seed << endl;
  for(int i = 0; i < Li_; i++)
  {
    for(int j = 0; j < Lj_; j++)
    {
      out_fig << left << setw(4) << field_.phi_[i * Lj_ + j];
    }
    out_fig << endl;
  }
  out_fig << endl; 
  //out_fig.close();
} 

