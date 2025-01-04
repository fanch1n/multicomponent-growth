#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "mc_ising.hpp"
#include <armadillo>
#include <stdio.h>
#include "read.h" 
using namespace std;
using namespace arma;
/*-----*/
int nw; 
int root_seed; 
int Li, Lj, N, mc_step; 
int N_sp, N_cell, cell_l, Q;
int window_pos;
int seed_label, seed_col;
double bond, mu; 
string load_path;
vector<mat> cell;
vector<int> init_seed; 
std::mt19937_64 root_prng;  
std::uniform_int_distribution<unsigned> udist(1, std::numeric_limits<unsigned>::max()); 
MC_Ising mcs;
void take_input(int argc, char *argv[]);
int m_x_mod(int i, int L);
int m_to_state(int sp, int ori);
mat get_U_int(vector<int> species, vector<int> orientation, int cell_l);
double delta_t;
double dt;
/*-----*/
int main (int argc, char *argv[])
{
  take_input(argc, argv); 
  if(root_seed < 0)
  {
    root_seed = std::chrono::system_clock::now().time_since_epoch().count(); 
  }
  root_seed = abs(root_seed); 
  root_prng.seed(root_seed);  
  cerr << "job id: " << nw << endl;
  cerr << "simulation_seed: " << root_seed << endl; 
  cerr << "bond strength " << bond << endl;
  cerr << "mu: " << mu << endl;
  cerr << "window center loc at: " << window_pos << endl;
  cerr << "load lattice configuration from: " << load_path << endl;
  Q = 4 * N_sp + 1;
  cell_l = int(sqrt(N_sp));
  if(window_pos < 0){ 
      seed_col = 40 * cell_l; //initial filled to 4 * seed_col
  }
  else{
      seed_col = window_pos;
  } 
  vector<int> species(N_sp);
  vector<int> orientation(N_sp); 
  vector<vector<int>> init_config(N_cell, vector<int>(N_sp));
  vector<int> config(N_sp);
 /*---- load seed configurations ----*/ 
  for(int m  = 0; m < N_cell; m++)
  {
    init_config[m] = getData(to_string(m) + "seed");
    for(int i = 0; i < N_sp; i++)
    {
      species[i] = 1 + (init_config[m][i] - 1) / 4;
      orientation[i] = (init_config[m][i] - 1) % 4;
    }
    cell.push_back(get_U_int(species, orientation, cell_l));
  }
 /*---- initialize simulation ----*/
  init_seed = getData(load_path);
  cerr << "run: load seed of size:" << init_seed.size() << endl;
  MC_Ising mcs;
  mcs.set_mc_args(nw, Li, Lj, bond, mu, udist(root_prng), cell, init_seed); //, seed_label);
  dt = 0.0;
  int u = 0;
  int threshold = Li * Lj * 0.9;
  int output_freq = 1e8;
  int config_idx = 0;
  bool first_loaded = false;
  if(init_seed.size() == Li * Lj){
    first_loaded = true;  
  }  
  string rate_log = to_string(nw) + "_rate.dat";

  string running_log = to_string(nw) + "_log.dat";
  string config_out = to_string(nw) + "_" + to_string(mc_step) + "_config.dat";
  cerr << "start with mc_step (output idx): " << mcs.mc_step_ << endl; 
  
  /*---- job starts -----*/
  while(mcs.time_ < 1e9 and mcs.window_end < threshold){
    if(first_loaded == true){
      string rate_begin = to_string(nw) + "_" + to_string(mc_step) + "_rate_before.dat";
      ofstream table_begin(rate_begin, ofstream::app); 
      mcs.RT_.print_table(table_begin);  
      table_begin << "# " << mcs.window_center << "\t" << mc_step << "\t" << config_idx << endl; 
    }
    if(u % 1000 == 0){
      mcs.update_window();

      if(first_loaded == true){
        string rate_check = to_string(nw) + "_" + to_string(mc_step) + "_rate_after.dat";
        ofstream table_check(rate_check, ofstream::app); 
        mcs.RT_.print_table(table_check);  
        table_check << "# " << mcs.window_center << "\t" << mc_step << "\t" << config_idx << endl; 
        first_loaded = false;
      }
      // check the first update once reload and restart
      // here the rate table should be exactly the same as the loaded one 
      // cerr << "after reload prev window end, window end: " << mcs.prev_window_end << "\t" << mcs.window_end << endl;
      mcs.GCMC_sweep(mcs.prev_window_end, mcs.window_end);
    }
    if(u % output_freq == 0){
      ofstream out_backup(running_log, ofstream::trunc);
      mcs.write_snapshot(out_backup);
     
      string rate_running = to_string(nw) + "_" + to_string(mc_step) + "_rate.dat";
      
      ofstream table_out(rate_running, ofstream::trunc);
      mcs.RT_.print_table(table_out);
      table_out << "# " << mcs.window_center << "\t" << mc_step << "\t" << config_idx << endl; 
 
      ofstream out_config(config_out, ofstream::app);
      mcs.write_snapshot(out_config);  
      mcs.to_file(); 
      // check rate table at the output instance 
      mcs.mc_step_ ++;
      config_idx += 1;
    }
    if(u == output_freq){
      u = 0;
    }
    mcs.KMC_step();
    u += 1;
  }
  mcs.out_prop_ << "#" << endl;
  cerr << "end with mc_step = " << mcs.mc_step_ << endl;
}
//****************************************************************************8
void take_input(int argc, char *argv[])
{
  int i = 1;
  nw        = int(stoi(argv[i++])); 
  root_seed = atoi(argv[i++]); 
  Li        = atoi(argv[i++]); 
  Lj        = atoi(argv[i++]); 
  bond      = stod(argv[i++]); 
  mu        = stod(argv[i++]); 
  mc_step   = int(stoi(argv[i++])); 
  N_sp      = atoi(argv[i++]); 
  N_cell    = atoi(argv[i++]); 
  seed_label= atoi(argv[i++]); 
  window_pos= int(stoi(argv[i++])); 
  load_path = argv[i++];
  
  return; 
}
/*----------------------------------------------*/
int m_x_mod(int i, int L)
{
  if(i < 0)
    return i + L; 
  else if(i >= L)
    return i - L; 
  else
    return i; 
}
/*----------------------------------------------*/
int m_to_state(int sp, int ori)
{
  if(sp == 0)
    return 0;
  else
    return (sp-1) * 4 + ori + 1; 
}
/*----------------------------------------------*/
mat get_U_int(vector<int> species, vector<int> orientation, int cell_l)
{	
  int Q = 4 * species.size() + 1; 
  mat cell = mat(Q, Q, fill::zeros);
  
  for(int i = 0; i < species.size(); i++)
  {
    int sp_1 = species[i];
    int dir_1 = orientation[i]; 
    int pos1_i = i / cell_l;
    int pos1_j = i % cell_l;
    
    // look at its top and left nbs to fill in the interaction martix;
    int i_nb = m_x_mod(pos1_i-1, cell_l);
    int j_nb = m_x_mod(pos1_j, cell_l); 
    int k_top = j_nb + i_nb * cell_l; 
    int y_top = 2;     
    int sticker_1 = m_x_mod(0 - dir_1, 4);
    int sp_2 = species[k_top];
    int sticker_2 = m_x_mod(y_top - orientation[k_top], 4);
    cell(m_to_state(sp_1, sticker_1), m_to_state(sp_2, sticker_2)) = 1;
    cell(m_to_state(sp_2, sticker_2), m_to_state(sp_1, sticker_1)) = 1;
    /*-------*/
    i_nb = m_x_mod(pos1_i, cell_l); 
    j_nb = m_x_mod(pos1_j-1, cell_l); 
    int k_left = j_nb + i_nb * cell_l;//k_left is to the left of k  
    int y_left = 1; 
    sp_2 = species[k_left];
    sticker_1 = m_x_mod(3 -dir_1, 4);
    sticker_2 = m_x_mod(y_left - orientation[k_left], 4);
    cell(m_to_state(sp_1, sticker_1), m_to_state(sp_2, sticker_2)) = 1;
    cell(m_to_state(sp_2, sticker_2), m_to_state(sp_1, sticker_1)) = 1;
  }

  return cell;
}
