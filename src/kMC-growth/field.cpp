#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "field.hpp"
using namespace std;
using namespace arma;
extern int Q;
extern int cell_l; 
extern int seed_col;
extern double init_frac; 
Field::Field()
{
//nothing
}
Field::Field(int Li, int Lj, double bond, double mu, unsigned prng_seed, int w, vector<mat> cell, vector<int> init_seed) 
  :Li_(Li), Lj_(Lj), bond_(bond), mu_(mu), w_(w), cell_(cell), init_seed_(init_seed) //i labels which row, j labels which colum, Li_ = # rows, Lj_ = # columns
{
  N_ = Li_ * Lj_; 
  prng_ = std::mt19937_64(prng_seed);  
  udist_ = std::uniform_real_distribution<double>(0.0, 1.0); 
  /*----initialize field----*/
  phi_ = vector<int>(N_, 0); 
  Npart_ = 0;
  /*-----initialize w/ seed configurtions---------*/
  if(init_seed_.size() == cell_l * cell_l)
  {
    /*-----initialize w/ init_seed---------*/
    for(int i = 0; i < seed_col; i += cell_l)
    {
      for(int j = 0; j < Lj_; j += cell_l)
      {
        for(int k = 0; k < init_seed_.size(); k++)
        {
          phi_[(i + k/cell_l) * Lj_ + j + k % cell_l] = init_seed_[k];
        }
        Npart_ += cell_l * cell_l;
      }
    }
  }
  /*-----initialize w/ loaded configurations---------*/
  else
  { 
    cerr << "with input seed" << endl;
    cerr << "load config size: " << init_seed_.size() << endl;
    for(int i = 0; i < init_seed_.size(); i++)
    {
      phi_[i] = init_seed_[i];
      if(phi_[i] > 0) Npart_ ++ ;
    }
  }

  //energy_ = get_energy(); 
  if(w_ == 0)
  {
    print_config();
  }
}
/*-----------------------------------------------*/
void Field::print_config()
{
  cout << "------------------" << endl; 
  for(int i = 0; i < Li_; i++)
  {
    for(int j = 0; j < Lj_; j++)
    {
      cout << left << setw(4) << phi_[i*Lj_+j]; 
    }
    cout << endl; 
  }
  return; 
}
/*-----------------------------------------------*/
double Field::get_energy()
//-----
// a = 0: 
//   0
// 3 m 1
//   2
//-----
// a = 1: 
//   3
// 2 m 0
//   1
{
  double energy = 0; 
  for(int k = 0; k < N_; k++)
  {
    //k = j+i*Lj = (i, j); 
    int i = k / Lj_; 
    int j = k % Lj_;  
    int i_nb = x_mod(i-1, Li_); 
    int j_nb = x_mod(j, Lj_); 
    int k_top = j_nb + i_nb * Lj_; 
    int y_top = 2;     
    /*-------*/
    i_nb = x_mod(i, Li_); 
    j_nb = x_mod(j-1, Lj_); 
    int k_left = j_nb + i_nb * Lj_;//k_left is to the left of k  
    int y_left = 1; 
    energy += get_bond_energy(k, /*y*/0, k_top, /*y_top*/2); 
    energy += get_bond_energy(k, /*y*/3, k_left, /*y_left*/1); 
  }
  energy += -mu_ * Npart_; 
  return energy; 
}
/*-----------------------------------------------*/
double Field::get_local_energy(int k)
  //if the orientation state is a, then the sticker at the top of the box
  //is 4-a, then mod(4-a+y,4) is the sticker along the direction of y
{
  //k = j+i*Lj = (i, j); 
  int i = k / Lj_; 
  int j = k % Lj_;  
  int i_nb = -1, j_nb = -1; 
  /*-------*/
  i_nb = x_mod(i+1, Li_); 
  j_nb = x_mod(j, Lj_); 
  int k_bottom = j_nb + i_nb * Lj_;  
  int y_bottom = 0; 
  /*-------*/
  i_nb = x_mod(i, Li_); 
  j_nb = x_mod(j-1, Lj_); 
  int k_left = j_nb + i_nb * Lj_;//k_left is to the left of k  
  int y_left = 1; 
  /*-------*/
  i_nb = x_mod(i-1, Li_); 
  j_nb = x_mod(j, Lj_); 
  int k_top = j_nb + i_nb * Lj_; 
  int y_top = 2;     
  /*-------*/
  i_nb = x_mod(i, Li_); 
  j_nb = x_mod(j+1, Lj_); 
  int k_right = j_nb + i_nb * Lj_;  
  int y_right = 3; 
  /*-------*/
  double local_e = 0; 
  local_e += get_bond_energy(k, /*y*/2, k_bottom, y_bottom); 
  local_e += get_bond_energy(k, /*y*/3, k_left, y_left); 
  local_e += get_bond_energy(k, /*y*/1, k_right, y_right); 
  local_e += get_bond_energy(k, /*y*/0, k_top, y_top); 

  return local_e; 
}
/*-----------------------------------------------*/
double Field::get_bond_energy(int k1, int y1, int k2, int y2)
{
  double bond_e = -1000000; 
  if(phi_[k1] == 0 || phi_[k2] == 0)
  {
    bond_e = 0; 
  }
  else 
  {
    int p1 = phi_[k1]-1; 
    int p2 = phi_[k2]-1; 
    int a1 = p1%4; //a: orientaion 
    int a2 = p2%4; 
    int s1 = p1/4; //s: species 
    int s2 = p2/4; 
    a1 = x_mod(y1-a1,4); 
    a2 = x_mod(y2-a2,4); 
    p1 = s1*4+a1+1; 
    p2 = s2*4+a2+1; 

    bool is_e = false; 
    for(int i = 0; i < cell_.size(); i++)
    {
      is_e = is_e || cell_[i](p1, p2) == 1; 
    }
    bond_e = is_e ? -bond_ : 0; 
  }

  return bond_e; 
}
/*-----------------------------------------------*/
vector<int> Field::get_nbindex(int k)
{
  // return the index of all neibor of k in the order: dw, lf, up, rt
  vector<int> nb_list;
  // k = j+i*Lj = (i,j), y = 0,1,2,3
  int i = k / Lj_;
  int j = k % Lj_;
  int i_nb = -1, j_nb = -1;
  /*-------*/
  i_nb = x_mod(i+1, Li_);
  j_nb = x_mod(j, Li_);
  int k_bottom = j_nb + i_nb * Lj_;
  /*-------*/
  i_nb = x_mod(i, Li_);
  j_nb = x_mod(j-1, Lj_);
  int k_left = j_nb + i_nb * Lj_;
  /*-------*/
  i_nb = x_mod(i-1, Li_);
  j_nb = x_mod(j, Lj_);
  int k_top = j_nb + i_nb * Lj_;
  /*-------*/
  i_nb = x_mod(i, Li_);
  j_nb = x_mod(j+1, Lj_);
  int k_right = j_nb + i_nb * Lj_;
  /*-------*/
  nb_list.push_back(k_bottom);
  nb_list.push_back(k_left);
  nb_list.push_back(k_top);
  nb_list.push_back(k_right);

  return nb_list;
}
/*----------------------------------------------*/
vector<double> Field::get_bond_energy_withtype(int k1, int y1, int k2, int y2)
{
  vector<double> bonde_type(cell_.size());
  if(phi_[k1] == 0 || phi_[k2] == 0)
  {
    for(int i = 0; i < cell_.size(); i++) bonde_type[i] = 0; 
  }
  else 
  {
    int p1 = phi_[k1]-1; 
    int p2 = phi_[k2]-1; 
    int a1 = p1%4; //a: orientaion 
    int a2 = p2%4; 
    int s1 = p1/4; //s: species 
    int s2 = p2/4; 
    a1 = x_mod(y1-a1,4); 
    a2 = x_mod(y2-a2,4); 
    p1 = s1*4+a1+1; 
    p2 = s2*4+a2+1; 

    vector<bool> is_e(cell_.size());
    for(int i = 0; i < cell_.size(); i++)
    {
      is_e[i] = false || cell_[i](p1, p2) == 1;
      bonde_type[i] = is_e[i] ? -bond_ : 0;
    }
  }
  return bonde_type; 
}

/*----------------------------------------------*/
int Field::count_nnbonds(int k)
{
  int re = 0;
  if(phi_[k] != 0){
    double eng = get_local_energy(k);
    re = int(-eng/bond_);
  }
  return re; 
}

int Field::count_neighbors(int k)
{
  int re = 0;
  vector<int> nbs = get_nbindex(k);
  for(int i = 0; i < 4; i++){
    if(phi_[nbs[i]] != 0) re += 1; 
  }
  return re; 
}
/*----------------------------------------------*/
int x_mod(int i, int L)
{
  if(i < 0)
    return i + L; 
  else if(i >= L)
    return i - L; 
  else
    return i; 
}
/*----------------------------------------------*/
int to_state(int sp, int ori)
{
  if(sp == 0)
    return 0;
  else
    return (sp-1) * 4 + ori + 1; 
}

