// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Aleksei Ivanov (University of Iceland)
                         Julien Tranchida (SNL)

   Please cite the related publication:
   Ivanov, A. V., Uzdin, V. M., & Jónsson, H. (2019). Fast and Robust
   Algorithm for the Minimisation of the Energy of Spin Systems. arXiv
   preprint arXiv:1904.02669.
------------------------------------------------------------------------- */

#include "min_spin_cg.h"
#include "fix_minimize.h"
#include "output.h"
#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "output.h"
#include "timer.h"
#include "universe.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;

static const char cite_minstyle_spin_cg[] =
  "min_style spin/cg command:\n\n"
  "@article{ivanov2019fast,\n"
  "title={Fast and Robust Algorithm for the Minimisation of the Energy of "
  "Spin Systems},\n"
  "author={Ivanov, A. V and Uzdin, V. M. and J{\'o}nsson, H.},\n"
  "journal={arXiv preprint arXiv:1904.02669},\n"
  "year={2019}\n"
  "}\n\n";

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8
#define ALPHA_MAX 0.1
#define ALPHA_REDUCE 0.5
#define DELAYSTEP 5
// #define BACKTRACK_SLOPE 0.4
#define BACKTRACK_SLOPE 0.1
#define EMACH 1.0e-3

#define QUADRATIC_TOL 0.1  
#define EPS_QUAD 1.0e-28
/* ---------------------------------------------------------------------- */

MinSpinCG::MinSpinCG(LAMMPS *lmp) :
  Min(lmp), g_old(nullptr), g_cur(nullptr), p_s(nullptr), sp_copy(nullptr), g(nullptr), h(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_minstyle_spin_cg);
  nlocal_max = 0;

  // nreplica = number of partitions
  // ireplica = which world I am in universe

  nreplica = universe->nworlds;
  ireplica = universe->iworld;
  use_line_search = 0;  // no line search as default option for CG

  discrete_factor = 10.0;
}

/* ---------------------------------------------------------------------- */

MinSpinCG::~MinSpinCG()
{
  memory->destroy(g_old);
  memory->destroy(g_cur);
  memory->destroy(p_s);
  memory->destroy(g);
  memory->destroy(h);
  if (use_line_search)
    memory->destroy(sp_copy);
}

/* ---------------------------------------------------------------------- */

void MinSpinCG::init()
{
  local_iter = 0;
  der_e_cur = 0.0;
  der_e_pr = 0.0;

  Min::init();

  // warning if line_search combined to gneb

  if ((nreplica >= 1) && (linestyle != 4) && (comm->me == 0))
    error->warning(FLERR,"Line search incompatible gneb");

  // set back use_line_search to 0 if more than one replica

  if (linestyle == 3 && nreplica == 1) {
    use_line_search = 1;
  }
  else{
    use_line_search = 0;
  }

  dts = dt = update->dt;
  last_negative = update->ntimestep;

  // allocate tables

  nlocal_max = atom->nlocal;
  
  double **sp = atom->sp;
  int count=0;
  for(int i=0; i<nlocal_max; i++){
    if(sp[i][3] != 0) count=count+1;
  }
  nlocal = atom->nlocal;
  nall = nlocal_max+count;
  nvec = 4 * nall;
  
  memory->grow(g_old,3*nlocal_max,"min/spin/cg:g_old");
  memory->grow(g_cur,3*nlocal_max,"min/spin/cg:g_cur");
  memory->grow(p_s,3*nlocal_max,"min/spin/cg:p_s");
  memory->grow(g,3*nall,"min/spin/cg:g");
  memory->grow(h,3*nall,"min/spin/cg:h");
  if (use_line_search)
    memory->grow(sp_copy,nlocal_max,3,"min/spin/cg:sp_copy");
}

/* ---------------------------------------------------------------------- */

void MinSpinCG::setup_style()
{
  double **v = atom->v;
  int nlocal = atom->nlocal;
  fix_minimize->add_vector(3);

  // check if the atom/spin style is defined

  if (!atom->sp_flag)
    error->all(FLERR,"min spin/cg requires atom/spin style");

  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
}

/* ---------------------------------------------------------------------- */

int MinSpinCG::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"discrete_factor") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    discrete_factor = utils::numeric(FLERR,arg[1],false,lmp);
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void MinSpinCG::reset_vectors()
{
  // atomic dof

  // size sp is 4N vector
  if (nvec) sp = atom->sp;

  // nvec = 3 * atom->nlocal;
  if (nvec) fmvec = atom->fm[0];

  if (nvec) x = atom->x[0];
  if (nvec) fvec = atom->f[0];
  x0 = fix_minimize->request_vector(0);
  for(int i=0; i<nlocal*3; i++){
    x0[i] = x[i];
  }
  
  for(int i=nlocal*3; i<nall*3; i++){
    x0[i] = x[i-nlocal*3] + sp[(i-nlocal*3)/3][(i-nlocal*3)%3] * sp[(i-nlocal*3)/3][3]/1.323*0.1;
  }
}

/* ----------------------------------------------------------------------
   minimization via orthogonal spin optimisation
------------------------------------------------------------------------- */

int MinSpinCG::iterate(int maxiter)
{
  int nlocal = atom->nlocal;
  bigint ntimestep;
  double fmdotfm,fmsq;
  int flag, flagall, i, j;
  sp = atom->sp;
  x = atom->x[0];
  double der_e_cur_tmp = 0.0;

  int count=0;
  for(int i=0; i<nlocal; i++){
    if(sp[i][3] != 0) count=count+1;
  }
  int nall = nlocal+count;
  if (nlocal_max < nlocal) {
    local_iter = 0;
    nlocal_max = nlocal;
    memory->grow(g_old,3*nlocal_max,"min/spin/cg:g_old");
    memory->grow(g_cur,3*nlocal_max,"min/spin/cg:g_cur");
    memory->grow(p_s,3*nlocal_max,"min/spin/cg:p_s");
    memory->grow(g,3*nall,"min/spin/cg:g");
    memory->grow(h,3*nall,"min/spin/cg:h");
    if (use_line_search)
      memory->grow(sp_copy,nlocal_max,3,"min/spin/cg:sp_copy");
  }
  // init
  const double hbar = 6.5821191e-04;
  double **v = atom->v;
  f = atom->f[0];
  fm = atom->fm[0];
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  xvec = new double[nall*3];
  // double *x0 = new double[nall*3];
  for(int i=0; i<nlocal*3; i++){
    xvec[i] = x[i];
  }
  
  for(int i=nlocal*3; i<nall*3; i++){
    xvec[i] = x[i-nlocal*3] + sp[(i-nlocal*3)/3][(i-nlocal*3)%3] * sp[(i-nlocal*3)/3][3]/1.323*0.1;
  }
  double * fvec = new double[nall*3];
  for(int i=0; i<nall*3; i++){
      if(i<nlocal*3){fvec[i]=f[i];}
      else {fvec[i]=fm[i-nlocal*3]*hbar/0.3*sp[(i-nlocal*3)/3][3]/1.323;}
  }

  nvec = 3 * nall;
  for (int i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];

  double beta,gg,dot[2],dotall[2],fdotf;
  // get fnorm 
  double local_norm2_sqr = 0.0;
  for (i = 0; i < nvec; i++) local_norm2_sqr += fvec[i]*fvec[i];
  double norm2_sqr = 0.0;
  MPI_Allreduce(&local_norm2_sqr,&norm2_sqr,1,MPI_DOUBLE,MPI_SUM,world);
  gg = norm2_sqr;

  for (int iter = 0; iter < maxiter; iter++) {
    // std::cout << "nall" << nall << std::endl;
    // std::cout << "nvec" << nvec << std::endl;
    // double * xvec = new double[nall*3];
    for(int i=0; i<nlocal*3; i++){
      xvec[i] = x[i];
    }
    
    for(int i=nlocal*3; i<nall*3; i++){
      xvec[i] = x[i-nlocal*3] + sp[(i-nlocal*3)/3][(i-nlocal*3)%3] * sp[(i-nlocal*3)/3][3]/1.323*0.1;
    }

    // double * fvec = new double[nall*3];
    for(int i=0; i<nall*3; i++){
        if(i<nlocal*3){fvec[i]=f[i];}
        else {fvec[i]=fm[i-nlocal*3]*hbar/0.3*sp[(i-nlocal*3)/3][3]/1.323;}
    }

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    dot[0] = dot[1] = 0.0;

    for (int i = 0; i < nvec; i++) {
      dot[0] += fvec[i]*fvec[i];
      dot[1] += fvec[i]*g[i];
    }

    MPI_Allreduce(dot,dotall,2,MPI_DOUBLE,MPI_SUM,world);

    beta = MAX(0.0,(dotall[0] - dotall[1])/gg);
    // if ((niter+1) % nlimit == 0) beta = 0.0;
    gg = dotall[0];

    for (i = 0; i < nvec; i++) {
      g[i] = fvec[i];
      h[i] = g[i] + beta*h[i];
    }

    fdotf = 0.0;
    if (update->ftol > 0.0) {
      if (normstyle == MAX) fdotf = fnorm_max();        // max force norm
      else if (normstyle == INF) fdotf = fnorm_inf();   // infinite force norm
      else if (normstyle == TWO) fdotf = dotall[0];     // same as fnorm_sqr(), Euclidean force 2-norm
      else error->all(FLERR,"Illegal min_modify command");
      std::cout << fdotf << "   " << update->ftol << std::endl;
      if (fdotf < update->ftol*update->ftol) return FTOL;
    }

    // eprevious = ecurrent;
    // reinitialize CG if new search direction h is not downhill
    dot[0] = 0.0;
    for (int i = 0; i < nvec; i++) dot[0] += g[i]*h[i];
    MPI_Allreduce(dot,dotall,1,MPI_DOUBLE,MPI_SUM,world);
    // std::cout << "dotall:    " << dotall[0] << std::endl;
    if (dotall[0] <= 0.0) {
      for (i = 0; i < nvec; i++) h[i] = g[i];
    }

    eprevious = ecurrent;
    linemin_backtrack(ecurrent, alpha_final, fvec, xvec, x0);

    if (fabs(ecurrent-eprevious) <
        update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

/* ----------------------------------------------------------------------
   calculate gradients
---------------------------------------------------------------------- */

void MinSpinCG::calc_gradient()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;
  double hbar = force->hplanck/MY_2PI;
  double factor;

  if (use_line_search)
    factor = hbar;
  else factor = evaluate_dt();

  // loop on all spins on proc.

  for (int i = 0; i < nlocal; i++) {
    g_cur[3 * i + 0] = (fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0]) * factor;
    g_cur[3 * i + 1] = -(fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2]) * factor;
    g_cur[3 * i + 2] = (fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1]) * factor;
  }
}

/* ----------------------------------------------------------------------
   search direction:
   The Fletcher-Reeves conj. grad. method
   See Jorge Nocedal and Stephen J. Wright 'Numerical
   Optimization' Second Edition, 2006 (p. 121)
---------------------------------------------------------------------- */

void MinSpinCG::calc_search_direction()
{
  int nlocal = atom->nlocal;
  double g2old = 0.0;
  double g2 = 0.0;
  double beta = 0.0;

  double g2_global = 0.0;
  double g2old_global = 0.0;

  double factor = 1.0;

  // for multiple replica do not move end points
  if (nreplica > 1)
    if (ireplica == 0 || ireplica == nreplica - 1)
      factor = 0.0;


  if (local_iter == 0 || local_iter % 5 == 0) {  // steepest descent direction
    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = -g_cur[i] * factor;
      g_old[i] = g_cur[i] * factor;
    }
  } else {                              // conjugate direction
    for (int i = 0; i < 3 * nlocal; i++) {
      g2old += g_old[i] * g_old[i];
      g2 += g_cur[i] * g_cur[i];
    }

    // now we need to collect/broadcast beta on this replica
    // need to check what is beta for GNEB

    MPI_Allreduce(&g2,&g2_global,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&g2old,&g2old_global,1,MPI_DOUBLE,MPI_SUM,world);

    // Sum over all replicas. Good for GNEB.

    if (nreplica > 1) {
      g2 = g2_global * factor;
      g2old = g2old_global * factor;
      MPI_Allreduce(&g2,&g2_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      MPI_Allreduce(&g2old,&g2old_global,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }
    if (fabs(g2_global) < 1.0e-60) beta = 0.0;
    else beta = g2_global / g2old_global;

    // calculate conjugate direction

    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = (beta * p_s[i] - g_cur[i]) * factor;
      g_old[i] = g_cur[i] * factor;
    }
  }

  local_iter++;
}

/* ----------------------------------------------------------------------
   rotation of spins along the search direction
---------------------------------------------------------------------- */

void MinSpinCG::advance_spins()
{
  int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double rot_mat[9];    // exponential of matrix made of search direction
  double s_new[3];

  // loop on all spins on proc.

  for (int i = 0; i < nlocal; i++) {
    rodrigues_rotation(p_s + 3 * i, rot_mat);

    // rotate spins

    vm3(rot_mat, sp[i], s_new);
    for (int j = 0; j < 3; j++) sp[i][j] = s_new[j];
  }
}

/* ----------------------------------------------------------------------
  calculate 3x3 matrix exponential using Rodrigues' formula
  (R. Murray, Z. Li, and S. Shankar Sastry,
  A Mathematical Introduction to
  Robotic Manipulation (1994), p. 28 and 30).

  upp_tr - vector x, y, z so that one calculate
  U = exp(A) with A= [[0, x, y],
                      [-x, 0, z],
                      [-y, -z, 0]]
------------------------------------------------------------------------- */

void MinSpinCG::rodrigues_rotation(const double *upp_tr, double *out)
{
  double theta,A,B,D,x,y,z;
  double s1,s2,s3,a1,a2,a3;

  if (fabs(upp_tr[0]) < 1.0e-40 &&
      fabs(upp_tr[1]) < 1.0e-40 &&
      fabs(upp_tr[2]) < 1.0e-40) {

    // if upp_tr is zero, return unity matrix

    for (int k = 0; k < 3; k++) {
      for (int m = 0; m < 3; m++) {
        if (m == k) out[3 * k + m] = 1.0;
        else out[3 * k + m] = 0.0;
      }
    }
    return;
  }

  theta = sqrt(upp_tr[0] * upp_tr[0] +
               upp_tr[1] * upp_tr[1] +
               upp_tr[2] * upp_tr[2]);

  A = cos(theta);
  B = sin(theta);
  D = 1.0 - A;
  x = upp_tr[0]/theta;
  y = upp_tr[1]/theta;
  z = upp_tr[2]/theta;

  // diagonal elements of U

  out[0] = A + z * z * D;
  out[4] = A + y * y * D;
  out[8] = A + x * x * D;

  // off diagonal of U

  s1 = -y * z *D;
  s2 = x * z * D;
  s3 = -x * y * D;

  a1 = x * B;
  a2 = y * B;
  a3 = z * B;

  out[1] = s1 + a1;
  out[3] = s1 - a1;
  out[2] = s2 + a2;
  out[6] = s2 - a2;
  out[5] = s3 + a3;
  out[7] = s3 - a3;

}

/* ----------------------------------------------------------------------
  out = vector^T x m,
  m -- 3x3 matrix , v -- 3-d vector
------------------------------------------------------------------------- */

void MinSpinCG::vm3(const double *m, const double *v, double *out)
{
  for (int i = 0; i < 3; i++) {
    out[i] = 0.0;
    for (int j = 0; j < 3; j++) out[i] += *(m + 3 * j + i) * v[j];
  }
}

/* ----------------------------------------------------------------------
  advance spins
------------------------------------------------------------------------- */

void MinSpinCG::make_step(double c, double *energy_and_der)
{
  double p_scaled[3];
  int nlocal = atom->nlocal;
  double rot_mat[9]; // exponential of matrix made of search direction
  double s_new[3];
  double **sp = atom->sp;
  double der_e_cur_tmp = 0.0;

  for (int i = 0; i < nlocal; i++) {

    // scale the search direction

    for (int j = 0; j < 3; j++) p_scaled[j] = c * p_s[3 * i + j];

    // calculate rotation matrix

    rodrigues_rotation(p_scaled, rot_mat);

    // rotate spins

    vm3(rot_mat, sp[i], s_new);
    for (int j = 0; j < 3; j++) sp[i][j] = s_new[j];
  }

  ecurrent = energy_force(0);
  calc_gradient();
  neval++;
  der_e_cur = 0.0;
  for (int i = 0; i < 3 * nlocal; i++) {
    der_e_cur += g_cur[i] * p_s[i];
  }
  MPI_Allreduce(&der_e_cur,&der_e_cur_tmp,1,MPI_DOUBLE,MPI_SUM,world);
  der_e_cur = der_e_cur_tmp;
  if (update->multireplica == 1) {
    MPI_Allreduce(&der_e_cur_tmp,&der_e_cur,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
  }

  energy_and_der[0] = ecurrent;
  energy_and_der[1] = der_e_cur;
}

/* ----------------------------------------------------------------------
  Calculate step length which satisfies approximate Wolfe conditions
  using the cubic interpolation
------------------------------------------------------------------------- */

int MinSpinCG::calc_and_make_step(double a, double b, int index)
{
  double e_and_d[2] = {0.0,0.0};
  double alpha,c1,c2,c3;
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  make_step(b,e_and_d);
  ecurrent = e_and_d[0];
  der_e_cur = e_and_d[1];
  index++;

  if (adescent(eprevious,e_and_d[0]) || index == 5) {
    MPI_Bcast(&b,1,MPI_DOUBLE,0,world);
    for (int i = 0; i < 3 * nlocal; i++) {
      p_s[i] = b * p_s[i];
    }
    return 1;
  }
  else {
    double r,f0,f1,df0,df1;
    r = b - a;
    f0 = eprevious;
    f1 = ecurrent;
    df0 = der_e_pr;
    df1 = der_e_cur;

    c1 = -2.0*(f1-f0)/(r*r*r)+(df1+df0)/(r*r);
    c2 = 3.0*(f1-f0)/(r*r)-(df1+2.0*df0)/(r);
    c3 = df0;

    // f(x) = c1 x^3 + c2 x^2 + c3 x^1 + c4
    // has minimum at alpha below. We do not check boundaries.

    alpha = (-c2 + sqrt(c2*c2 - 3.0*c1*c3))/(3.0*c1);
    MPI_Bcast(&alpha,1,MPI_DOUBLE,0,world);

    if (alpha < 0.0) alpha = r/2.0;

    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < 3; j++) sp[i][j] = sp_copy[i][j];
    }
    calc_and_make_step(0.0, alpha, index);
   }

  return 0;
}

/* ----------------------------------------------------------------------
  Approximate descent
------------------------------------------------------------------------- */

int MinSpinCG::adescent(double phi_0, double phi_j) {

  double eps = 1.0e-6;

  if (phi_j<=phi_0+eps*fabs(phi_0))
    return 1;
  else
    return 0;
}

/* ----------------------------------------------------------------------
   evaluate max timestep
---------------------------------------------------------------------- */

double MinSpinCG::evaluate_dt()
{
  double dtmax;
  double fmsq;
  double fmaxsqone,fmaxsqloc,fmaxsqall;
  int nlocal = atom->nlocal;
  double **fm = atom->fm;

  // finding max fm on this proc.

  fmsq = fmaxsqone = fmaxsqloc = fmaxsqall = 0.0;
  for (int i = 0; i < nlocal; i++) {
    fmsq = fm[i][0]*fm[i][0]+fm[i][1]*fm[i][1]+fm[i][2]*fm[i][2];
    fmaxsqone = MAX(fmaxsqone,fmsq);
  }

  // finding max fm on this replica

  fmaxsqloc = fmaxsqone;
  MPI_Allreduce(&fmaxsqone,&fmaxsqloc,1,MPI_DOUBLE,MPI_MAX,world);

  // finding max fm over all replicas, if necessary
  // this communicator would be invalid for multiprocess replicas

  fmaxsqall = fmaxsqloc;
  if (update->multireplica == 1) {
    fmaxsqall = fmaxsqloc;
    MPI_Allreduce(&fmaxsqloc,&fmaxsqall,1,MPI_DOUBLE,MPI_MAX,universe->uworld);
  }

  if (fmaxsqall == 0.0)
    error->all(FLERR,"Incorrect fmaxsqall calculation");

  // define max timestep by dividing by the
  // inverse of max frequency by discrete_factor

  dtmax = MY_2PI/(discrete_factor*sqrt(fmaxsqall));

  return dtmax;
}


int MinSpinCG::linemin_backtrack(double eoriginal, double &alpha, double *fvec, double *xvec, double *x0)
{
  int i,m,n;
  double fdothall,fdothme,hme,hmax,hmaxall;
  double de_ideal,de;

  fdothme = 0.0;
  // std::cout << "NNNNNVEC::::   " << nvec << "    " << fvec[nvec-1] << std::endl;
  for (i = 0; i < nvec; i++) fdothme += fvec[i]*h[i];
  MPI_Allreduce(&fdothme,&fdothall,1,MPI_DOUBLE,MPI_SUM,world);
  std::cout << "fdothall:   " << fdothall << std::endl; 
  if (fdothall <= 0.0) return DOWNHILL;

  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme,fabs(h[i]));
  MPI_Allreduce(&hme,&hmaxall,1,MPI_DOUBLE,MPI_MAX,world);
  alpha = MIN(1,0.01/hmaxall);
  // alpha = MIN(1,0.1/hmaxall);
  std::cout << "hmaxall:   " << hmaxall << std::endl;
  if (hmaxall == 0.0) return ZEROFORCE;

  fix_minimize->store_box();
  for (i = 0; i < nvec; i++) x0[i] = xvec[i];
  while (true) {
    ecurrent = alpha_step(alpha,1,x0);
    de_ideal = -BACKTRACK_SLOPE*alpha*fdothall;
    de = ecurrent - eoriginal;
    if (de <= de_ideal) {
      return 0;
    }

    alpha *= ALPHA_REDUCE;

    if (alpha <= 0.0 || de_ideal >= -EMACH) {
      ecurrent = alpha_step(0.0, 0, x0);

      if (de < 0.0) return ETOL;
      else return ZEROALPHA;
    }
  }
}

int MinSpinCG::linemin_qua(double eoriginal, double &alpha, double *fvec, double *xvec, double *x0)
{
  int i,m,n;
  double fdothall,fdothme,hme,hmax,hmaxall;
  double de_ideal,de;
  double delfh,engprev,relerr,alphaprev,fhprev,fh,alpha0;
  double dot[2],dotall[2];
  double *xatom,*x0atom,*fatom,*hatom;
  double alphamax;
  fdothme = 0.0;
  for (i = 0; i < nvec; i++) fdothme += fvec[i]*h[i];
  MPI_Allreduce(&fdothme,&fdothall,1,MPI_DOUBLE,MPI_SUM,world);
  if (fdothall <= 0.0) return DOWNHILL;
  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme,fabs(h[i]));
  MPI_Allreduce(&hme,&hmaxall,1,MPI_DOUBLE,MPI_MAX,world);
  alphamax = MIN(ALPHA_MAX,0.01/hmaxall);
  if (hmaxall == 0.0) return ZEROFORCE;
  fix_minimize->store_box();
  for (i = 0; i < nvec; i++) x0[i] = xvec[i];
  alpha = alphamax;
  fhprev = fdothall;
  engprev = eoriginal;
  alphaprev = 0.0;
  const double hbar = 6.5821191e-04;
  while (true) {
    ecurrent = alpha_step(alpha,1,x0);
    double * fvec = new double[nall*3];
    for(int i=0; i<nall*3; i++){
        if(i<nlocal*3){fvec[i]=f[i];}
        else {fvec[i]=fm[i-nlocal*3]*hbar/0.3*sp[(i-nlocal*3)/3][3]/1.323;}}
    dot[0] = dot[1] = 0.0;
    for (i = 0; i < nvec; i++) {
      dot[0] += fvec[i]*fvec[i];
      dot[1] += fvec[i]*h[i];
    }
    MPI_Allreduce(dot,dotall,2,MPI_DOUBLE,MPI_SUM,world);
    fh = dotall[1];
    delfh = fh - fhprev;
    if (fabs(fh) < EPS_QUAD || fabs(delfh) < EPS_QUAD) {
      ecurrent = alpha_step(0.0,0,x0);
      return ZEROQUAD;
    }
    relerr = fabs(1.0-(0.5*(alpha-alphaprev)*(fh+fhprev)+ecurrent)/engprev);
    alpha0 = alpha - (alpha-alphaprev)*fh/delfh;
    if (relerr <= QUADRATIC_TOL && alpha0 > 0.0 && alpha0 < alphamax) {
      ecurrent = alpha_step(alpha0,1,x0);
      if (ecurrent - eoriginal < EMACH) {
        return 0;
      }
    }
    de_ideal = -BACKTRACK_SLOPE*alpha*fdothall;
    de = ecurrent - eoriginal;
    if (de <= de_ideal) {
      return 0;
    }

    fhprev = fh;
    engprev = ecurrent;
    alphaprev = alpha;
    alpha *= ALPHA_REDUCE;
    if (alpha <= 0.0 || de_ideal >= -EMACH) {
      ecurrent = alpha_step(0.0,0,x0);
      return ZEROALPHA;
    }
  }



}

double MinSpinCG::alpha_step(double alpha, int resetflag, double *x0)
{
  int i,n,m;
  double *xvec_cp = new double[nall*3];
  // std::cout << "alpha_Nvec:   " << nvec << std::endl;
  for (i = 0; i < nvec; i++) xvec_cp[i] = x0[i];
  // std::cout << xvec_cp[16*3] << "  " << x0[0] << "  " << sp[0][3] << "  " << nvec <<std::endl;
  // std::cout << "ALPHA:   "  << alpha << std::endl;
  if (resetflag ==  1.0) {
    for (i = 0; i < nlocal*3; i++) xvec_cp[i] += alpha*h[i];
    for(int i=nlocal*3; i<nall*3; i++) xvec_cp[i] += alpha*h[i];

    for(int i=nlocal*3; i<nall*3; i++){
      sp[(i-nlocal*3)/3][(i-nlocal*3)%3] = xvec_cp[i] - x0[i-nlocal*3];
    }
    for(int i=nlocal; i<nall; i++){
      double sp_norm  = sqrt(sp[i-nlocal][0]*sp[i-nlocal][0] + sp[i-nlocal][1]*sp[i-nlocal][1] + sp[i-nlocal][2]*sp[i-nlocal][2]);
      sp[i-nlocal][3] = 1.323*sp_norm/0.1;
      for(int j=0; j<3; j++) sp[i-nlocal][j] /=  sp_norm;
    }
    for(int i=0; i<nlocal*3; i++){
      x[i] = xvec_cp[i];
    }
  } else {
    for (i = 0; i < nvec; i++) xvec_cp[i] += 0.01*h[i];
    for(int i=nlocal*3; i<nall*3; i++){
      sp[(i-nlocal*3)/3][(i-nlocal*3)%3] = x0[i] - x0[i-nlocal*3];
    }
    for(int i=nlocal; i<nall; i++){
      double sp_norm  = sqrt(sp[i-nlocal][0]*sp[i-nlocal][0] + sp[i-nlocal][1]*sp[i-nlocal][1] + sp[i-nlocal][2]*sp[i-nlocal][2]);

      // sp[i-nlocal][3] = 1.323*sp_norm/0.1;
      for(int j=0; j<3; j++) sp[i-nlocal][j] /=  sp_norm;
    }
    for(int i=0; i<nlocal*3; i++){
      x[i] = xvec_cp[i];
    }
  
  }
  neval++;
  return energy_force(resetflag);
}
