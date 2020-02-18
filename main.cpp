
/* This is a example code for 1d plasma simulation. */
#include "PFM.hpp"

using namespace std ;
int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,(char*)0,NULL);

  CreatePetscDMDA_1D() ;
  CreateMesh_1D() ;
  cout<<"CreateMesh_1D"<<endl;

  InitializeLinearSystemSolver() ;
  cout<<"InitializeLinearSystemSolver"<<endl;
  InitializePetscVector();
  cout<<"InitializePetscVector"<<endl;
  convection_diffusion_eqn(); 
  cout<<"convection_diffusion_eqn"<<endl;
  //PetscEnd();
  Poisson_eqn(100.0) ;
  
  output("flow2.dat");

  PetscFinalize();
  return 0 ;
}


