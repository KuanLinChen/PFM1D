
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
  UpdateTransportCoefficients() ;
  cout<<"UpdateTransportCoefficients"<<endl;
  InitialCondition() ;
  cout<<"InitialCondition"<<endl;

  for ( int iCycle=0 ; iCycle <   1 ; iCycle++ ){
    for (int iStep=0 ; iStep < 100 ; iStep++ ) {

      Poisson_eqn( 0.0, 0.0 ) ;
      ComputeElectricField( 0.0, 0.0 ) ;
      electron_continuity_eqn() ;
    }//End step
    output("flow-"+to_string(iCycle)+".dat" ) ;
  }//End cycle



  PetscEnd();
  //convection_diffusion_eqn(); 
  cout<<"convection_diffusion_eqn"<<endl;


  

  PetscFinalize();
  return 0 ;
}


