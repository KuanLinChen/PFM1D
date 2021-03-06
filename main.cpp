
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

  PetscScalar voltage=0.0, PhyicalTime=0.0 ;


	for ( int iCycle=1 ; iCycle < nCycle ; iCycle++ ) {
		PetscPrintf(PETSC_COMM_WORLD, "Cyele %d\n", iCycle ) ;
    for ( int iStep=0 ;  iStep <  nStep ;  iStep++ ) {

      voltage = V*sin(2.0*PI*(f)*iStep*dt) ;
      //cout<<voltage<<endl;
      UpdateSourceTerm() ;
      //
      Poisson_eqn( voltage, 0.0 ) ;
      //
      ComputeElectricField( voltage, 0.0 ) ;
      //
      electron_continuity_eqn() ;
      electron_flux() ;
      electron_energy_density_eqn();
      Compute_Te();
      //
      ion_continuity_eqn() ;

      PhyicalTime += DTime ; 

      if ( (iStep)%(nStep/Output_Step) == 0 and (iCycle)%Output_Cycle == 0 ){
        cout<<"out"<<endl;
        output( "./output/flow_"+to_string(iCycle)+"_"+to_string(iStep)+".dat" ) ;
      }

    }//End step

  }//End cycle



  PetscEnd();
  //convection_diffusion_eqn(); 
  cout<<"convection_diffusion_eqn"<<endl;


  

  PetscFinalize();
  return 0 ;
}


