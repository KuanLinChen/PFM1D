#include <petsc.h>
#include <fstream>
#include <stdio.h>
#include <iostream>
using namespace std;
typedef struct {
	Mat A ;
	Vec B ;
	Vec x ;
	KSP ksp ;
	PetscReal nrm ;
} LinearSysSolver; //Format of data output

typedef struct {
	PetscScalar sign_q, mass ; //Species chearge.
	Vec U0, //nunmber density.
			U1, //flux in x-direction.
			U2, //energy flux.
			T	; //Temperature

	Vec  D,  //Diffusion coefficient.
			Mu,  //Mobility 
			rate;//rate constant.
} Species ;

/*--- Mesh & PETSc DMDA parameters ---*/
PetscInt nCell = 200     ;      // Number of cells.
PetscInt nNode = nCell+1 ;      // Number of node. (calculated from nCell)
PetscScalar mesh_factor = 5.0 ; //the grid point will close to the wall wnen value is large.

PetscScalar *x,         // Node coordinate.
						*xc,        // Cell coordinate.
						*dx ;       // Cell width 

PetscInt overlope = 1 ; // Overlope cell in processor boundary.
PetscInt dof = 1 ;

/*--- Simulation & Physical parameters of the problem.  ---*/
PetscScalar gap_length        =      0.02 ; //gap distance
PetscScalar Pressure_in_torr  =       1.0 ; //Chamber pressure.
PetscScalar Frequency         =  13.56E+6 ; //Applied frequency [Hz].
PetscScalar Amplitude         =     40.0  ; //Applied voltage [V]. (half of peak-to-peak voltage).
PetscScalar T_background      =     293.0 ;
PetscScalar N_background      = Pressure_in_torr*133.32/1.38064880E-23/T_background ;

PetscInt nCycle = 51    ;  //Number of cycle you want to simulated.
PetscInt Output_Cycle = 50    ;  //Number of cycle you want to simulated.
PetscInt nStep  = 2000 ;  //Number of step within a cycle.
PetscInt Output_Step  = 4 ;  //Number of step within a cycle.

PetscScalar DTime=(1.0/Frequency)/(nStep) ; //Time step size [s].


/*--- Physical parameter ---*/
PetscScalar Qe       = 1.60217657E-19 ;
PetscScalar Kb       = 1.38064880E-23 ;
PetscScalar epsilon0 = 8.85418800E-12 ;
PetscScalar Na       = 6.02214129E+23 ;
PetscScalar PI       = 4.0*atan(1.0)  ;

/*--- non-dimensional parameters ---*/
PetscScalar Mu_ref  = 1.0 ;  
PetscScalar L_ref   = 1.0 ;//gap_length ;
PetscScalar Phi_ref = 1.0 ;//Amplitude  ; 
PetscScalar Qe_ref  = 1.0 ;//Qe  ; 
PetscScalar epslion_ref = 1.0 ;//epsilon0
//
PetscScalar  n_ref   = epslion_ref*Phi_ref/Qe_ref/L_ref/L_ref ; // reference number density
PetscScalar  D_ref   = Mu_ref*Phi_ref ; // reference diffusion
PetscScalar en_ref   = Qe_ref*Phi_ref*n_ref ; // reference energy density
PetscScalar Time_ref = L_ref*L_ref/D_ref ; //reference time   
PetscScalar  T_ref   = en_ref*n_ref ; // reference temperature
PetscScalar  k_ref   = 1.0/n_ref/Time_ref ; // reference rate constant.
PetscScalar  f_ref   = 1.0/Time_ref ; // reference time.
PetscScalar  E_ref   = Phi_ref/L_ref ; //reference electric field.  

//for simulation.
PetscScalar dt = DTime/Time_ref ;
PetscScalar V  = Amplitude/Phi_ref ;
PetscScalar f  = Frequency/f_ref ;

/*--- MPI & Petsc Distribution Memory ---*/
DM da ;
DMDALocalInfo da_info ;

PetscMPIInt mpi_rank, /* Processor id. */
						mpi_size; /* number of nodes */

/* Declaration of Linear system Solver */
LinearSysSolver poisson, electron_continuity, electron_energy, ion_continuity ;

/* Declaration of solution variables */
Vec potential, Ex, Ee, Ew, gVec, gField[2], Rdot, InelasticLoss, JouleHeating ;
Species ele, ion ;


/*--- Declaration of functions ---*/
void CreatePetscDMDA_1D();
void CreateMesh_1D();
void InitializeLinearSystemSolver();
void InitializePetscVector();
void Poisson_eqn( PetscScalar, PetscScalar ) ;
void UpdateSourceTerm();
void electron_continuity_eqn();
void electron_flux();
void ion_continuity_eqn();
void Compute_Te();
void electron_energy_density_eqn();
void output(string);
PetscScalar f1( PetscScalar ) ;
PetscScalar f2( PetscScalar ) ;


void UpdateTransportCoefficients() ;

/*--- functions ---*/
void CreatePetscDMDA_1D()
{
	MPI_Comm_size( PETSC_COMM_WORLD, &mpi_size ) ; 
	MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_rank ) ;  
	DMDACreate1d ( PETSC_COMM_WORLD, DM_BOUNDARY_NONE, nCell, dof, 1, NULL, &da ) ;
	DMSetUp( da ) ;

	DMDAGetLocalInfo( da, &da_info ) ;

	/* print dmda informations. */
	#if false
		PetscSynchronizedPrintf(PETSC_COMM_SELF, "global number of grid points in x : %d in rank: %d \n", da_info.mx, mpi_rank ) ;
		PetscSynchronizedPrintf(PETSC_COMM_SELF, "starting point of processor [%d] (excluding ghosts): %d\n", mpi_rank, da_info.xs ) ;
		PetscSynchronizedPrintf(PETSC_COMM_SELF, "number of grid points on processor [%d] (excluding ghosts): %d\n", mpi_rank, da_info.xm ) ;
		PetscSynchronizedFlush(PETSC_COMM_SELF,PETSC_STDOUT) ;
	#endif
}

void CreateMesh_1D()
{
	/* Note: all processor have same mesh informations. */
	x  = new PetscScalar [nNode] ;
	xc = new PetscScalar [nCell] ;
	dx = new PetscScalar [nCell] ;
	
	/* hyperbolic tangent stretching scheme */
	for ( PetscInt i = 0 ; i < nNode ; i++ ) {
		x[ i ] = 0.5*gap_length*( 1.0 + tanh( mesh_factor*( (PetscScalar)i/nCell-0.5 ) ) /tanh(0.5*mesh_factor) ) ;
		//PetscPrintf(PETSC_COMM_SELF, " %15.6e \t  3 \n", x[i] ) ;
	}
	//normalized 
	for ( PetscInt i = 0 ; i < nNode ; i++ ) {
		x[ i ] = x[ i ]/ L_ref ;
	} 

	/* cell center. */
	for( PetscInt i= 0 ; i < nCell ; i++ ) {
		xc[ i ] = 0.5*( x[i]+x[i+1] ) ;
		dx[ i ] = x[i+1] - x[i] ; 
	}
	//PetscEnd() ;
}
void InitializeLinearSystemSolver()
{
	//ksp
	KSPCreate( PETSC_COMM_WORLD, &poisson.ksp ) ;
	KSPCreate( PETSC_COMM_WORLD, &electron_continuity.ksp ) ;
	KSPCreate( PETSC_COMM_WORLD, &ion_continuity.ksp ) ;

	KSPCreate( PETSC_COMM_WORLD, &electron_energy.ksp ) ;

	//Matrix A
	DMCreateMatrix( da, &poisson.A ) ; 
	DMCreateMatrix( da, &electron_continuity.A ) ; 
	DMCreateMatrix( da, &ion_continuity.A ) ; 

	DMCreateMatrix( da, &electron_energy.A ) ; 

	//Source term vector B 
	DMCreateGlobalVector( da, &poisson.B ) ;

	DMCreateGlobalVector( da, &electron_continuity.B ) ;
	DMCreateGlobalVector( da, &ion_continuity.B ) ;

	DMCreateGlobalVector( da, &electron_energy.B ) ;

	//Solution vector
	DMCreateGlobalVector( da, &poisson.x ) ;
	DMCreateGlobalVector( da, &electron_continuity.x ) ;
	DMCreateGlobalVector( da, &ion_continuity.x ) ;

	DMCreateGlobalVector( da, &electron_energy.x ) ;

	//SetUp Ksp
	KSPSetOperators( poisson.ksp, poisson.A, poisson.A ) ;
	KSPSetOperators( electron_continuity.ksp, electron_continuity.A, electron_continuity.A ) ;
	KSPSetOperators( ion_continuity.ksp, ion_continuity.A, ion_continuity.A ) ;
	KSPSetOperators( electron_energy.ksp, electron_energy.A, electron_energy.A ) ;

	//Get input options
	KSPSetFromOptions( poisson.ksp ) ;
	KSPSetFromOptions( electron_continuity.ksp ) ;
	KSPSetFromOptions( ion_continuity.ksp ) ;
	KSPSetFromOptions( electron_energy.ksp ) ;

}
void InitializePetscVector()
{
	DMCreateGlobalVector( da, &gVec ) ;
	DMCreateGlobalVector( da, &gField[0] ) ;
	DMCreateGlobalVector( da, &gField[1] ) ;


	DMCreateGlobalVector( da, &Rdot ) ;
	DMCreateGlobalVector( da, &InelasticLoss ) ;
	DMCreateGlobalVector( da, &JouleHeating ) ;

	//electrostatic 
	DMCreateLocalVector ( da, &potential ) ;//cell center potential.
	DMCreateLocalVector  ( da,  &Ex ) ;//cell center electric field.
	DMCreateGlobalVector ( da,  &Ee ) ;//face electric field.
	DMCreateGlobalVector ( da,  &Ew ) ;//face electric field.

	//Electron
	ele.sign_q = -1.0 ;
	ele.mass = 9.10938291E-31 ;
	DMCreateLocalVector ( da, &ele.U0 ) ;//Number density of electron at cell.
	DMCreateLocalVector ( da, &ele.U1 ) ;//Number density of electron at cell.
	DMCreateLocalVector ( da, &ele.U2 ) ;//Energy density of electron at cell.
	DMCreateLocalVector ( da, &ele. T ) ;//temperature of electron at cell.
	DMCreateLocalVector ( da, &ele.Mu ) ;//Mobility of electron at cell.
	DMCreateLocalVector ( da, &ele.D  ) ;//Diffusion of electron at cell.

	//Ion
	ion.sign_q =  1.0 ;
	ion.mass = (39.948)/Na/1000 ; //Argon AMU/Na/1000
	DMCreateLocalVector ( da, &ion.U0 ) ;//Number density of ion at cell.
	DMCreateLocalVector ( da, &ion. T ) ;//temperature of ion at cell.
	DMCreateLocalVector ( da, &ion.Mu ) ;//Mobility of ion at cell.
	DMCreateLocalVector ( da, &ion.D  ) ;//Diffusion of ion at cell.

}
void InitialCondition()
{
	PetscScalar N0 = 1.0E+15, Te0 = 2.0 ;
	VecSet(               ele.U0, N0/n_ref ) ; 
	VecSet(electron_continuity.x, N0/n_ref  ) ;
	//
	VecSet( 					ele.U2, N0*Te0*Qe/en_ref ) ;
	VecSet(electron_energy.x, N0*Te0*Qe/en_ref ) ;
	//
	VecSet( ele.T,   Te0/T_ref ) ;

	VecSet(          ion.U0, N0/n_ref ) ; 
	VecSet(ion_continuity.x, N0/n_ref  ) ;// I want to reuse the global in linear solver, so I keep it in old solution.
	VecSet( ion.T,  0.026/T_ref ) ;

	/* Print simulation conditions */
	PetscPrintf(PETSC_COMM_SELF, " /*----------Simulation Conditions ----------*/ \n" ) ;
	PetscPrintf(PETSC_COMM_SELF, "Background pressuer         : %15.6e Torr\n", Pressure_in_torr  ) ;
	PetscPrintf(PETSC_COMM_SELF, "Background temperature      : %15.6e    K\n", T_background      ) ;
	PetscPrintf(PETSC_COMM_SELF, "Background number density   : %15.6e m^-3\n", N_background      ) ;
	//
	PetscPrintf(PETSC_COMM_SELF, "Gap distance                : %15.6e   m \n", gap_length ) ;
	PetscPrintf(PETSC_COMM_SELF, "Applied frequency           : %15.6e MHz \n", Frequency  ) ;
	PetscPrintf(PETSC_COMM_SELF, "Applied voltage (Amplitude) : %15.6e V   \n", Amplitude  ) ;
	PetscPrintf(PETSC_COMM_SELF, "Time step size              : %15.6e Sec.\n", DTime  ) ;

	PetscPrintf(PETSC_COMM_SELF, "\n /*----------Non-dimensional parameters ----------*/ \n" ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference Mobility          : %15.6e \n",  Mu_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference Diffusion         : %15.6e \n",   D_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference Length            : %15.6e \n",   L_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference Potential         : %15.6e \n", Phi_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference Qe                : %15.6e \n",  Qe_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference density           : %15.6e \n",   n_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference energy density    : %15.6e \n",  en_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference time              : %15.6e \n",Time_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference rate constant     : %15.6e \n",   k_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference frequency         : %15.6e \n",   f_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference temperature       : %15.6e \n",   T_ref ) ;
	PetscPrintf(PETSC_COMM_SELF, "Reference electric field    : %15.6e \n",   E_ref ) ;
//PetscEnd();
}
void UpdateTransportCoefficients()
{
	//Davoudabadi, M., Shrimpton, J., and Mashayek, F., “On accuracy and performance of high-order finite volume methods in local 
	//          mean energy model of non-thermal plasmas,” Journal of Computational Physics, vol. 228, no. 7, pp. 2468-2479, 2009.
	VecSet( ele.Mu,   30.0/Pressure_in_torr/Mu_ref ) ;
	VecSet( ele.D ,  120.0/Pressure_in_torr/ D_ref ) ;

	VecSet( ion.Mu,   0.14/Pressure_in_torr/Mu_ref ) ;
	VecSet( ion.D , 4.0E-3/Pressure_in_torr/ D_ref ) ;

}
void UpdateSourceTerm()
{
	//Davoudabadi, M., Shrimpton, J., and Mashayek, F., “On accuracy and performance of high-order finite volume methods in local 
	//          mean energy model of non-thermal plasmas,” Journal of Computational Physics, vol. 228, no. 7, pp. 2468-2479, 2009.
	PetscScalar *Ne, *R_dot, *Te, *inelastic_loss ;
	PetscScalar ionization_energy = 15.578*Qe ; //[J]
	PetscScalar Te_unit=0.0, RateConstant=0.0 ;

	DMDAVecGetArray( da, ele.U0, &Ne ) ;
	DMDAVecGetArray( da, ele.T , &Te ) ;
	DMDAVecGetArray( da, Rdot, &R_dot ) ;
	DMDAVecGetArray( da, InelasticLoss, &inelastic_loss ) ;
	PetscScalar Energy = 5.3 ;
	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {

		Te_unit = Te[i]*T_ref ;
		//Ne_unit = Ne[i]*n_ref ;

		if( Te_unit > Energy ) {
			RateConstant = (8.7E-15)*(Te_unit-Energy)*exp(-4.9/sqrt(Te_unit-Energy) )/k_ref ;
		} else {
			RateConstant = 0.0 ;
		}

		R_dot[i] = RateConstant*Ne[i]*(N_background/n_ref);

		inelastic_loss[i] = R_dot[i]*(ionization_energy/T_ref) ;
	}
	DMDAVecRestoreArray( da, ele.U0, &Ne ) ;
	DMDAVecRestoreArray( da, ele.T , &Te ) ;
	DMDAVecRestoreArray( da, Rdot, &R_dot ) ;
	DMDAVecRestoreArray( da, InelasticLoss, &inelastic_loss ) ;
}
void Poisson_eqn( PetscScalar left_voltage, PetscScalar right_voltage )
{
	MatStencil  row, col[3] ;
	PetscScalar v[3], d1, d2, *source, *Ni, *Ne ;

	DMDAVecGetArray( da, poisson.B, &source ) ;
	DMDAVecGetArray( da, ele.U0, &Ne ) ;
	DMDAVecGetArray( da, ion.U0, &Ni ) ;

	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		source[ i ] = 0.0 ;
		row.i = i ;

		for ( int k=0 ; k < 3 ; k++ ) v[k]=0.0;

		if ( i==0 ) {

			col[0].i = i   ;
			col[1].i = i+1 ;

			//eFace 
			d1  = 0.5*dx[ i ] ;
			d2 = 0.5*dx[i+1] ;
			v[0] += -1.0/(d1+d2) ;
			v[1] +=  1.0/(d1+d2) ;

			//wFace 
			d2 = 0.5*dx[ i ] ;
			v[0] +=  -1.0/(d2) ;

			MatSetValuesStencil( poisson.A, 1, &row, 2, col, v, INSERT_VALUES ) ;
			source[ i ] = -1.0/(d1)*left_voltage ;

		}else if( i == nCell-1 ) {

			col[0].i = i-1 ;
			col[1].i = i   ;

			//wFace 
			d1  = 0.5*dx[i-1] ;
			d2 = 0.5*dx[ i ] ;
			v[0] +=  1.0/(d1+d2) ;
			v[1] += -1.0/(d1+d2) ;

			//eFace 
			d1  = 0.5*dx[ i ] ;
			v[1] +=  -1.0/(d1) ;

			MatSetValuesStencil( poisson.A, 1, &row, 2, col, v, INSERT_VALUES ) ;
			source[ i ] = -1.0/(d1)*right_voltage ;

		} else {


			col[0].i = i-1 ;
			col[1].i = i   ;
			col[2].i = i+1 ;

			//wFace 
			d1  = 0.5*dx[i-1] ;
			d2 = 0.5*dx[ i ] ;
			v[0] +=  1.0/(d1+d2) ;
			v[1] += -1.0/(d1+d2) ;


			//eFace 
			d1  = 0.5*dx[ i ] ;
			d2 = 0.5*dx[i+1] ;
			v[1] += -1.0/(d1+d2) ;
			v[2] +=  1.0/(d1+d2) ;
			MatSetValuesStencil( poisson.A, 1, &row, 3, col, v, INSERT_VALUES ) ;
		}

		source[ i ] += -Qe*( Ni[i]-Ne[i] )/epsilon0*dx[i] ;
		//source[ i ] /= epsilon0 ;

	}

	DMDAVecRestoreArray( da, poisson.B, &source ) ;
	DMDAVecRestoreArray( da, ele.U0, &Ne ) ;
	DMDAVecRestoreArray( da, ion.U0, &Ni ) ;

	MatAssemblyBegin(poisson.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(poisson.A, MAT_FINAL_ASSEMBLY);
	KSPSolve( poisson.ksp, poisson.B, poisson.x );

	/* Update ghost cell value */
	DMGlobalToLocal( da, poisson.x, INSERT_VALUES, potential ) ;
}
void ComputeElectricField( PetscScalar left_voltage, PetscScalar right_voltage )
{
	PetscScalar *phi, *E_e, *E_w, *E_c, d1, d2 ;

	DMDAVecGetArray( da, potential, &phi ) ;

	DMDAVecGetArray( da,      gVec, &E_c ) ;
	DMDAVecGetArray( da, gField[0], &E_e ) ;
	DMDAVecGetArray( da, gField[1], &E_w ) ;

	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {

		if ( i==0 ) {
			//
			d1 = 0.5*dx[ i ] ;
			d2 = 0.5*dx[i+1] ;
			E_e[ i ] = -( phi[i+1] - phi[ i ] )/( d1+d2 ) ;
			//
			d1 = 0.0 ;
			d2 = 0.5*dx[ i ] ;
			E_w[ i ] = -( phi[ i ] - left_voltage )/( d1+d2 ) ;
			//
			E_c[ i ] = 0.5*( E_w[ i ] + E_e[ i ] ) ;

		}else if( i == nCell-1 ) {
			//
			d1 = 0.5*dx[ i ] ;
			d2 = 0.0 ;
			E_e[ i ] = -( right_voltage - phi[ i ] )/( d1+d2 ) ;
			//
			d1 = 0.5*dx[i-1] ;
			d2 = 0.5*dx[ i ] ;
			E_w[ i ] = -( phi[ i ] - phi[i-1] )/( d1+d2 ) ;
			//
			E_c[ i ] = 0.5*( E_w[ i ] + E_e[ i ] ) ;

		} else {
			//
			d1 = 0.5*dx[ i ] ;
			d2 = 0.5*dx[i+1] ;
			E_e[ i ] = -( phi[i+1] - phi[ i ] )/( d1+d2 ) ;
			//
			d1 = 0.5*dx[i-1] ;
			d2 = 0.5*dx[ i ] ;
			E_w[ i ] = -( phi[ i ] - phi[i-1] )/( d1+d2 ) ;
			//
			E_c[ i ] = 0.5*( E_w[ i ] + E_e[ i ] ) ;
		}
	}

	DMDAVecRestoreArray( da, potential, &phi ) ;
	DMDAVecRestoreArray( da,      gVec, &E_c ) ;
	DMDAVecRestoreArray( da, gField[0], &E_e ) ;
	DMDAVecRestoreArray( da, gField[1], &E_w ) ;
	/* Update ghost cell value */
	DMGlobalToLocal( da, gVec, INSERT_VALUES, Ex ) ;
	DMGlobalToLocal( da, gField[0], INSERT_VALUES, Ee ) ;
	DMGlobalToLocal( da, gField[1], INSERT_VALUES, Ew ) ;
}
PetscScalar f1( PetscScalar XX )
{
	PetscScalar B=0.0 ;
	if( fabs(XX) < 1.0E-10 ) 
	{
		B = 1.0 ;
	} else {
		B = XX / ( exp(XX) - 1.0 ) ;
	}
	return B ;
}
PetscScalar f2( PetscScalar XX )
{
	PetscScalar B=0.0 ;
	if( fabs(XX) < 1.0E-10 ) 
	{
		B = 1.0 ;
	} else {
		B = XX*exp(XX) / ( exp(XX) - 1.0 ) ;
	}
	return B ;
}
void electron_continuity_eqn()
{
	PetscScalar sign_q = ele.sign_q ;
	MatStencil  row, col[3] ;
	PetscScalar v[3], *s,  *E_e, *E_w, *D, *Mu, *solution_old, *T, *R_dot ;
	PetscScalar d1, d2, D_face, Mu_face, X ;
	//
	MatZeroEntries( electron_continuity.A ) ;
	//
	DMDAVecGetArray( da, electron_continuity.B, &s ) ;
	DMDAVecGetArray( da, electron_continuity.x, &solution_old ) ;
	DMDAVecGetArray( da, ele.T, &T ) ;

	//
	DMDAVecGetArray( da, Ee, &E_e ) ;
	DMDAVecGetArray( da, Ew, &E_w ) ;
	//
	DMDAVecGetArray( da, ele.Mu, &Mu ) ;
	DMDAVecGetArray( da, ele.D ,  &D ) ;
	//
	DMDAVecGetArray( da, Rdot, &R_dot ) ;

	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		row.i = i ;
		 v[0] = 0.0 ;
		 v[1] = 0.0 ;
		 v[2] = 0.0 ;
		 s[i] = 0.0 ;
		if ( i==0 ) {

			col[0].i = i  ;
			col[1].i = i+1;

			/*--- Unsteady ---*/
			v [0] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[0] += -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[1] += -D_face/(d1+d2)*( f1(X) ) ;

			/*--- w-face ---*/
			v[0] += -D[i]/d1 ;

			MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		}else if( i == nCell-1 ) {

			col[0].i = i-1;
			col[1].i = i  ;

			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;


			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -D_face/(d1+d2)*( f2( X) ) ;
			//i
			v[1] += -D_face/(d1+d2)*(-f1(-X) ) ;

			/*--- e-face ---*/
			v[1] += D[i]/d1 ;

			MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		} else {

			col[0].i = i-1 ;
			col[1].i = i   ;
			col[2].i = i+1 ;
			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;


			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[1] += -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[2] += -D_face/(d1+d2)*( f1(X) ) ;
			//cout<<"fe: "<<v[2]<<"\t";

			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; 
			d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -D_face/(d1+d2)*( f2(X) ) ;
			//i
			v[1] += -D_face/(d1+d2)*(-f1(X) ) ;
			//cout<<"1: "<<v[0]<<" "<<v[1]<<" "<<v[2]<<endl;
			// cout<<"fw: "<<v[0]<<"\t";
			// cout<<"fc: "<<v[1]<<"\t"<<1.0-v[2]-v[1] <<endl;
			MatSetValuesStencil( electron_continuity.A, 1, &row, 3, col, v, INSERT_VALUES ) ;
		}
		/* Chemical */
		s[ i ]+= R_dot[i]*dx[i] ;
	}
	//cout<<"end"<<endl;
	DMDAVecRestoreArray( da, electron_continuity.B, &s ) ;
	DMDAVecRestoreArray( da, electron_continuity.x, &solution_old ) ;
	DMDAVecRestoreArray( da, ele.T, &T ) ;

	DMDAVecRestoreArray( da, ele.Mu, &Mu ) ;
	DMDAVecRestoreArray( da, ele.D ,  &D ) ;

	DMDAVecRestoreArray( da, Ee, &E_e ) ;
	DMDAVecRestoreArray( da, Ew, &E_w ) ;
	//
	DMDAVecRestoreArray( da, Rdot, &R_dot ) ;

	MatAssemblyBegin(electron_continuity.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(electron_continuity.A, MAT_FINAL_ASSEMBLY);

	KSPSolve( electron_continuity.ksp, electron_continuity.B, electron_continuity.x );
	DMGlobalToLocal( da, electron_continuity.x, INSERT_VALUES, ele.U0 ) ;
}
void electron_flux()
{
	PetscScalar sign_q = ele.sign_q ;
	PetscScalar v[3], *E_e, *E_w, *D, *Mu, *Ne, *Flux ;
	PetscScalar d1, d2, D_face, Mu_face, X, Fe, Fw ;

	DMDAVecGetArray( da, ele.U0, &Ne ) ;
	DMDAVecGetArray( da, ele.U1, &Flux ) ;
	//
	DMDAVecGetArray( da, Ee, &E_e ) ;
	DMDAVecGetArray( da, Ew, &E_w ) ;
	//
	DMDAVecGetArray( da, ele.Mu, &Mu ) ;
	DMDAVecGetArray( da, ele.D ,  &D ) ;
	//

	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {

		if ( i==0 ) {

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[0] = -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[1] = -D_face/(d1+d2)*( f1(X) ) ;
			Fe = v[0]*Ne[i] + v[1]*Ne[i+1] ;

			/*--- w-face ---*/
			v[0] = -D[i]/d1 ;
			Fw=v[0]*Ne[i] ;

			Flux[i] = 0.5*(Fw+Fe) ;

		}else if( i == nCell-1 ) {



			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] = -D_face/(d1+d2)*( f2( X) ) ;
			//i
			v[1] = -D_face/(d1+d2)*(-f1(-X) ) ;
			Fw = v[0]*Ne[i-1] + v[1]*Ne[i] ;

			/*--- e-face ---*/
			v[1] = D[i]/d1 ;
			Fe = v[1]*Ne[i] ;

			Flux[i] = 0.5*(Fe+Fw) ;

		} else {

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[1] = -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[2] = -D_face/(d1+d2)*( f1(X) ) ;
			Fe = v[1]*Ne[i] + v[2]*Ne[i+1] ;


			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; 
			d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] = -D_face/(d1+d2)*( f2(X) ) ;
			//i
			v[1] = -D_face/(d1+d2)*(-f1(X) ) ;
			Fw = v[0]*Ne[i-1] + v[1]*Ne[i] ;

			Flux[i] = 0.5*(Fe+Fw) ;

		}
	}

	DMDAVecRestoreArray( da, ele.U0, &Ne ) ;
	DMDAVecRestoreArray( da, ele.U1, &Flux ) ;
	//
	DMDAVecRestoreArray( da, Ee, &E_e ) ;
	DMDAVecRestoreArray( da, Ew, &E_w ) ;
	//
	DMDAVecRestoreArray( da, ele.Mu, &Mu ) ;
	DMDAVecRestoreArray( da, ele.D ,  &D ) ;
	//
}
void ion_continuity_eqn()
{
	PetscScalar sign_q = ion.sign_q ;
	MatStencil  row, col[3] ;
	PetscScalar v[3], *s,  *E_e, *E_w, *D, *Mu, *solution_old, *T, *R_dot ;
	PetscScalar d1, d2, D_face, Mu_face, X ;
	//
	MatZeroEntries( ion_continuity.A ) ;
	//
	DMDAVecGetArray( da, ion_continuity.B, &s ) ;
	DMDAVecGetArray( da, ion_continuity.x, &solution_old ) ;
	DMDAVecGetArray( da, ion.T, &T ) ;
	//
	DMDAVecGetArray( da, Ee, &E_e ) ;
	DMDAVecGetArray( da, Ew, &E_w ) ;
	//
	DMDAVecGetArray( da, ion.Mu, &Mu ) ;
	DMDAVecGetArray( da, ion.D ,  &D ) ;
	//
	DMDAVecGetArray( da, Rdot ,  &R_dot ) ;


	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {

		row.i = i ;
		for ( PetscInt k=0 ; k < 3 ; k++ ) v[k] = 0.0 ;

		s[i] = 0.0 ;

		if ( i==0 ) {

			col[0].i = i  ;
			col[1].i = i+1;

			/*--- Unsteady ---*/
			v [0] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;


			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 

			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[0] += -D_face/(d1+d2)*(-f1(X) ) ;
			//i+1
			v[1] += -D_face/(d1+d2)*( f2(X) ) ;

			/*--- w-face ---*/
			//v[0] += max( 0.0, sign_q*Mu[i]*E_w[i]*(-1.0) );

			MatSetValuesStencil( ion_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		}else if( i == nCell-1 ) {

			col[0].i = i-1;
			col[1].i = i  ;

			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- e-face ---*/
			//v[1] += max( 0.0, sign_q*Mu[i]*E_e[i]*(1.0));

			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; 
			d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -D_face/(d1+d2)*( f2( X) ) ;
			//i
			v[1] += -D_face/(d1+d2)*(-f1(-X) ) ;

			MatSetValuesStencil( ion_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		} else {

			col[0].i = i-1 ;
			col[1].i = i   ;
			col[2].i = i+1 ;
			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[1] += -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[2] += -D_face/(d1+d2)*( f1(X) ) ;

			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; 
			d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -D_face/(d1+d2)*( f2(X) ) ;
			//i
			v[1] += -D_face/(d1+d2)*(-f1(X) ) ;
			//cout<<"1: "<<v[0]<<" "<<v[1]<<" "<<v[2]<<endl;
			MatSetValuesStencil( ion_continuity.A, 1, &row, 3, col, v, INSERT_VALUES ) ;
		}
		/* Chemical */
		s[ i ]+= R_dot[i]*dx[i] ;
	}

	DMDAVecRestoreArray( da, ion_continuity.B, &s ) ;
	DMDAVecRestoreArray( da, ion_continuity.x, &solution_old ) ;
	DMDAVecRestoreArray( da, ion.T, &T ) ;
	//
	DMDAVecRestoreArray( da, ion.Mu, &Mu ) ;
	DMDAVecRestoreArray( da, ion.D ,  &D ) ;
	//
	DMDAVecRestoreArray( da, Ee, &E_e ) ;
	DMDAVecRestoreArray( da, Ew, &E_w ) ;
	//
	DMDAVecRestoreArray( da, Rdot ,  &R_dot ) ;

	MatAssemblyBegin(ion_continuity.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(ion_continuity.A, MAT_FINAL_ASSEMBLY);

	KSPSolve( ion_continuity.ksp, ion_continuity.B, ion_continuity.x );
	DMGlobalToLocal( da, ion_continuity.x, INSERT_VALUES, ion.U0 ) ;
}
void electron_energy_density_eqn()
{
	PetscScalar C53 = 5.0/3.0 ;
	PetscScalar sign_q = ele.sign_q ;
	MatStencil  row, col[3] ;
	PetscScalar v[3], *s,  *E_e, *E_w, *D, *Mu, *solution_old, *T, *inelastic_loss, *E_x, *Flux, *JHeating ;
	PetscScalar d1, d2, D_face, Mu_face, X ;
	//
	MatZeroEntries( electron_energy.A ) ;
	//
	DMDAVecGetArray( da, electron_energy.B, &s ) ;
	DMDAVecGetArray( da, electron_energy.x, &solution_old ) ;
	DMDAVecGetArray( da, ele.T, &T ) ;

	DMDAVecGetArray( da, ele.U1, &Flux ) ;
	DMDAVecGetArray( da, InelasticLoss, &inelastic_loss ) ;
	DMDAVecGetArray( da,  JouleHeating, &JHeating ) ;

	//
	DMDAVecGetArray( da, Ee, &E_e ) ;
	DMDAVecGetArray( da, Ew, &E_w ) ;
	DMDAVecGetArray( da, Ex, &E_x ) ;
	//
	DMDAVecGetArray( da, ele.Mu, &Mu ) ;
	DMDAVecGetArray( da, ele.D ,  &D ) ;


	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		row.i = i ;
		for ( PetscInt k=0 ; k < 3 ; k++ ) v[k] = 0.0 ;
		s[i] = 0.0 ;


		if ( i==0 ) {

			col[0].i = i  ;
			col[1].i = i+1;

			/*--- Unsteady ---*/
			v [0] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[0] += -C53*D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[1] += -C53*D_face/(d1+d2)*( f1(X) ) ;

			/*--- w-face ---*/
			v[0] += -C53*D[i]/d1 ;

			MatSetValuesStencil( electron_energy.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		}else if( i == nCell-1 ) {

			col[0].i = i-1;
			col[1].i = i  ;

			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -C53*D_face/(d1+d2)*( f2( X) ) ;
			//i
			v[1] += -C53*D_face/(d1+d2)*(-f1(-X) ) ;

			/*--- e-face ---*/
			v[1] += C53*D[i]/d1 ;

			MatSetValuesStencil( electron_energy.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		} else {

			col[0].i = i-1 ;
			col[1].i = i   ;
			col[2].i = i+1 ;
			/*--- Unsteady ---*/
			v [1] += 1.0/dt*dx[i] ; 
			s[ i ]+= solution_old[i]/dt*dx[i] ;

			/*--- e-face ---*/
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[i+1] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[ i ] + d1*d1/( d1*d1 + d2*d2 )* Mu[i+1] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[ i ] + d1*d1/( d1*d1 + d2*d2 )*  D[i+1] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[1] += -C53*D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			v[2] += -C53*D_face/(d1+d2)*( f1(X) ) ;


			/*--- w-face ---*/
			d1 = 0.5*dx[i-1] ; 
			d2 = 0.5*dx[ i ] ;
			Mu_face = d2*d2/( d1*d1 + d2*d2 )* Mu[i-1] + d1*d1/( d1*d1 + d2*d2 )* Mu[ i ] ; 
			 D_face = d2*d2/( d1*d1 + d2*d2 )*  D[i-1] + d1*d1/( d1*d1 + d2*d2 )*  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			v[0] += -C53*D_face/(d1+d2)*( f2(X) ) ;
			//i
			v[1] += -C53*D_face/(d1+d2)*(-f1(X) ) ;
			
			MatSetValuesStencil( electron_energy.A, 1, &row, 3, col, v, INSERT_VALUES ) ;

		}
		/* Chemical */
		JHeating[ i ] = -Qe*E_x[i]*Flux[i]*dx[i] ;
		s[ i ]+= JHeating[ i ] - inelastic_loss[i]*dx[i]  ;
	}

	DMDAVecRestoreArray( da, electron_energy.B, &s ) ;
	DMDAVecRestoreArray( da, electron_energy.x, &solution_old ) ;
	DMDAVecRestoreArray( da, ele.T, &T ) ;

	//
	DMDAVecRestoreArray( da, ele.U1, &Flux ) ;
	DMDAVecRestoreArray( da, InelasticLoss, &inelastic_loss ) ;
	DMDAVecRestoreArray( da,  JouleHeating, &JHeating ) ;
	//
	DMDAVecRestoreArray( da, Ee, &E_e ) ;
	DMDAVecRestoreArray( da, Ew, &E_w ) ;
	DMDAVecRestoreArray( da, Ex, &E_x ) ;
	//
	DMDAVecRestoreArray( da, ele.Mu, &Mu ) ;
	DMDAVecRestoreArray( da, ele.D ,  &D ) ;

	MatAssemblyBegin(electron_energy.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(electron_energy.A, MAT_FINAL_ASSEMBLY);

	KSPSolve( electron_energy.ksp, electron_energy.B, electron_energy.x );
	DMGlobalToLocal( da, electron_energy.x, INSERT_VALUES, ele.U2 ) ;
}
void Compute_Te()
{
	PetscScalar *energy, *Ne, *T ;

	DMDAVecGetArray( da, ele.T, &T ) ;
	DMDAVecGetArray( da, ele.U0, &Ne ) ;
	DMDAVecGetArray( da, ele.U2, &energy ) ;


	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		T[ i ] = energy[i]/Ne[i]*2.0/3.0/Qe ;
	}

	DMDAVecRestoreArray( da, ele.T, &T ) ;
	DMDAVecRestoreArray( da, ele.U0, &Ne ) ;
	DMDAVecRestoreArray( da, ele.U2, &energy ) ;
}
void output(string filename)
{
	PetscScalar *value ;
	PetscInt count=0;
	FILE *file_pointer ;
	Vec vout ;
	VecScatter ctx ;
	/* create the vector on processor_0 and ready for scatter data on it. */
  VecScatterCreateToZero( potential, &ctx, &vout ) ;
	if ( mpi_rank==0 ) {

/*--- Create a new file and write the title and xc points. ---*/
  	file_pointer = fopen( filename.c_str(),"w" );
  	fprintf( file_pointer,"VARIABLES=\"X [m]\", \"potential\", \"Ex [V/m]\", \"n<sub>e</sub>\", \"n<sub>i</sub>\", \"Te [eV]\", \"electron flux\", \"Rdot\", \"Heating\", \"inelastic loss\" \n"  ) ;
  //cout<<"A"<<endl; PetscEnd();
		fprintf( file_pointer,"ZONE I=%d, DATAPACKING=BLOCK\n", nCell) ;
		//fprintf( file_pointer,"T =\"ZONE\"\n") ;
/*--- cell center location ---*/
		for( PetscInt i = 0 ; i < nCell ; i++ ) {
	 		fprintf( file_pointer,"%15.6e \t", xc[i]*L_ref ) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
	}

/*--- Potential ---*/
  VecScatterBegin( ctx, potential, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, potential, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*Phi_ref ) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- electric field ---*/
  VecScatterBegin( ctx, Ex, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, Ex, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*E_ref ) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;


/*--- electron nunber density. ---*/
  VecScatterBegin( ctx, ele.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, ele.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*n_ref) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- ion nunber density. ---*/
  VecScatterBegin( ctx, ion.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, ion.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*n_ref) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- electron flux ---*/
  VecScatterBegin( ctx, ele.T, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, ele.T, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*T_ref) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- electron flux ---*/
  VecScatterBegin( ctx, ele.U1, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, ele.U1, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*n_ref*E_ref*Mu_ref) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- source term ---*/
  VecScatterBegin( ctx, Rdot, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, Rdot, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]*n_ref*n_ref*k_ref) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- Heating term ---*/
  VecScatterBegin( ctx, JouleHeating, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, JouleHeating, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			//not complete yet.
			fprintf( file_pointer,"%15.6e \t", value[i]/Qe) ; //[w]
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

/*--- Heating term ---*/
  VecScatterBegin( ctx, InelasticLoss, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, InelasticLoss, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			//not complete yet.
			fprintf( file_pointer,"%15.6e \t", value[i]/Qe) ; //[w]
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;


	if ( mpi_rank==0 ){
  	fclose (file_pointer);
	}
  VecScatterDestroy(&ctx);
  VecDestroy(&vout);
}



