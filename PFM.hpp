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
	PetscScalar sgn_q ; //Species chearge.
	Vec U0, //nunmber density.
			U1, //flux in x-direction.
			U2; //energy flux.

	Vec  D, //Diffusion coefficient.
			Mu, //Mobility 
			rate;//rate constant.
} Species ; //Format of data output

/*--- Mesh & PETSc DMDA parameters ---*/
PetscInt nCell = 100 ; //number of cells 
PetscScalar Pressure_in_torr=1.0 ;
PetscScalar mesh_factor = 1.0 ; //the grid point will close to the wall wnen value is large.

PetscScalar *x, 	// node coordinate.
						*xc,  // cell coordinate.
						*dx ; //Cell width 
PetscScalar gap_length = 1.0 ; //gap distance

PetscInt nNode = nCell+1 ;/* number of node. (calculated from nCell)*/
PetscInt dof = 1 ;
PetscInt overlope = 1 ;

/*--- MPI & Petsc Distribution Memory ---*/
DM da ;
DMDALocalInfo da_info ;

PetscMPIInt mpi_rank, 
						mpi_size; /* number of nodes */

/* Declaration of Linear system Solver */
LinearSysSolver poisson, electron_continuity, electron_energy, ion_continuity ;

Vec potential, Mu_e, D_e, Mu_i, D_i, Ex, Ee, Ew, gVec, N_e, N_i ;
Species ele, ion ;
/**/
PetscScalar me = 9.10938356E-31 ;
PetscScalar Qe = 1.60217662E-19 ;


/*--- Declaration of functions ---*/
void CreatePetscDMDA_1D();
void CreateMesh_1D();
void InitializeLinearSystemSolver();
void InitializePetscVector();
void Poisson_eqn(PetscScalar);
void electron_continuity_eqn();
void convection_diffusion_eqn();
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
	//Note: all processor have same mesh informations.
	x  = new PetscScalar [nNode] ;
	xc = new PetscScalar [nCell] ;
	dx = new PetscScalar [nCell] ;
	
	/* hyperbolic tangent stretching scheme */
	for ( PetscInt i = 0 ; i < nNode ; i++ ) {
		x[ i ] = 0.5*gap_length*( 1.0 + tanh( mesh_factor*( (PetscScalar)i/nCell-0.5 ) ) /tanh(0.5*mesh_factor) );
		//PetscPrintf(PETSC_COMM_SELF, "x: %e\n", x[i] ) ;
		PetscPrintf(PETSC_COMM_SELF, " %15.6e \t  3 \n", x[i] ) ;
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


	//electrostatic 
	DMCreateLocalVector ( da, &potential ) ;//cell center potential.
	DMCreateLocalVector  ( da,  &Ex ) ;//cell center electric field.
	DMCreateGlobalVector ( da,  &Ee ) ;//face electric field.
	DMCreateGlobalVector ( da,  &Ew ) ;//face electric field.

	//Electron
	DMCreateLocalVector ( da, &ele.U0 ) ;//Mobility of electron at cell.
	DMCreateLocalVector ( da, &ele.Mu ) ;//Mobility of electron at cell.
	DMCreateLocalVector ( da, &ele.D  ) ;//Diffusion of electron at cell.

	//Ion
	DMCreateLocalVector ( da, &ion.U0 ) ;//Mobility of electron at cell.
	DMCreateLocalVector ( da, &ion.Mu ) ;//Mobility of electron at cell.
	DMCreateLocalVector ( da, &ion.D  ) ;//Diffusion of electron at cell.

}
void UpdateTransportCoefficients()
{
	//Davoudabadi, M., Shrimpton, J., and Mashayek, F., “On accuracy and performance of high-order finite volume methods in local 
	//          mean energy model of non-thermal plasmas,” Journal of Computational Physics, vol. 228, no. 7, pp. 2468-2479, 2009.
	VecSet( ele.Mu,   30.0/Pressure_in_torr) ;
	VecSet( ele.D ,  120.0/Pressure_in_torr) ;

	VecSet( ion.Mu,   0.14/Pressure_in_torr) ;
	VecSet( ion.D , 4.0E-3/Pressure_in_torr) ;

}
void Poisson_eqn( PetscScalar voltage )
{
	MatStencil  row, col[3] ;
	PetscInt count_col ;
	PetscScalar v[3], d1, d2, *source ;

	DMDAVecGetArray( da, poisson.B, &source ) ;

	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		source[ i ] = 0.0 ;
		row.i = i ;

		count_col = 0 ;

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
			//source = -1.0/(d1+d2)*0.0  alwaws ground.

			MatSetValuesStencil( poisson.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

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
			source[ i ] = -1.0/(d1)*voltage ;

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
	}

	DMDAVecRestoreArray( da, poisson.B, &source ) ;

	MatAssemblyBegin(poisson.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(poisson.A, MAT_FINAL_ASSEMBLY);
	KSPSolve( poisson.ksp, poisson.B, poisson.x );
	//VecView(poisson.x, PETSC_VIEWER_STDOUT_WORLD ) ;
	//DMGlobalToLocal( da, Vec g,InsertMode mode,Vec l)
}
void convection_diffusion_eqn()
{
	PetscScalar sign_q = 1.0 ;
	VecSet( ele.Mu, 10.0 ) ;
	VecSet( ele.D , 1.0 ) ;

	VecSet( Ee, 1.0 ) ;
	VecSet( Ew, 1.0 ) ;

	MatStencil  row, col[3] ;

	PetscScalar v[3], *s,  *E_e, *E_w, *D, *Mu ;

	PetscScalar d1, d2, v1, v2, D_face, Mu_face, X ;
	MatZeroEntries(electron_continuity.A);

	DMDAVecGetArray( da, electron_continuity.B, &s ) ;

	DMDAVecGetArray( da, Ee, &E_e ) ;
	DMDAVecGetArray( da, Ew, &E_w ) ;

	DMDAVecGetArray( da, ele.Mu, &Mu ) ;
	DMDAVecGetArray( da, ele.D ,  &D ) ;


	for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
		s[ i ] = 0.0 ;
		row.i = i ;

		for ( int k=0 ; k < 3 ; k++ ) v[k]=0.0;

		if ( i==0 ) {

			col[0].i = i   ;
			col[1].i = i+1 ;

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
			d1 = 0.5*dx[ i ] ; 
			d2 = 0.5*dx[ i ] ;

			Mu_face = Mu[ i ] ; 
			 D_face =  D[ i ] ; 
			X = sign_q*E_w[i]*Mu_face*(d1+d2)/D_face ;
			//i-1
			s[i] -= -D_face/(d1+d2)*( f2(X) )*1.0 ;
			//i
			v[0] += -D_face/(d1+d2)*(-f1(X) ) ;

			MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		}else if( i == nCell-1 ) {

			col[0].i = i-1 ;
			col[1].i = i   ;

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
			d1 = 0.5*dx[ i ] ; d2 = d1 ;

			Mu_face = Mu[ i ] ; 
			 D_face =  D[ i ] ; 
			X = sign_q*E_e[i]*Mu_face*(d1+d2)/D_face ;
			//i
			v[1] += -D_face/(d1+d2)*(-f2(X) ) ;
			//i+1
			s[i] -= -D_face/(d1+d2)*( f1(X) )*0.0 ;

			MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

		} else {

			col[0].i = i-1 ;
			col[1].i = i   ;
			col[2].i = i+1 ;


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
			MatSetValuesStencil( electron_continuity.A, 1, &row, 3, col, v, INSERT_VALUES ) ;

		}
	}

	DMDAVecRestoreArray( da, electron_continuity.B, &s ) ;
	DMDAVecRestoreArray( da, ele.Mu, &Mu ) ;
	DMDAVecRestoreArray( da, ele.D ,  &D ) ;

	DMDAVecRestoreArray( da, Ee, &E_e ) ;
	DMDAVecRestoreArray( da, Ew, &E_w ) ;

	MatAssemblyBegin(electron_continuity.A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(electron_continuity.A, MAT_FINAL_ASSEMBLY);

	KSPSolve( electron_continuity.ksp, electron_continuity.B, electron_continuity.x );
	//MatView(electron_continuity.A, PETSC_VIEWER_STDOUT_WORLD) ;
	//VecView(electron_continuity.x, PETSC_VIEWER_STDOUT_WORLD ) ;
	DMGlobalToLocal( da, electron_continuity.x, INSERT_VALUES, ele.U0 ) ;
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
	// MatStencil  row, col[3] ;
	// PetscInt count_col ;
	// PetscScalar v[3], dx_plus, dx_minus, *source, D_eFace, D_wFace, Mu_eFace, Mu_wFace, *E_eFace, *E_wFace, *D, *Mu ;

	// DMDAVecGetArray( da, electron_continuity.B, &source ) ;
	// DMDAVecGetArray( da, electron_continuity.B, &source ) ;
	// DMDAVecGetArray( da, electron_continuity.B, &source ) ;

	// DMDAVecGetArray( da,  D_e,  &D ) ;
	// DMDAVecGetArray( da, Mu_e, &Mu ) ;


	// for ( PetscInt i=da_info.xs; i < da_info.xs+da_info.xm ; i++ ) {
	// 	source[ i ] = 0.0 ;
	// 	row.i = i ;

	// 	count_col = 0 ;

	// 	for ( int k=0 ; k < 3 ; k++ ) v[k]=0.0;

	// 	if ( i==0 ) {

	// 		col[0].i = i   ;
	// 		col[1].i = i+1 ;

	// 		//eFace 
	// 		dx_plus  = 0.5*dx[ i ] ;
	// 		dx_minus = 0.5*dx[i+1] ;
	// 		v[0] += -1.0/(dx_plus+dx_minus) ;
	// 		v[1] +=  1.0/(dx_plus+dx_minus) ;

	// 		//wFace 
	// 		dx_minus = 0.5*dx[ i ] ;
	// 		v[0] +=  -1.0/(dx_minus) ;
	// 		//source = -1.0/(dx_plus+dx_minus)*0.0  alwaws ground.

	// 		MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;

	// 	}else if( i == nCell-1 ) {

	// 		col[0].i = i-1 ;
	// 		col[1].i = i   ;

	// 		//wFace 
	// 		dx_plus  = 0.5*dx[i-1] ;
	// 		dx_minus = 0.5*dx[ i ] ;
	// 		v[0] +=  1.0/(dx_plus+dx_minus) ;
	// 		v[1] += -1.0/(dx_plus+dx_minus) ;

	// 		//eFace 
	// 		dx_plus  = 0.5*dx[ i ] ;
	// 		v[1] +=  -1.0/(dx_plus) ;

	// 		MatSetValuesStencil( electron_continuity.A, 1, &row, 2, col, v, INSERT_VALUES ) ;
	// 		source[ i ] = -1.0/(dx_plus)*10 ;

	// 	} else {
			
	// 		Mu_eFace = 0.5*( Mu[i+1]+Mu[ i ] ) ; D_eFace = 0.5*( D[i+1]+D[ i ] ) ;
	// 		Mu_wFace = 0.5*( Mu[ i ]+Mu[i-1] ) ; D_wFace = 0.5*( D[ i ]+D[i-1] ) ;

	// 		col[0].i = i-1 ;
	// 		col[1].i = i   ;
	// 		col[2].i = i+1 ;

	// 		//wFace 
	// 		dx_plus  = 0.5*dx[i-1] ;
	// 		dx_minus = 0.5*dx[ i ] ;
	// 		v[0] +=  1.0/(dx_plus+dx_minus) ;
	// 		v[1] += -1.0/(dx_plus+dx_minus) ;


	// 		//eFace 
	// 		dx_plus  = 0.5*dx[ i ] ;
	// 		dx_minus = 0.5*dx[i+1] ;
	// 		v[1] += -1.0/(dx_plus+dx_minus) ;
	// 		v[2] +=  1.0/(dx_plus+dx_minus) ;
	// 		MatSetValuesStencil( electron_continuity.A, 1, &row, 3, col, v, INSERT_VALUES ) ;
	// 	}
	// }

	// DMDAVecRestoreArray( da, electron_continuity.B, &source ) ;
	// DMDAVecRestoreArray( da,  D_e,  &D ) ;
	// DMDAVecRestoreArray( da, Mu_e, &Mu ) ;



	// MatAssemblyBegin(electron_continuity.A, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(electron_continuity.A, MAT_FINAL_ASSEMBLY);
	// KSPSolve( electron_continuity.ksp, electron_continuity.B, electron_continuity.x );

	// VecView(electron_continuity.x, PETSC_VIEWER_STDOUT_WORLD ) ;

}
void output(string filename)
{
	PetscScalar *value ;
	PetscInt count=0;
	FILE *file_pointer ;
	Vec vout ;
	VecScatter ctx ;
  VecScatterCreateToZero( poisson.x, &ctx, &vout ) ;
	//here use c language.

	//Create a new file and write the title and xc points.
	if ( mpi_rank==0 )
	{
  	file_pointer = fopen( filename.c_str(),"w" );
  	fprintf( file_pointer,"VARIABLES=\"X [m]\", \"potential\", \"n<sub>e</sub>\" \n"  ) ;
		fprintf( file_pointer,"ZONE I=%d, DATAPACKING=BLOCK\n", nCell) ;

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
	 		fprintf( file_pointer,"%15.6e \t", xc[i]) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
	}

	//Potential
  VecScatterBegin( ctx, poisson.x, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, poisson.x, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;

	//electron nunber density.
  VecScatterBegin( ctx, ele.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecScatterEnd  ( ctx, ele.U0, vout, INSERT_VALUES, SCATTER_FORWARD ) ;
  VecGetArray( vout, &value ) ;
  if ( mpi_rank==0 ) {

		for( PetscInt i = 0 ; i < nCell ; i++ ) {
			fprintf( file_pointer,"%15.6e \t", value[i]) ;
			count++ ;
			if( count == 6 ) {
				fprintf( file_pointer,"\n" ) ;
				count=0 ;
			}
		}
		fprintf( file_pointer,"\n" ) ;
  }
  VecRestoreArray( vout, &value ) ;


	if ( mpi_rank==0 )
  fclose (file_pointer);
  VecScatterDestroy(&ctx);
  VecDestroy(&vout);
}



