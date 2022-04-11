#include "capd/capdlib.h"
#include "capd/mpcapdlib.h"
using namespace std;
using namespace capd;
using namespace matrixAlgorithms;
using capd::autodiff::Node;

// global constant
constexpr double massSunJupiter = 0.00095388114032796904;

/**
  This routine implements vector field of the CR3BP.
  The routine is written in the format required by the constructor
  of the class DMap and IMap from the CAPD library.
*/

void cr3bpVectorField(Node t, Node in[], int dimIn, Node out[], int dimOut, Node params[], int noParams)
{
	  Node xMu = in[0] + params[0]; // params[0] - relative mass of the first body
	  Node xMj = in[0] + params[1]; // params[1] = params[0] - 1
	  Node xMuSquare = xMu^2;       // square
	  Node xMjSquare = xMj^2;
	  Node ySquare = in[1]^2;
	  Node zSquare = in[2]^2;
	  Node yzSquare = ySquare + zSquare;
	  
	  Node factor1 = params[1]*((xMuSquare+yzSquare)^-1.5);
	  Node factor2 = params[0]*((xMjSquare+yzSquare)^-1.5);
	  Node factor = factor1 - factor2;

	  out[0] = in[3];
	  out[1] = in[4];
	  out[2] = in[5];

	  out[3] = in[0] + xMu*factor1 - xMj*factor2 + 2*in[4];
	  out[4] = in[1]*(1+factor) - 2*in[3];
	  out[5] = in[2]*factor;
}


DVector step(DVector u, DPoincareMap& pm){
	  DMatrix D(6,6); 
	  DVector v = pm(u,D);
	  DMatrix DP = pm.computeDP(v,D);
	  
	  DVector G({v[3], v[5]});
	  DMatrix DG({{DP[3][0], DP[3][4]},{DP[5][0],DP[5][4]}});

	  DVector tmp = matrixAlgorithms::gauss(DG, G);
	  // cout << "tmp = " << tmp << endl;
	  // (X,DY) = (x,dy) - DG(x,dy){-1}*G(x,dy)
	  DVector next_u = u;
	  next_u[0] -= tmp[0];
	  next_u[4] -= tmp[1];
	  // cout <<"P(P(u))-u = " << pm(pm(next_u, D), D)-u << endl; 
	  return next_u;
}

#define __HH_USE_MP__
#ifdef __HH_USE_MP__
     typedef MpIVector HHVector;
     typedef MpIMatrix HHMatrix;
     typedef MpC0Rect2Set HHC0Set;
     typedef MpC1Rect2Set HHC1Set;
     typedef MpIMap HHMap;
     typedef MpIOdeSolver HHSolver;
     typedef MpIPoincareMap HHPoincareMap;
     typedef MpICoordinateSection HHSection;
     typedef MpInterval HHInterval;
#else
     typedef IVector HHVector;
     typedef IMatrix HHMatrix;
     typedef C0Rect2Set HHC0Set;
     typedef C1Rect2Set HHC1Set;
     typedef IMap HHMap;
     typedef IOdeSolver HHSolver;
     typedef IPoincareMap HHPoincareMap;
     typedef ICoordinateSection HHSection;
     typedef interval HHInterval;
#endif

bool isSubset(HHVector x0, HHVector& delta_x, HHPoincareMap& pm){
	  HHC0Set s0(x0);
	  HHC1Set s(x0+delta_x);
	  
	  HHVector v0 = pm(s0);
	  
	  HHMatrix D(6,6); 
	  HHVector v = pm(s, D);
	  HHMatrix DP = pm.computeDP(v,D);
	  
	  HHVector G({v0[3], v0[5]});
	  HHMatrix DG({{DP[3][0], DP[3][4]},{DP[5][0],DP[5][4]}});
	  
       
	  HHVector tmp = -matrixAlgorithms::gauss(DG, G);
	  cout << "tmp = " << tmp << endl;
	  cout << "dx  = " << HHVector({delta_x[0],delta_x[4]}) << endl;
	  // (X,DY) = (x,dy) - DG(x,dy){-1}*G(x,dy)	  
	  return tmp[0].subset(delta_x[0]) and tmp[1].subset(delta_x[4]);
}


int main(){
	  cout.precision(17);
	  DMap crvf(cr3bpVectorField,6,6,2);
	  crvf.setParameter(0,massSunJupiter);
	  crvf.setParameter(1,massSunJupiter-1);
	  
	  DVector u({0.300306912615489806842, 0., 0., 0., 2.07276046319409481687, 0.}); // punkt poczatkowy
	  double delta = 1./(1<<12);
	  
	  DOdeSolver solver(crvf,20);
	  DCoordinateSection section(6,1);
	  DPoincareMap pm(solver,section);
	  
	  std::ofstream myfile;
	  myfile.open ("results.csv");
	  myfile << "x,z,dy,\n" ;
	  
	  int n=1+0.07/delta;
	  vector<DVector> res(2*n+1);
	  res[0] = u;
	  myfile << u[0] << "," << u[2] << "," << u[4] << ",\n";
	  DMaxNorm norm;
	  for(int i=1; i<n+1; i++){
		  u[2] += delta;
            u[0] -= delta;
		  for(int i=0; i<25; i++){
			 DVector v = u;
                u = step(u, pm);
                if(norm(u-v)<1e-13) break;
		  }
		  res[i] = u;
		  cout << "next_u = " << u << endl;
		  myfile << u[0] << "," << u[2] << "," << u[4] << ",\n";
	  }
	  for(int i=1; i<n+1; i++){   
		  res[n+i] = res[i];
            res[n+i][2] = -res[n+i][2];
            res[n+i][5] = -res[n+i][5];
		  myfile << res[n+i][0] << "," << res[n+i][2] << "," << res[n+i][4] << ",\n";
	  }
      myfile.close();
      
	 MpFloat::setDefaultPrecision(100);

	  HHMap _crvf(cr3bpVectorField,6,6,2);
	  _crvf.setParameter(0,HHInterval(massSunJupiter));
	  _crvf.setParameter(1,HHInterval(massSunJupiter)-1);
	  HHSolver _solver(_crvf,40);
       _solver.setAbsoluteTolerance(1e-18);
       _solver.setRelativeTolerance(1e-18);
	  HHSection _section(6,1);
	  HHPoincareMap _pm(_solver,_section);
	  
	  HHVector delta_x({5e-11*HHInterval(-1,1),HHInterval(0.),HHInterval(0.),HHInterval(0.), 1e-10*HHInterval(-1,1), HHInterval(0.)}); 
		  
	  for(int i=1;i<=n; i++){
		  cout << i << ", z = " << res[i][2] << endl;
            DVector u = res[i];
		  HHVector p({HHInterval(u[0]),HHInterval(u[1]),HHInterval(u[2]),HHInterval(u[3]),HHInterval(u[4]),HHInterval(u[5])});
            if(isSubset(p, delta_x, _pm)){
               cout << "OK" << endl;
            }
            else{
               cout << "ERROR" << endl;
               return 0;
            }
	  }
}
