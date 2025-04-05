#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd A1;  
	A1.resize(2, 2);
	VectorXd b1;
	b1.resize(2);
	A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	
	MatrixXd A2;  
	A2.resize(2, 2);
	VectorXd b2;
	b2.resize(2);
	A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	MatrixXd A3;  
	A3.resize(2, 2);
	VectorXd b3;
	b3.resize(2);
	A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	VectorXd x_soluzione;
	x_soluzione.resize(2);
    x_soluzione << -1.0, -1.0;
	
	
	//prima coppia
	FullPivLU<MatrixXd> LU1(A1);
	VectorXd x1_LU = LU1.solve(b1);
	double errore_rel_LU1 = (x1_LU - x_soluzione).norm() /x_soluzione.norm();
	cout << "Errore relativo coppia1(LU): " << errore_rel_LU1 << endl;
	
	HouseholderQR<MatrixXd> QR1(A1);
    VectorXd x1_QR = QR1.solve(b1);
	double errore_rel_QR1 = (x1_QR - x_soluzione).norm() / x_soluzione.norm();
	cout << "Errore relativo coppia1(QR): " << errore_rel_QR1 << endl;
	
	
	
	//seconda coppia
	FullPivLU<MatrixXd> LU2(A2);
	VectorXd x2_LU = LU2.solve(b2);
	double errore_rel_LU2 = (x2_LU - x_soluzione).norm() /x_soluzione.norm();
	cout << "Errore relativo coppia2(LU): " << errore_rel_LU2 << endl;
	
	HouseholderQR<MatrixXd> QR2(A2);
    VectorXd x2_QR = QR2.solve(b2);
	double errore_rel_QR2 = (x2_QR - x_soluzione).norm() / x_soluzione.norm();
	cout << "Errore relativo coppia2(QR): " << errore_rel_QR2 << endl;
	
	
	//terza coppia
	FullPivLU<MatrixXd> LU3(A3);
	VectorXd x3_LU = LU3.solve(b3);
	double errore_rel_LU3 = (x3_LU - x_soluzione).norm() /x_soluzione.norm();
	cout << "Errore relativo coppia3(LU): " << errore_rel_LU3 << endl;
	
	HouseholderQR<MatrixXd> QR3(A3);
    VectorXd x3_QR = QR3.solve(b3);
	double errore_rel_QR3 = (x3_QR - x_soluzione).norm() / x_soluzione.norm();
	cout << "Errore relativo coppia3(QR): " << errore_rel_QR3 << endl;
	
	
	

	
    return 0;
}
