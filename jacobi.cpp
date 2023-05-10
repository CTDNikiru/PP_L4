#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include "mpi.h" 
#include "math_function.cpp"


//MPI_Isend((void*)currentDestPtr[xLocalLength - 2], messageLength, MPI_DOUBLE, childProcess, 104, MPI_COMM_WORLD, &arr_recv[2]);

//MPI_Irecv((void*)currentDestPtr[xLocalLength - 2], messageLength, MPI_DOUBLE, parentProcess, 105, MPI_COMM_WORLD, &arr_recv[1]);


using namespace std;

const int MAIN_PROC_RANK = 0;

const int LOWER_BOUND_TAG = 104;
const int UPPER_BOUND_TAG = 105;

struct Jacobian
{
    const double a = 10.0e5;

    const double epsilon = 1.0e-8;
    // Начальное приближение
    const double phi0 = 0.0;

    // Границы области
    const double xStart = -1.0;
    const double xEnd = 1.0;

    const double yStart = -1.0;
    const double yEnd = 1.0;

    const double zStart = -1.0;
    const double zEnd = 1.0;

    // Размеры области
    const double Dx = xEnd - xStart;
    const double Dy = yEnd - yStart;
    const double Dz = zEnd - zStart;

    // Количество узлов сетки
    const int Nx = 15;
    const int Ny = 15;
    const int Nz = 15;

    // Размеры шага на сетке
    const double hx = Dx / (Nx - 1);
    const double hy = Dy / (Ny - 1);
    const double hz = Dz / (Nz - 1);

    double ***grid;
} jacobian;

MathFunc math = MathFunc(jacobian.a);


void initGrid(double***& grid, int xLength, double xLocalStart)
{
	// Создаём массив и заполняем начальными значениями
	grid = new double** [xLength];
	for (int i = 0; i < xLength; i++) {
		grid[i] = new double* [jacobian.Ny];

		for (int j = 0; j < jacobian.Ny; j++) {
			grid[i][j] = new double[jacobian.Nz];

			for (int k = 1; k < jacobian.Nz - 1; k++) {
				grid[i][j][k] = jacobian.phi0;
			}
	}
	}

	// Записываем краевые значения
	double xCurr;
	for (int i = 0; i < xLength; i++) {
		xCurr = math.toReal(xLocalStart, jacobian.hx, i);

		for (int k = 0; k < jacobian.Nz; k++) {
			// При j = 0
			grid[i][0][k] = math.phi(xCurr, jacobian.yStart, math.toReal(jacobian.zStart, jacobian.hz, k));
		}

		for (int k = 0; k < jacobian.Nz; k++) {
			// При j = Ny - 1
			grid[i][jacobian.Ny - 1][k] = math.phi(xCurr, jacobian.yEnd, math.toReal(jacobian.zStart, jacobian.hz, k));
		}

		double yCurr;
		for (int j = 1; j < jacobian.Ny - 1; j++) {
			yCurr = math.toReal(jacobian.yStart, jacobian.hy, j);

			// При k = 0
			grid[i][j][0] = math.phi(xCurr, yCurr, jacobian.zStart);

			// При k = Nz - 1
			grid[i][j][jacobian.Nz - 1] = math.phi(xCurr, yCurr, jacobian.zEnd);
		}
	}
}

void deleteGrid(double*** grid, int xLocalLength)
{
	for (int i = 0; i < xLocalLength; i++) {
		for (int j = 0; j < jacobian.Ny; j++) {
			delete[] grid[i][j];
		}
		delete[] grid[i];
	}
	delete[] grid;
}

// Считаем точность, как максимальное значение модуля отклонения
// от истинного значения функции
double getPrecisionMPI(double*** grid, int xLocalLength, double xLocalStart, int rootProcRank)
{
	// Значение ошибки на некотором узле
	double currErr;
	// Максимальное значение ошибки данного процесса
	double maxLocalErr = 0.0;

	for (int i = 1; i < xLocalLength - 1; i++) {
		for (int j = 1; j < jacobian.Ny - 1; j++) {
			for (int k = 1; k < jacobian.Nz - 1; k++) {
				currErr = abs(grid[i][j][k] -
					math.phi(math.toReal(xLocalStart, jacobian.hx, i), math.toReal(jacobian.yStart, jacobian.hy, j), math.toReal(jacobian.zStart, jacobian.hz, k)));
				if (currErr > maxLocalErr) {
					maxLocalErr = currErr;
				}
			}
		}
	}

	// Максимальное значение ошибки по всем процессам
	double absoluteMax = -1;
	MPI_Reduce((void*)&maxLocalErr, (void*)&absoluteMax, 1, MPI_DOUBLE, MPI_MAX, rootProcRank, MPI_COMM_WORLD);

	return absoluteMax;
}

void jacobi(double***& grid1, int xLocalLength, int xLocalStartIndx, int lowerProcRank, int upperProcRank)
{

	MPI_Request arr_recv[4];// send_request_child, send_request_parent, recv_request_child, recv_request_parent;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Status status;

	// Значение сходимости для некоторого узла сетки
	double currConverg;
	// Максимальное значение сходимости по всем узлам на некоторой итерации
	double maxLocalConverg;

	// Флаг, показывающий, является ли эпсилон меньше любого значения сходимости для данного процесса
	char isEpsilonLower;

	const double hx2 = pow(jacobian.hx, 2);
	const double hy2 = pow(jacobian.hy, 2);
	const double hz2 = pow(jacobian.hz, 2);

	// Константа, вынесенная за скобки
	double c = 1 / ((2 / hx2) + (2 / hy2) + (2 / hz2) + jacobian.a);

	// Второй массив для того, чтобы использовать значения предыдущей итерации
	double*** grid2;
	// Просто копируем входной массив
	grid2 = new double** [xLocalLength];
	for (int i = 0; i < xLocalLength; i++) {
		grid2[i] = new double* [jacobian.Ny];

		for (int j = 0; j < jacobian.Ny; j++) {
			grid2[i][j] = new double[jacobian.Nz];

			for (int k = 0; k < jacobian.Nz; k++) {
				grid2[i][j][k] = grid1[i][j][k];
			}
		}
}

	// Указатель на массив, из которого на некоторой итерации
	// берутся значения для расчёта
	double*** currentSourcePtr = grid1;
	// Указатель на массив, в который на некоторой итерации
	// Записываются новые значения
	double*** currentDestPtr = grid2;
	// Вспомогательный указатель для перемены указателей на массивы
	double*** tmpPtr;

	int messageLength = (jacobian.Ny - 2) * (jacobian.Nz - 2);
	
	double* messageBufSend_1 = new double[messageLength];
	double* messageBufSend_2 = new double[messageLength];

	double* messageBufReqv_1 = new double[messageLength];
	double* messageBufReqv_2 = new double[messageLength];

	// Флаг, который показывает, что нужно продолжать вычисления 
	char loopFlag = 1;

	MPI_Request requests[4];

	while (loopFlag) {
		maxLocalConverg = 0.0;

		// Сначала вычисляем граничные значения
		// При i = 1
		for (int j = 1; j < jacobian.Ny - 1; j++) {
			for (int k = 1; k < jacobian.Nz - 1; k++) {
				// Первая дробь в скобках
				currentDestPtr[1][j][k] = (currentSourcePtr[2][j][k] + currentSourcePtr[0][j][k]) / hx2;

				// Вторая дробь в скобках
				currentDestPtr[1][j][k] += (currentSourcePtr[1][j + 1][k] + currentSourcePtr[1][j - 1][k]) / hy2;

				// Третья дробь в скобках
				currentDestPtr[1][j][k] += (currentSourcePtr[1][j][k + 1] + currentSourcePtr[1][j][k - 1]) / hz2;

				// Остальная часть вычисления нового значения для данного узла
				currentDestPtr[1][j][k] -= math.rho(currentSourcePtr[1][j][k]);
				currentDestPtr[1][j][k] *= c;

				// Сходимость для данного узла
				currConverg = abs(currentDestPtr[1][j][k] - currentSourcePtr[1][j][k]);
				if (currConverg > maxLocalConverg) {
					maxLocalConverg = currConverg;
				}
			}
		}

		// Если процесс должен отправить свой крайний слой с младшим значением x (не содержит слоя с x = 0)
		if (lowerProcRank != -1) {
		for (int j = 0; j < jacobian.Ny - 2; j++) {
				for (int k = 0; k < jacobian.Nz - 2; k++) {
					messageBufSend_1[(jacobian.Ny - 2) * j + k] = currentDestPtr[1][j + 1][k + 1];
				}
			}
			// Отправляем слой младшему процессу
			//cout << rank << "start sending to child process" << endl;
			MPI_Isend((void*)messageBufSend_1, messageLength, MPI_DOUBLE, lowerProcRank, UPPER_BOUND_TAG, MPI_COMM_WORLD, &requests[0]);
			//MPI_Send((void*)messageBuf, messageLength, MPI_DOUBLE, lowerProcRank, UPPER_BOUND_TAG, MPI_COMM_WORLD);
			MPI_Irecv((void*)messageBufReqv_1, messageLength, MPI_DOUBLE, lowerProcRank, LOWER_BOUND_TAG, MPI_COMM_WORLD, &requests[2]);
		}


		// Если процесс обрабатывает более одного слоя
		if (xLocalLength != 3) {
			// Вычисляем граничные значения
			// При i = xLength - 2
			for (int j = 1; j < jacobian.Ny - 1; j++) {
				for (int k = 1; k < jacobian.Nz - 1; k++) {
					// Первая дробь в скобках
					currentDestPtr[xLocalLength - 2][j][k] = (currentSourcePtr[xLocalLength - 1][j][k] + currentSourcePtr[xLocalLength - 3][j][k]) / hx2;

					// Вторая дробь в скобках
					currentDestPtr[xLocalLength - 2][j][k] += (currentSourcePtr[xLocalLength - 2][j + 1][k] + currentSourcePtr[xLocalLength - 2][j - 1][k]) / hy2;

					// Третья дробь в скобках
					currentDestPtr[xLocalLength - 2][j][k] += (currentSourcePtr[xLocalLength - 2][j][k + 1] + currentSourcePtr[xLocalLength - 2][j][k - 1]) / hz2;

					// Остальная часть вычисления нового значения для данного узла
					currentDestPtr[xLocalLength - 2][j][k] -= math.rho(currentSourcePtr[xLocalLength - 2][j][k]);
					currentDestPtr[xLocalLength - 2][j][k] *= c;

					// Сходимость для данного узла
					currConverg = abs(currentDestPtr[xLocalLength - 2][j][k] - currentSourcePtr[xLocalLength - 2][j][k]);
					if (currConverg > maxLocalConverg) {
						maxLocalConverg = currConverg;
					}
				}
			}
		}

		// Если процесс должен отправить свой крайний слой со старшим значением x (не содержит слоя с x = Nx - 1)
		if (upperProcRank != -1) {
			for (int j = 0; j < jacobian.Ny - 2; j++) {
				for (int k = 0; k < jacobian.Nz - 2; k++) {
					messageBufSend_2[(jacobian.Ny - 2) * j + k] = currentDestPtr[xLocalLength - 2][j + 1][k + 1];
				}
			}
			// Отправляем слой старшему процессу
			//cout << rank << "start sending to parent process" << endl;
			//MPI_Send((void*)messageBuf, messageLength, MPI_DOUBLE, upperProcRank, LOWER_BOUND_TAG, MPI_COMM_WORLD);
			MPI_Isend((void*)messageBufSend_2, messageLength, MPI_DOUBLE, upperProcRank, LOWER_BOUND_TAG, MPI_COMM_WORLD, &requests[1]);
			MPI_Irecv((void*)messageBufReqv_2, messageLength, MPI_DOUBLE, upperProcRank, UPPER_BOUND_TAG, MPI_COMM_WORLD, &requests[3]);
		}
		

		for (int i = 2; i < xLocalLength - 2; i++) {
			for (int j = 1; j < jacobian.Ny - 1; j++) {
				for (int k = 1; k < jacobian.Nz - 1; k++) {

					// Первая дробь в скобках
					currentDestPtr[i][j][k] = (currentSourcePtr[i + 1][j][k] + currentSourcePtr[i - 1][j][k]) / hx2;

					// Вторая дробь в скобках
					currentDestPtr[i][j][k] += (currentSourcePtr[i][j + 1][k] + currentSourcePtr[i][j - 1][k]) / hy2;

					// Третья дробь в скобках
					currentDestPtr[i][j][k] += (currentSourcePtr[i][j][k + 1] + currentSourcePtr[i][j][k - 1]) / hz2;

					// Остальная часть вычисления нового значения для данного узла
					currentDestPtr[i][j][k] -= math.rho(currentSourcePtr[i][j][k]);
					currentDestPtr[i][j][k] *= c;

					// Сходимость для данного узла
					currConverg = abs(currentDestPtr[i][j][k] - currentSourcePtr[i][j][k]);
					if (currConverg > maxLocalConverg) {
						maxLocalConverg = currConverg;
					}

				}
			}
		}

		//// Если процесс должен получить слой соседнего процесса для просчёта своего слоя с младшим значением x (не содержит слоя с x = 0)
		//if (lowerProcRank != -1) {
		//	cout << rank << "start recving from child process" << endl;
		//	MPI_Irecv((void*)messageBufReqv_1, messageLength, MPI_DOUBLE, lowerProcRank, LOWER_BOUND_TAG, MPI_COMM_WORLD, &requests[2]);
		//	//MPI_Recv((void*)messageBuf, messageLength, MPI_DOUBLE, lowerProcRank, LOWER_BOUND_TAG, MPI_COMM_WORLD, &status);
		//	/*for (int j = 0; j < Ny - 2; j++) {
		//		for (int k = 0; k < Nz - 2; k++) {
		//			currentDestPtr[0][j + 1][k + 1] = messageBufReqv_1[(Ny - 2) * j + k];
		//		}
	//	}*/
		//}

		//// Если процесс должен получить слой соседнего процесса для просчёта своего слоя со старшим значением x (не содержит слоя с x = Nx - 1)
		//if (upperProcRank != -1) {
		//	cout << rank << "start recving from parent process" << endl;
		//	MPI_Irecv((void*)messageBufReqv_1, messageLength, MPI_DOUBLE, upperProcRank, UPPER_BOUND_TAG, MPI_COMM_WORLD, &requests[3]);
		//	//MPI_Recv((void*)messageBufReqv_2, messageLength, MPI_DOUBLE, upperProcRank, UPPER_BOUND_TAG, MPI_COMM_WORLD, &status);
		//	/*for (int j = 0; j < Ny - 2; j++) {
		//		for (int k = 0; k < Nz - 2; k++) {
		//			currentDestPtr[xLocalLength - 1][j + 1][k + 1] = messageBufReqv_2[(Ny - 2) * j + k];
		//		}
		//	}*/
		//}

		if (jacobian.epsilon < maxLocalConverg) {
			isEpsilonLower = 1;
		}
		else {
			isEpsilonLower = 0;
		}

		// Применяем логичекую операцию ИЛИ над флагом сходимости между всеми процессами и помещаем результат во флаг цикла.
		// Таким образом, цикл завершится, когда у всех процессов сходимость будет меньше, чем эпсилон
		MPI_Reduce((void*)&isEpsilonLower, (void*)&loopFlag, 1, MPI_CHAR, MPI_BOR, MAIN_PROC_RANK, MPI_COMM_WORLD);
		MPI_Bcast((void*)&loopFlag, 1, MPI_CHAR, MAIN_PROC_RANK, MPI_COMM_WORLD);

		// Меняем местами указатели на массив-источник и массив-приёмник
		tmpPtr = currentSourcePtr;
		currentSourcePtr = currentDestPtr;
		currentDestPtr = tmpPtr;

		MPI_Status status;

		if (lowerProcRank != -1)
		{
			MPI_Wait(&requests[0], &status);
		}
		
		if (upperProcRank != -1)
		{
			MPI_Wait(&requests[1], &status);
		}

		if (lowerProcRank != -1)
		{
			MPI_Wait(&requests[2], &status);
			for (int j = 0; j < jacobian.Ny - 2; j++) {
				for (int k = 0; k < jacobian.Nz - 2; k++) {
					currentDestPtr[0][j + 1][k + 1] = messageBufReqv_1[(jacobian.Ny - 2) * j + k];
				}
			}
		}
		if (upperProcRank != -1)
		{
			MPI_Wait(&requests[3], &status);
			for (int j = 0; j < jacobian.Ny - 2; j++) {
				for (int k = 0; k < jacobian.Nz - 2; k++) {
					currentDestPtr[xLocalLength - 1][j + 1][k + 1] = messageBufReqv_2[(jacobian.Ny - 2) * j + k];
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}


	// В итоге массив должен содержать значения последней итерации
	if (currentSourcePtr == grid1) {
		deleteGrid(grid2, xLocalLength);
	}
	else {
		tmpPtr = grid1;
		grid1 = currentSourcePtr;
		deleteGrid(tmpPtr, xLocalLength);
	}

	delete[] messageBufSend_1;
	delete[] messageBufSend_2;
	delete[] messageBufReqv_1;
	delete[] messageBufReqv_2;
}

