#include "mpi.h"
#include "jacobi.cpp"

const int X_LOCAL_START_TAG = 102;
const int X_LOCAL_END_TAG = 103;

const int LOWER_PROC_RANK_TAG = 100;
const int UPPER_PROC_RANK_TAG = 101;

int main(int argc, char **argv)
{
    // Количество процессов
    int processCount;

    // Ранг текущего процесса
    int rank;

    // Ранг младшего процесса
    int myLowerProcRank;

    // Ранг старшего процесса
    int myUpperProcRank;

    // Глобальный индекс по x, с которого начинается область текущего процесса (включительно)
    int xMyLocalStartIndx;
    // Глобальный индекс по x, на котором заканчивается область текущего процесса (включительно)
    int xMyLocalEndIndx;

    // Статус возврата
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    // Если процессов больше, чем внутренних слоёв
    if (processCount > (jacobian.Nx - 2))
    {
        if (rank == MAIN_PROC_RANK)
        {
            cout << "Too many processes!" << endl;
        }

        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    if (rank == MAIN_PROC_RANK)
    {
        cout << "Main process initializing data..." << endl;

        int integerPart = (jacobian.Nx - 2) / processCount;
        int remainder = (jacobian.Nx - 2) % processCount;

        // Размер партии (сразу вычисляем для главного процесса)
        int batchSize = integerPart + ((rank < remainder) ? 1 : 0);

        myLowerProcRank = -1;
        myUpperProcRank = processCount == 1 ? -1 : 1;

        xMyLocalStartIndx = 1;
        xMyLocalEndIndx = xMyLocalStartIndx + batchSize - 1;

        // Младший процесс, для процесса-получателя
        int lowerProcRank;
        // Старший процесс, для процесса-получателя
        int upperProcRank;

        // Начало области для процесса-получателя
        int xLocalStartIndx = 1;
        // Конец области для процесса-получателя
        int xLocalEndIndx;

        // Рассылаем всем процессам необходимую информацию
        for (int destRank = 1; destRank < processCount; destRank++)
        {
            lowerProcRank = destRank - 1;
            MPI_Send((void *)&lowerProcRank, 1, MPI_INT, destRank, LOWER_PROC_RANK_TAG, MPI_COMM_WORLD);

            upperProcRank = destRank == processCount - 1 ? -1 : destRank + 1;
            MPI_Send((void *)&upperProcRank, 1, MPI_INT, destRank, UPPER_PROC_RANK_TAG, MPI_COMM_WORLD);

            xLocalStartIndx += batchSize;
            batchSize = integerPart + ((destRank < remainder) ? 1 : 0);

            MPI_Send((void *)&xLocalStartIndx, 1, MPI_INT, destRank, X_LOCAL_START_TAG, MPI_COMM_WORLD);

            xLocalEndIndx = xLocalStartIndx + batchSize - 1;
            MPI_Send((void *)&xLocalEndIndx, 1, MPI_INT, destRank, X_LOCAL_END_TAG, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Остальные процессы получают от главного необходимую информацию
        MPI_Recv((void *)&myLowerProcRank, 1, MPI_INT, MAIN_PROC_RANK, LOWER_PROC_RANK_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv((void *)&myUpperProcRank, 1, MPI_INT, MAIN_PROC_RANK, UPPER_PROC_RANK_TAG, MPI_COMM_WORLD, &status);

        MPI_Recv((void *)&xMyLocalStartIndx, 1, MPI_INT, MAIN_PROC_RANK, X_LOCAL_START_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv((void *)&xMyLocalEndIndx, 1, MPI_INT, MAIN_PROC_RANK, X_LOCAL_END_TAG, MPI_COMM_WORLD, &status);
    }

    for (int i = 0; i < processCount; i++)
    {
        if (i == rank)
        {
            cout << "Proc rank: " << i << endl;
            cout << "Lower proc rank: " << myLowerProcRank << endl;
            cout << "Upper proc rank: " << myUpperProcRank << endl;
            cout << "x start: " << xMyLocalStartIndx << endl;
            cout << "x end:   " << xMyLocalEndIndx << endl;
            cout << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == MAIN_PROC_RANK)
    {
        cout << "Main process finish sends initial data." << endl;
    }

    // Длина области текущего процесса
    int xMyLocalLength = xMyLocalEndIndx - xMyLocalStartIndx + 1;
    // Добавляем два слоя, которые вычисляют соседние процессы, чтобы производить сво вычисления на крайних слоях
    // (либо это константные значения внешних слоёв, если таковых процессов нет)
    xMyLocalLength += 2;

    // Инициализируем решётку
    initGrid(jacobian.grid, xMyLocalLength, math.toReal(jacobian.xStart, jacobian.hx, xMyLocalStartIndx - 1));

    // Если процесс содержит первый внешний слой, где x - const (x = xStart)
    if (rank == MAIN_PROC_RANK)
    {
        // Записываем краевые значения
        // При i = xStart
        for (int j = 1; j < jacobian.Ny - 1; j++)
        {
            for (int k = 1; k < jacobian.Nz - 1; k++)
            {
                jacobian.grid[0][j][k] = math.phi(jacobian.xStart, math.toReal(jacobian.yStart, jacobian.hy, j), math.toReal(jacobian.zStart, jacobian.hz, k));
            }
        }
    }

    // Если процесс содержит второй внешний слой, где x - const (x = xEnd)
    if (rank == processCount - 1)
    {
        // Записываем краевые значения
        // При i = xEnd
        for (int j = 1; j < jacobian.Ny - 1; j++)
        {
            for (int k = 1; k < jacobian.Nz - 1; k++)
            {
                jacobian.grid[xMyLocalLength - 1][j][k] = math.phi(jacobian.xEnd, math.toReal(jacobian.yStart, jacobian.hy, j), math.toReal(jacobian.zStart, jacobian.hz, k));
            }
        }
    }

    for (int i = 0; i < processCount; i++)
    {
        if (rank == i)
        {
            cout << "Process " << rank << " finish grid initialization." << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == MAIN_PROC_RANK)
    {
        cout << "Calculation start" << endl;
    }

    double startTime = MPI_Wtime();

    // Производим вычисления
    jacobi(jacobian.grid, xMyLocalLength, xMyLocalStartIndx, myLowerProcRank, myUpperProcRank);

    double endTime = MPI_Wtime();

    double elapsedTime = endTime - startTime;

    // Считаем точность
    double precsision = getPrecisionMPI(jacobian.grid, xMyLocalLength, math.toReal(jacobian.xStart, jacobian.hx, xMyLocalStartIndx - 1), MAIN_PROC_RANK);
    if (rank == MAIN_PROC_RANK)
    {
        cout << endl
             << "Precsicion: " << precsision << endl;
        cout << endl
             << "Calculation time: " << elapsedTime << " s." << endl;
    }

    MPI_Finalize();

    deleteGrid(jacobian.grid, xMyLocalLength);
}