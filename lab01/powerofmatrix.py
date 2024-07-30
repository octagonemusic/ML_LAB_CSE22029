def matrix_mult(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to the number of rows in B")
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    
    return result

def matrix_power(A, m):
    if not all(len(row) == len(A) for row in A):
        raise ValueError("Matrix A should be square")
    
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m should be a positive integer")

    n = len(A)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for _ in range(m):
        result = matrix_mult(result, A)
    
    return result

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    m = 3

    result = matrix_power(A, m)

    print("Matrix A:")
    for row in A:
        print(row)
    print(f"Matrix A^{m}:")
    for row in result:
        print(row)
