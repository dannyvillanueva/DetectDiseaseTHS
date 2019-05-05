import numpy as np

def matrix_cosine_similary(M, N):
    if not (M.shape[1] == N.shape[1]):
        raise ValueError("Arguments cannot be multiplied.")
    # first compute M x N with
    mult_M_N = np.dot(M, N.T)

    #compute norm of M
    sqr_M = M**2
    norm_M = sqr_M.sum(1, keepdims=True) ** 0.5
    #compute norm of N
    sqr_N = N**2
    norm_N = sqr_N.sum(1, keepdims=True) ** 0.5

    # Compute product of norm matrices
    norm_mult = np.dot(norm_M, norm_N.T)

    #now compute reciprocals
    r_norm_mult = norm_mult ** -1

    # finally compute elemetwise product
    # each entry will have the consine similary
    return mult_M_N * r_norm_mult

def distance_similarity_matrix(S):
    I  = np.ones(S.shape)
    return np.average(I - S)

def TriUL_DM(S):
    triu = np.triu(S)
    tril = np.tril(S)
    return np.abs(np.sum(triu-tril))

def TriUL_sim(A,B):
    S = matrix_cosine_similary(A, B)
    return  TriUL_DM(S)

def Frobenius_Norm(M):
    entry_sqr_sum = M**2
    return np.sqrt(np.sum(entry_sqr_sum))

def L1_Norm(M):
    entry_sum = np.sum(M)
    return entry_sum

def Frobenius_Distance(A, B):
    M = A - B
    return Frobenius_Norm(M)

def L1_Distance(A, B):
    M = A- B
    return L1_Norm(M)

def build_matrix(row_list):
    M = np.array(row_list[0])
    for r in range(1, len(row_list)):
        M = np.vstack((M, np.array(row_list[r])))
    return M
