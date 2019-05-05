from ths.utils.similarity import matrix_cosine_similary, distance_similarity_matrix
import numpy as np

def main():
    M1 = np.array([1, 2, 3])
    temp = np.array([1, 2, 3])
    M1 = np.vstack((M1, np.array([4, 5, 6])))
    sim = matrix_cosine_similary(M1, M1)
    print("M1: \n", M1)
    print("M1.T: \n", M1.T)
    print("self similary: \n", sim)
    print("sim value: \n", distance_similarity_matrix(sim))

    M2= np.vstack((temp, np.array([4, 5, 7])))
    sim = matrix_cosine_similary(M1, M2)
    print("\n M1 and M2\n")
    print("M1: \n", M1)
    print("M2: \n", M2)
    print("similary: \n", sim)

    M3= np.vstack((M1, np.array([4, 5, 7])))
    print("\n M1 and M3\n")
    print("M1: \n", M1)
    print("M3: \n", M3)
    sim = matrix_cosine_similary(M1, M3)
    print("similary: \n", sim)

    print("Compare two matrix in full, same dim")
    #M4= np.array([1, 2, 3])
    #M4= np.stack(M4, np.array([4, 5, 6]))
    I = np.ones((2,2))
    M5 = np.array([10, -2, 11])
    M5 = np.stack((M5, np.array([40, 11.2, -6])))

    print("\n M1 and M2\n")
    print("M1: \n", M1)
    print("M2: \n", M2)
    sim = matrix_cosine_similary(M1, M2)
    print("similary: \n", sim)
    print("diff: \n", I - sim)
    print("avg diff: \n", np.average(I - sim))

    print("\n M1 and M5\n")
    print("M1: \n", M1)
    print("M5: \n", M5)
    sim = matrix_cosine_similary(M1, M5)
    print("similary: \n", sim)
    print("diff: \n", I - sim)
    print("avg diff: \n", np.average(I - sim))
    print("distnce: \n", distance_similarity_matrix(sim))
    # print("\n M1 and M5\n")
    # M1 = np.stack((M1, np.array([0, 4, 10])))
    # print("M1: \n", M1)
    # print("M5: \n", M5)
    # sim = matrix_cosine_similary(M1, M5)
    # print("similary: \n", sim)
    # print("diff: \n", I - sim)
    # print("avg diff: \n", np.average(I - sim))
    #print("Eigenvalues M1: ", np.linalg.eigvals(M1))
    #print("Eigenvalues M5: ", np.linalg.eigvals(M5))


if __name__ == "__main__":
        main()