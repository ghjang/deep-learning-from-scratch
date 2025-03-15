import numpy as np

# 원본 행렬 정의
A = np.array([1, 2, 3, 4])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
C = np.array([[1, -1, 2, 0], [0, 3, -2, 1], [4, 5, 6, -3]])
D = np.array([[2, 3], [-1, 4], [5, 6], [7, 8]])

# Step 1: AB와 BA 계산
print("=== Step 1 ===")
AB = A @ B
print("AB:", AB.shape)
print(AB)

BA = B.T @ A.reshape(-1, 1)
print("\nBA:", BA.shape)
print(BA)

# Step 2: (AB)C와 C(BA) 계산
print("\n=== Step 2 ===")
ABC = AB @ C
print("ABC:", ABC.shape)
print(ABC)

CBA = C.T @ BA
print("\nCBA:", CBA.shape)
print(CBA)

# Step 3: (ABC)D와 D(CBA) 계산 - 최종 결과
print("\n=== Step 3 ===")
ABCD = ABC @ D
print("ABCD (E):", ABCD.shape)
print(ABCD)

DCBA = D.T @ CBA
print("\nDCBA (E^T):", DCBA.shape)
print(DCBA)

# 검증: (E^T)^T = E
print("\n=== 검증 ===")
print("(E^T)^T = E 성립 여부:", np.array_equal(DCBA.T, ABCD))
