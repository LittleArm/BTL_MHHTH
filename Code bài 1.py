from pulp import LpVariable, LpInteger, LpProblem, lpSum, LpMinimize, LpAffineExpression

import numpy as np

# Dữ liệu
D = np.array([8, 5, 9, 8, 4, 2, 3, 7])
b = np.array([100, 200, 90, 150, 180])
s = np.array([70, 80, 40, 100, 120])
l = np.array([80, 113, 112, 70, 110, 220, 250, 100])
q = np.array([9500, 1100, 1290, 820, 1230, 2200, 2100, 1500])
A = np.array([[4, 2, 3, 1, 2],
              [0, 1, 3, 4, 0],
              [3, 2, 2, 0, 1],
              [5, 1, 2, 1, 2],
              [1, 0, 1, 1, 0],
              [0, 2, 0, 2, 1],
              [2, 2, 4, 1, 1],
              [1, 0, 0, 2, 3]])

# Tạo bài toán
problem = LpProblem(name="minimize_example", sense=LpMinimize)

# Tạo biến quyết định x là vector kích thước 5
x = [LpVariable(f"x_{i}", cat='Integer') for i in range(5)]

# Tạo biến quyết định y là vector kích thước 5
y = [LpVariable(f"y_{i}", cat='Integer') for i in range(5)]

# Tạo biến quyết định z là vector kích thước 8
z = [LpVariable(f"z_{i}", cat='Integer') for i in range(8)]

# Khởi tạo hàm mục tiêu
objective_function = lpSum(b[i] * x[i] for i in range(5)) + lpSum((l[i] - q[i]) * z[i] for i in range(8)) - lpSum(s[i] * y[i] for i in range(5))
problem += objective_function, "objective"  # Thêm hàm mục tiêu vào bài toán
# Thêm ràng buộc và các phần khác của bài toán
for j in range(5):
    problem += (y[j] == (x[j]) - lpSum(A[i][j]*z[i] for i in range(8)))
for i in range(5):
    problem += y[i] >= 0
for i in range(5):
    problem+=x[i]>=0
for i in range(8):
    problem += z[i] >= 0
for i in range(8):
    problem += z[i] <= D[i]
# Print problem
print(problem)

# The problem data is written to an .lp file
problem.writeLP("problem.lp")
# Giải bài toán
problem.solve()

# Hiển thị kết quả
print("Status:", problem.status)
print("Objective value:", problem.objective.value())
print("Optimal values for the variables:")
for variable in problem.variables():
    print(f"{variable.name}: {variable.value()}")
