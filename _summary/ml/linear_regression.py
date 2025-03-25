from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# NOTE: 보스턴 데이터셋은 윤리적(?) 문제가 있어 대신 캘리포니아 주택 데이터셋 사용
housing = fetch_california_housing()
X = housing.data
y = housing.target


# ==== Linear Regression ====
lr1 = LinearRegression()
lr1.fit(X, y)

print("Linear Regression")
for f, w in zip(housing.feature_names, lr1.coef_):
    print(f"{f:7s}: {w:6.2f}")
print(f"coef = {sum(lr1.coef_ ** 2):4.2f}")


# ==== Ridge Regression ====
lr2 = Ridge(alpha=1.0)
lr2.fit(X, y)

print("\nRidge Regression")
for f, w in zip(housing.feature_names, lr2.coef_):
    print(f"{f:7s}: {w:6.2f}")
print(f"coef = {sum(lr2.coef_ ** 2):4.2f}")

# ==== Lasso Regression ====
lr3 = Lasso(alpha=0.1)
lr3.fit(X, y)

print("\nLasso Regression")
for f, w in zip(housing.feature_names, lr3.coef_):
    print(f"{f:7s}: {w:6.2f}")
print(f"coef = {sum(lr3.coef_ ** 2):4.2f}")
