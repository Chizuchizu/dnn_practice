import numpy as np
import matplotlib.pyplot as plt


# 目的関数
def J(x):
    return (x - 2) ** 3


def differential(x):
    """
    数値微分
    :param x: x
    :return: J(x)の勾配
    """
    h = 1e-5
    return (J(x + h) - J(x)) / h


x = np.linspace(-5, 5, 500)
np.random.seed(0)
stack = []

omega = np.random.rand() * 5
eta = 0.01

for i in range(50):
    stack.append(omega)
    omega = omega - eta * differential(omega)

    # 収束判定
    if eta * differential(omega) <= 0.00001:
        break

plt.xlim(1, 4)
plt.ylim(-2, 2)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$J(\omega)$")
plt.plot(x, J(x), "b", lw=3, label="a")
plt.plot(stack, J(np.array(stack)), color="r", marker="o")
plt.show()

# plt.xlabel(r"number of steps")
# plt.ylabel(r"$\omega$")
# plt.plot(stack, "r")
# plt.show()
