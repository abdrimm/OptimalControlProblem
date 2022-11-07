import time
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy.interpolate import UnivariateSpline
from scipy import integrate

def method_euler_straight_ahead(func_1, func_2, init1, init2, n, h):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = init1
    y[0] = init2
    for k in range(1, n):
        x[k] = x[k-1] + h * func_1[k-1]
        y[k] = y[k-1] + h * func_2[k-1]
    return [x, y]


def method_euler_in_reverse(func_1, func_2, init1, init2, n, h):
    x = np.zeros(n)
    y = np.zeros(n)
    x[n-1] = init1
    y[n-1] = init2
    for k in range(n-2, -1, -1):
        x[k] = x[k+1] - h * func_1[k+1]
        y[k] = y[k+1] - h * func_2[k+1]
    return [x, y]


def animate(data, iteration):
    names = ['x1', 'x2', 'u1', 'u2']
    for k in range(4):
        fig, ax = plt.subplots()
        plt.title(names[k] + '(t)')
        plt.xlabel('t')
        plt.ylabel(names[k])
        camera = Camera(fig)
        for j in range(iteration):
            ax.plot(np.arange(number_of_partitions) * 5.0 / number_of_partitions, data[j][k], label=names[k])
            ax.text(1, 1, 'iterations: ' + str(j))
            camera.snap()
        animation = camera.animate()
        animation.save('Animate_' + names[k] + '.gif')
        plt.close()


def f(x):
    return f1(x) ** 2 + f2(x) ** 2


if __name__ == "__main__":
    tic = time.perf_counter()
    number_of_partitions = 1000
    step = 5.0 / number_of_partitions
    u1 = np.zeros(number_of_partitions)
    u2 = np.zeros(number_of_partitions)
    x1 = np.zeros(number_of_partitions)
    x2 = np.zeros(number_of_partitions)
    box = [[-1, 1], [-1, 1]]
    A = 10
    max_iteration = 1000
    x1_init = 1
    x2_init = -1
    psi1_init = ((-2) * x1[number_of_partitions-1] + 2) * A
    psi2_init = ((-2) * x2[number_of_partitions-1] + 2) * A
    alpha = 0.01
    eps = 0.001
    dataSet = np.empty((max_iteration, 4, number_of_partitions))
    dataSet[0] = np.array([x1, x2, u1, u2])
    i = 0
    J_next = 0
    for i in range(500):
        i += 1
        print("iteration: " + str(i))
        x1, x2 = method_euler_straight_ahead(u1, u2, x1_init, x2_init, number_of_partitions, step)
        psi1_init = ((-2) * x1[number_of_partitions - 1] + 2) * A
        psi2_init = ((-2) * x2[number_of_partitions - 1] + 2) * A
        psi1, psi2 = method_euler_in_reverse(2 * x1, 2 * x2, psi1_init, psi2_init, number_of_partitions, step)
        u1 = u1 + alpha * psi1
        u2 = u2 + alpha * psi2
        for j in range(0, number_of_partitions):
            if u1[j] < box[0][0]:
                u1[j] = box[0][0]
            if u1[j] > box[0][1]:
                u1[j] = box[0][1]
            if u2[j] < box[1][0]:
                u2[j] = box[1][0]
            if u2[j] > box[1][1]:
                u2[j] = box[1][1]
        f1 = UnivariateSpline(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x1)
        f2 = UnivariateSpline(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x2)
        J_pred = J_next
        J_next, err = integrate.quad(f, 0, 3)
        dataSet[i] = np.array([x1, x2, u1, u2])
        if (abs(J_next - J_pred) <= eps):
            print('Is finished!')
            break
        if i == max_iteration - 1:
            print('Not finished')
            break
    tac1 = time.perf_counter()
    fig, ax = plt.subplots()
    plt.title('Фазовые переменные')
    plt.xlabel('t')
    plt.ylabel('z')
    ax.plot(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x1, label="x1(t)")
    ax.plot(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x2, label="x2(t)")
    ax.legend()
    fig.savefig('Фазовые_переменные.png')
    plt.close()
    fig, ax = plt.subplots()
    plt.title('Управления')
    plt.xlabel('t')
    plt.ylabel('u')
    ax.plot(np.arange(number_of_partitions) * 5.0 / number_of_partitions, u1, label="u1(t)")
    ax.plot(np.arange(number_of_partitions) * 5.0 / number_of_partitions, u2, label="u2(t)")
    ax.legend()
    fig.savefig('Управления.png')
    plt.close()
    animate(dataSet, i)
    f1 = UnivariateSpline(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x1)
    f2 = UnivariateSpline(np.arange(number_of_partitions) * 5.0 / number_of_partitions, x2)
    J_min, err = integrate.quad(f, 0, 3)
    tac2 = time.perf_counter()
    f = open('Конец_данные.txt', 'w')
    f.write('Время работы алгоритма: ' + str(tac1 - tic) + '\n')
    f.write('Время работы программы: ' + str(tac2 - tic) + '\n')
    f.write('Количество итераций: ' + str(i) + '\n')
    f.write('Значение функционала: ' + str(J_min))
    f.close()

#%%
