import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
from matplotlib import animation as ani

# set graph range
x_low = -4
x_high = 4
y_low = -4
y_high = 4

# set field
X = np.linspace(x_low, x_high, 1000)
Y = np.linspace(y_low, y_high, 1000)
X, Y = np.meshgrid(X, Y)


def J(x, y):
    return 3 * x ** 2 + 5 * y ** 2 - 6 * x * y
    # return 2 * x ** 3 + 4 * y ** 2 - 6 * x ** 2 + x * y - x * y * y


Z = J(X, Y)


def get_grad_vec_num(x, y):
    h = 1e-5
    return [(J(x + h, y) - J(x, y)) / h, (J(x, y + h) - J(x, y)) / h]


def get_grad_vec(x, y):
    grad_x = 6 * x - 6 * y
    grad_y = 10 * y - 6 * x
    return [grad_x, grad_y]


def sgd(x, y, grads, lr=0.01):
    n_xs = x - lr * grads[0]
    n_ys = y - lr * grads[1]

    return n_xs, n_ys


def momentum(x, y, grads, list_w, lr=0.01, alpha=0.9):
    """
        if list_w[0]:
        delta_w_t_x = alpha * list_w[0][-1] - (1 - alpha) * lr * grads[0]
        delta_w_t_y = alpha * list_w[1][-1] - (1 - alpha) * lr * grads[1]
        return x - delta_w_t_x, y - delta_w_t_y
    else:
        delta_w_t_x = -(1 - alpha) * lr * grads[0]
        delta_w_t_y = -(1 - alpha) * lr * grads[1]
        return x + delta_w_t_x, y + delta_w_t_y
    :param x:
    :param y:
    :param grads:
    :param list_w:
    :param lr:
    :param alpha:
    :return:
    """
    if list_w[0]:
        delta_w_t_x = alpha * list_w[0][-1] + lr * grads[0]
        delta_w_t_y = alpha * list_w[1][-1] + lr * grads[1]
        return x - delta_w_t_x, y - delta_w_t_y, delta_w_t_x, delta_w_t_y
    else:
        delta_w_t_x = -(1 - alpha) * lr * grads[0]
        delta_w_t_y = -(1 - alpha) * lr * grads[1]
        return x + delta_w_t_x, y + delta_w_t_y, delta_w_t_x, delta_w_t_y


def adagrad(x, y, grads, list_w, lr=0.01):
    list_w[0].append(x)
    list_w[1].append(y)

    memo_x = sum([xx ** 2 for xx in list_w[0]])
    memo_y = sum([yy ** 2 for yy in list_w[1]])

    delta_w_x = -(lr / np.sqrt(memo_x)) * grads[0]
    delta_w_y = -(lr / np.sqrt(memo_y)) * grads[1]
    # print(memo_x, delta_w_y)
    return delta_w_x, delta_w_y


def rmsprop(x, y, grads, list_w, list_v, lr, alpha=0.9):
    if list_w[0]:
        v_x_t = alpha * list_v[0][-1] + (1 - alpha) * (grads[0] ** 2)
        v_y_t = alpha * list_v[1][-1] + (1 - alpha) * (grads[1] ** 2)
    else:
        v_x_t = (1 - alpha) * (grads[0] ** 2)
        v_y_t = (1 - alpha) * (grads[1] ** 2)

    delta_w_x = -(lr / (np.sqrt(v_x_t + 1e-6))) * grads[0]
    delta_w_y = -(lr / (np.sqrt(v_y_t + 1e-6))) * grads[1]

    return delta_w_x, delta_w_y, v_x_t, v_y_t


def calc_2val_norm(init_x=4, init_y=0, learning_ratio=.1, precision=3):
    list_xs = []
    list_ys = []
    list_nxs = []
    list_nys = []
    list_diff = []

    list_vx = []
    list_vy = []

    xs = init_x
    ys = init_y
    i = 0
    for i in range(50):

        grad_vec = get_grad_vec_num(xs, ys)
        # n_xs, n_ys, vx, vy = rmsprop(xs, ys, grad_vec, [list_xs, list_ys], [list_vx, list_vy], learning_ratio)
        n_xs, n_ys, vx, vy = momentum(xs, ys, grad_vec, [list_vx, list_vy])
        list_xs.append(xs)
        list_ys.append(ys)
        list_nxs.append(n_xs)
        list_nys.append(n_ys)

        list_vx.append(vx)
        list_vy.append(vy)

        # judge convergence
        diff = np.sqrt(grad_vec[0] ** 2 + grad_vec[1] ** 2)
        print(diff)
        list_diff.append(diff)
        if diff < 0.1 ** precision:
            break

        xs = n_xs
        ys = n_ys

    ret_dict = {}
    ret_dict['num'] = i + 1
    ret_dict['list_xs'] = list_xs
    ret_dict['list_ys'] = list_ys
    ret_dict['list_nxs'] = list_nxs
    ret_dict['list_nys'] = list_nys
    ret_dict['list_diff'] = list_diff

    return ret_dict


def animate(i):
    list_xs = ret_dict['list_xs']
    list_ys = ret_dict['list_ys']
    list_nxs = ret_dict['list_nxs']
    list_nys = ret_dict['list_nys']
    list_diff = ret_dict['list_diff']

    if i == 0:
        plt.scatter(list_xs[i], list_ys[i], s=20, c="b", alpha=0.6)
        plt.title("n %2d, x %.5f, y %.5f, diff %.5f" % (i, list_xs[i], list_ys[i], list_diff[i]))
    else:
        # draw graph
        plt.scatter(list_xs[i - 1], list_ys[i - 1], s=20, c="b", alpha=0.6)
        plt.plot([list_xs[i - 1], list_nxs[i - 1]], [list_ys[i - 1], list_nys[i - 1]])
        plt.title("n %2d, x %.5f, y %.5f, diff %.5f" % (i, list_xs[i - 1], list_ys[i - 1], list_diff[i - 1]))


fig = plt.figure(figsize=(6, 4))
ret_dict = calc_2val_norm(init_x=0, init_y=-3, learning_ratio=.01)
interval = [x ** 2 for x in range(50)]
CS = plt.contour(X, Y, Z, interval)
plt.clabel(CS, inline=0, fontsize=-3)
print(ret_dict['num'])

# Writer = ani.writers['ffmpeg']
# writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)

anim = ani.FuncAnimation(fig, animate, frames=ret_dict['num'], blit=False)
anim.save('quadratic_decent_anim_2.gif', fps=10)
