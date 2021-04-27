import numpy as np
def mfc_controller(model, input, number_step, max_step):

    u_cur = input[0, 3]
    delta = 1 / number_step
    weight = np.array([1, 2, 4])
    ufinal = 0
    costfinal = 9e10
    u0 = np.array([u_cur, u_cur, u_cur])
    top = np.array([1, 1, 1])
    bot = np.array([0, 0, 0])

    for i in range(-max_step, max_step+1):
        du = delta * i
        u = u0 + np.array([du, du * 2, du * 3])
        u = np.minimum(np.maximum(bot, u), top)
        input[0, 4:7] = u
        output = model(input)[-1]
        cost = np.sum(np.abs(output)**3 * weight)
        if cost < costfinal:
            costfinal = cost
            ufinal = u[0]

    return ufinal

