import numpy as np
from numpy import dot, diag
from tensorflow import keras
import tensorflow as tf

def mfc_controller(model, state, activation, omgd, number_step, max_step):
    u_cur = activation[0, 1]
    delta = 1 / number_step
    weight = np.array([1, 1.5, 2.25])
    ufinal = 0
    costfinal = 9e10
    u0 = np.array([u_cur, u_cur, u_cur])
    top = np.array([1, 1, 1])
    bot = np.array([0, 0, 0])

    for i in range(-max_step, max_step + 1):
        du = delta * i
        cost = 0
        sum = 1e-10
        u = u0 + np.array([du, du * 2, du * 3])
        u = np.minimum(np.maximum(bot, u), top)
        activationt = activation.copy()

        activationt[0, 1] = u[0]

        activation_ = np.array([activationt])
        output, ss = model([activation_, state])

        cost += np.abs(output[-1, -1, -1]) ** 3
        sum += np.abs(output[-1, -1, -1])
        for j in range(1, 3):
            activationt[0, 0] = output[-1, -1, -1]
            activationt[0, 1] = u[j]
            activationt[0, 3] += omgd
            activationt[0, 2] += activationt[0, 3]
            activation_ = np.array([activationt])

            output, ss = model([activation_, ss])
            cost += np.abs(output[-1, -1, -1]) ** 3
            sum += np.abs(output[-1, -1, -1])	
        cost /= sum 
        if cost < costfinal:
            costfinal = cost
            ufinal = u[0]
		
    return ufinal


if __name__ == "__main__":
    ewinlist = np.load('ewinlist.npy')
    uwinlist = np.load('uwinlist.npy')
    omegalist = np.load('omegalist.npy')
    omegalist2 = omegalist[1:] - omegalist[:-1]
    u = uwinlist[-1]
    t = 30
    ewin = 0.1 + ewinlist[-1] 	
    model = keras.models.load_model("my_model")
    input_layer = tf.keras.Input(shape=(None, 4))
    ssss = tf.keras.Input(shape=(24))
    gru_layer = model.layers[2]
    outputlayer = model.layers[3]
    gru_out, final_state = gru_layer(input_layer, initial_state=ssss)
    modelout = outputlayer(gru_out)
    model2 = tf.keras.Model(
        inputs=[input_layer, ssss],
        outputs=[modelout, final_state])

    l = 5
    inputt = np.column_stack((ewinlist[-l - 1:-1], uwinlist[-l:]
                              , omegalist[-l - 1:-1], omegalist2[-l - 1:-1]))
    o, state = model2([np.array([inputt]), np.zeros((1, 24))])
    temp = np.array([[ewin, u, np.sin(np.pi * 2 * t * 1),
                      np.sin(np.pi * 2 * t * 1) - np.sin(np.pi * 2 * (t - 0.1) * 1)]])
    
    u = mfc_controller(model2, state, temp, temp[0, 3] - omegalist2[-1], 40, 200)