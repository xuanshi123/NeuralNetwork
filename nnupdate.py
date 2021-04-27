import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import socket
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import CSVLogger
import signal
from subprocess import Popen, PIPE
import logging
from mpc import *
import SwarmPackagePy as sp

port = 10000
lr = 0.0007
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

if os.path.isfile('training.log'):
    os.remove('training.log')

csv_logger = CSVLogger('training.log', append=True)

try:

    process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    for process in str(stdout.decode("utf-8")).split("\n")[1:]:
        data = [x for x in process.split(" ") if x != '']
        if (len(data) <= 1):
            continue
        os.kill(int(data[1]), signal.SIGKILL)

    # Create a TCP/IP socket

    num_output = 3
    num_input = num_output + 2 + 2 + 2
    num_nuero = 18
    num_nuero2 = num_output + 1
    numpercycle = 10
    freq = 1
    F1 = 1.8
    ud = 0
    lim = 0.004
    tstart = 0
    uwin = np.zeros(5)
    ewin = np.zeros(0)
    dt = num_output / numpercycle
    input_ = np.zeros((0, num_input))
    output_ = np.zeros((0, num_output))
    loss = 99
    number_step = 200
    delta = 1 / number_step
    max_step = 60
    MFC_start = 0

    if os.path.isfile('earray.npy'):
        logging.info("Model File Exist")
        model = keras.models.load_model("my_model")
        earray = np.load('earray.npy')
        qlist = np.load('Q.npy')
        MFC_start = earray[0]
        uwin = qlist[-5:]
        ud = qlist[0]
        u = qlist[-1]
    else:

        model = tf.keras.Sequential([
            keras.Input(shape=(num_input,)),
            keras.layers.Dense(num_nuero, kernel_initializer=initializers.RandomUniform(minval=-0.5, maxval=1),
                               bias_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                               activation='sigmoid'),
            keras.layers.Dense(num_output, kernel_initializer=initializers.RandomUniform(minval=-0.5, maxval=1),
                               activation='tanh', use_bias=False)
        ])

        model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr), loss=keras.losses.MeanSquaredError())
        earray = np.zeros(2)
        u = 0
        ewin = np.zeros(0)
        epold = 0
        qlist = np.zeros(1)

    model = keras.models.load_model("my_model")
    model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr), loss=keras.losses.MeanSquaredError())

    # model.loss_weights = [1, 0.9, 0.81, 0.729, 0.6561]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('localhost', port)
    logging.info('{}:starting up on port'.format(port))
    sock.bind(server_address)
    sock.listen(1)

    input1 = np.zeros((1, num_input))
    bin = np.array([])


    def cost_function(x):
        u_cur = input1[0, 3]
        top = np.array([1, 1, 1])
        bot = np.array([0, 0, 0])
        u = np.array([u_cur, u_cur, u_cur])
        u += np.array([x[0], x[0] + x[1], x[0] + x[1] + x[2]])
        u = np.minimum(np.maximum(bot, u), top)
        input1[0, 4:7] = u
        output = model(input1)[-1]
        cost = output[2] ** 2 * np.abs(output[2])
        return cost


    while True:
        connection, client_address = sock.accept()
        # print('recieve connection')

        data = connection.recv(1024)
        data_print = data.decode().split(",")
        logging.info(data)
        if len(data_print) == 3:

            e = float(data_print[0]) / lim
            # u = float(data_print[1]) / F1
            t = float(data_print[2])

            earray = np.concatenate((earray, [e]))

            if len(ewin) == 0:
                if len(earray) < numpercycle + 2:
                    ewin = np.repeat(np.mean(earray[2:]), num_output + 2)
                else:
                    eave = np.mean(earray[-numpercycle:])
                    ewin = np.array([eave])
                    for i in range(1, num_output + 2):
                        if len(earray) < numpercycle * (i + 1) + 2:
                            ewin = np.concatenate(([ewin[0]], ewin))
                            continue
                        eave = np.mean(earray[-numpercycle * (i + 1):-numpercycle * i])
                        ewin = np.concatenate(([eave], ewin))
            else:
                eave = np.mean(earray[-numpercycle:])
                ewin = np.concatenate((ewin, [eave]))[-num_output - 2:]

            if len(qlist) >= 2:
                uwin = np.concatenate((uwin, [qlist[-1]]))[-num_output - 2:]

            uold = u
            ud = 0.4 * ewin[-1] + 0.04 * (ewin[-1] - ewin[-2])

            if MFC_start == 1:
                input1 = np.array([np.concatenate((ewin[-2:], qlist[-2:], qlist[-1:], qlist[-1:], qlist[-1:],
                                                   [np.sin(np.pi * 2 * t * freq), np.cos(np.pi * 2 * t * freq)]))])

                u = mfc_controller(model, input1, number_step, max_step)

                logging.info('MFC start')
                # bin = sp.cso(40, cost_function, 0, 0.3, 3, 15, pa=0.3, nest=40).get_Gbest()
                # bin2 = sp.cso(40, cost_function, -0.3, 0, 3, 15, pa=0.3, nest=40).get_Gbest()

                # if cost_function(bin) < cost_function(bin2):
                    # u += bin[0]
                # else:
                    # u += bin2[0]
                # bin = np.delete(bin, 0)

                if uold > 0.95 and uold + ud < u:
                    u = uold + 3 * ud

                if uold < 0.05 and uold + ud > u:
                    u = uold + 3 * ud

                if u > 1 or u < 0:
                    u = 0 if u < 0 else 1

            if MFC_start == 0:
                u = uold + ud
                if u > 1 or u < 0:
                    u = 0 if u < 0 else 1

            u = round(u / delta) * delta
            data = "{0:.8g}".format(u * F1)

            connection.sendall(data.encode())

            t0 = t - dt
            input = [np.concatenate((ewin[:2], uwin,
                                     [np.sin(np.pi * 2 * t0 * freq), np.cos(np.pi * 2 * t0 * freq)]))]

            input_ = np.concatenate((input, input_), axis=0)[:10]
            output = [ewin[-3:]]
            output_ = np.concatenate((output, output_), axis=0)[:10]

            history = model.fit(input_[0::2], output_[0::2], epochs=4
                                , validation_data=(input_[-9::2], output_[-9::2]), callbacks=[csv_logger])
            loss = history.history['val_loss'][-1]

            logging.info('u {} loss {} t {}'.format(u, loss, t))

            if loss < 0.01:
                MFC_start = 1
            if loss > 0.02:
                MFC_start = 0
                bin = np.array([])

            qlist = np.concatenate((qlist, [u]))

        elif len(data_print) == 1:
            e = float(data_print[0]) / lim
            earray = np.concatenate((earray, [e]))

        elif len(data_print) == 2 and len(qlist) > 1:

            save_path = data_print[1]

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            earray[0] = MFC_start
            np.save("earray", earray)
            np.save(save_path + "/earray", earray)
            model.save("my_model")
            model.save(save_path + "/my_model")
            np.save(save_path + "/Q", qlist)
            np.save("Q", qlist)
            np.savez(save_path + "/io", input_, output_)
            np.savez("io", input_, output_)

        connection.close()

except Exception:
    logging.exception("Fatal error in main loop")

















