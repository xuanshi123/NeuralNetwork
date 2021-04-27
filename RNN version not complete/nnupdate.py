import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import socket
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import initializers
import time
import signal
from subprocess import Popen, PIPE
# from osysid import OnlineLinearModel
from mpc import *
import logging
from tensorflow.keras.callbacks import CSVLogger

# import pysindy as ps

csv_logger = CSVLogger('training.log', append=True)

logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

port = 10005
process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
for process in str(stdout.decode("utf-8")).split("\n")[1:]:
    data = [x for x in process.split(" ") if x != '']
    if (len(data) <= 1):
        continue
    os.kill(int(data[1]), signal.SIGKILL)

time.sleep(1)
# Create a TCP/IP socket

num_input = 4
num_output = 1
num_nuero = 24
num_nuero2 = num_output + 1
numpercycle = 10

number_step = 200
max_step = 50
freq = 1
F1 = 1.8
ud = 0
tstart = 0
lr = 0.0007

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

earray = np.zeros(2)
u = 0
ewin = 0
ewinold = 0
ewinlist = np.array([])
uwinlist = np.array([])
omegalist = np.array([])
omegalist2 = np.array([])
MFC_start = 0
input_ = []
output_ = []
inputt = []
loss = 0.99

if -1 < 0 and os.path.isfile('earray.npy'):
    earray = np.load('earray.npy')
    MFC_start = earray[0]
    ewinlist = np.load('ewinlist.npy')
    uwinlist = np.load('uwinlist.npy')
    omegalist = np.load('omegalist.npy')
    omegalist2 = omegalist[1:] - omegalist[:-1]
    u = uwinlist[-1]
    model = keras.models.load_model("my_model")
    input_layer = tf.keras.Input(shape=(None, num_input))
    ssss = tf.keras.Input(shape=(num_nuero))
    gru_layer = model.layers[2]
    outputlayer = model.layers[3]
    gru_out, final_state = gru_layer(input_layer, initial_state=ssss)
    modelout = outputlayer(gru_out)
    model2 = tf.keras.Model(
        inputs=[input_layer, ssss],
        outputs=[modelout, final_state])
else:

    input_layer = tf.keras.Input(shape=(None, num_input))

    ssss = tf.keras.Input(shape=(num_nuero))

    gru_layer = keras.layers.RNN(
        keras.layers.GRUCell(num_nuero), input_shape=(None, num_input), return_sequences=True, return_state=True)

    gru_out, final_state = gru_layer(input_layer, initial_state=ssss)

    outputlayer = keras.layers.Dense(num_output, kernel_initializer=initializers.RandomUniform(minval=-0.5, maxval=0.5),
                                     activation=None, use_bias=False)

    modelout = outputlayer(gru_out)

    model = tf.keras.Model(
        inputs=[input_layer, ssss],
        outputs=[modelout])

    model2 = tf.keras.Model(
        inputs=[input_layer, ssss],
        outputs=[modelout, final_state])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.MeanSquaredError())

# print(model.layers[2])
io = np.zeros((4, 10, 4))
oo = np.ones((4, 10, 1))
state = np.zeros((4, 24))
# o = model.fit([io, state], oo)
# o1 = model([io, state], oo)
# gru_layer.return_sequences = False
# o, st = model2([io, state], oo)
# print(o.shape, st.shape)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_address = ('localhost', port)
print('starting up on port', server_address)

sock.bind(server_address)

sock.listen(1)
logging.info('{}:starting up on port'.format(port))

try:

    while True:
        connection, client_address = sock.accept()
        # print('recieve connection')

        data = connection.recv(1024)
        data_print = data.decode().split(",")

        if len(data_print) == 3:

            e = float(data_print[0]) / 0.004
            # u = float(data_print[1]) / 1.8
            t = float(data_print[2])
            # print('error', e, 'u', u, 'time', t)

            earray = np.concatenate((earray, [e]))
            ewinold = ewin

            if len(earray) < numpercycle + 2:
                ewin = e
            else:
                ewin = np.mean(earray[-numpercycle:])

            if MFC_start == 1:
                l = 5
                inputt = np.column_stack((ewinlist[-l:], uwinlist[-l:]
                                          , omegalist[-l:], omegalist2[-l:]))
                o, state = model2([np.array([inputt]), np.zeros((1, num_nuero))])
                temp = np.array([[ewin, u, np.sin(np.pi * 2 * t * 1),
                                  np.sin(np.pi * 2 * t * 1) - np.sin(np.pi * 2 * (t - 0.1) * 1)]])
                u = mfc_controller(model2, state, temp, temp[0, 3] - omegalist2[-1], number_step, max_step)

            elif len(ewinlist) > 5:
                u += 0.4 * ewin + 0.04 * (ewin - ewinlist[-1])

            u = 0 if u < 0 else u
            u = 1 if u > 1 else u
            u = round(u * number_step) / number_step

            data = "{0:.8g}".format(u * F1)

            connection.sendall(data.encode())

            ewinlist = np.append(ewinlist, [ewin])
            omegalist = np.append(omegalist, [np.sin(np.pi * 2 * t * 1)])
            omegalist2 = np.append(omegalist2, [np.sin(np.pi * 2 * t * 1) - np.sin(np.pi * 2 * (t - 0.1) * 1)])
            uwinlist = np.append(uwinlist, [u])

            if 3 < len(ewinlist) <= 10:
                l = len(ewinlist)
                inputt = np.column_stack((ewinlist[-l:], uwinlist[-l:]
                                          , omegalist[-l:], omegalist2[-l:]))
                io = np.array([inputt])

                gru_layer.return_sequences = False
                model.fit([io[:, :3], np.zeros((1, num_nuero))], io[:, 3, :1], epochs=1)

                o, state = model2([io[:, :3], np.zeros((1, num_nuero))])

                gru_layer.return_sequences = True

                if l > 4:
                    history = model.fit([io[:, 3:-1], state], io[:, 4:, :1], epochs=1, callbacks=[csv_logger])
                    loss = history.history['loss'][-1]

            elif len(ewinlist) > 10:
                l = 10
                inputt = np.column_stack((ewinlist[-l:], uwinlist[-l:]
                                          , omegalist[-l:], omegalist2[-l:]))

                io = np.concatenate((io, [inputt]), axis=0)[-5:]
                io2 = io[-1::-2]
                ll = len(io2)

                for k in range(4):

                    gru_layer.return_sequences = False
                    model.fit([io2[:, :3], np.zeros((ll, num_nuero))], io2[:, 3, :1], epochs=1)

                    o, state = model2([io2[:, :3], np.zeros((ll, num_nuero))])

                    gru_layer.return_sequences = True

                    history = model.fit([io2[:, 3:-1], state], io2[:, 4:, :1], epochs=1, callbacks=[csv_logger])

                loss = history.history['loss'][-1]

            logging.info('u {} ewin {} loss {} t {}'.format(u, ewin, loss, t))

            if loss < 0.01 and MFC_start == 0 and len(ewinlist) > 30:
                MFC_start = 1
                model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr),
                              loss=keras.losses.MeanSquaredError())
                logging.info('MFC Start')

            if loss > 0.06 and MFC_start == 1:
                MFC_start = 0
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                              loss=keras.losses.MeanSquaredError())



        elif len(data_print) == 1:
            e = float(data_print[0]) / 0.004
            # print(e)
            earray = np.concatenate((earray, [e]))

        elif len(data_print) == 2 and len(uwinlist) >= 1:

            save_path = 'time/' + data_print[1]

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            earray[0] = MFC_start
            np.save("earray", earray)
            np.save(save_path + "/earray", earray)
            model.save(save_path + "/my_model")
            np.save(save_path + "/ewinlist", ewinlist)
            np.save(save_path + "/uwinlist", uwinlist)
            np.save(save_path + "/omegalist", omegalist)

        connection.close()


except Exception:
    logging.exception("Fatal error in main loop")
