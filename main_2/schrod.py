import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
from keras.models import load_model
#import tensorflow.compat.v1 as tfc

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
#from plotting import newfig, savefig
#from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

tf.compat.v1.disable_eager_execution()

class PhysicsInformedNN(tf.keras.Model):
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, pinn_layers, lb, ub):
        super(PhysicsInformedNN, self).__init__()
        self.model = tf.keras.Model
        X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0

        # Initialize NNs
        self.pinn_layers = pinn_layers
        self.pinn_weights, self.biases = self.initialize_NN(pinn_layers)

        # tf Placeholders
        self.x0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.x_lb_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss = tf.reduce_mean(input_tensor=tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(input_tensor=tf.square(self.f_v_pred))

        # Optimizers
        self.optimizer = tf.keras.optimizers.Adam(self.loss)
        self.save_weights('schrod_sol.h5', overwrite=True)
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        #                                                         method='L-BFGS-B',
        #                                                         options={'maxiter': 50000,
        #                                                                 'maxfun': 50000,
        #                                                                  'maxcor': 50,
        #                                                                  'maxls': 50,
        #                                                                  'ftol': 1.0 * np.finfo(float).eps})
        # learning rate = 0.001 -> jumps around
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00065)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)

    def initialize_NN(self, pinn_layers):
        weights = []
        biases = []
        num_layers = len(pinn_layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[pinn_layers[l], pinn_layers[l + 1]])
            b = tf.Variable(tf.zeros([1, pinn_layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, t):
        X = tf.concat([x, t], 1)

        uv = self.neural_net(X, self.pinn_weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = tf.gradients(ys=u, xs=x)[0]
        v_x = tf.gradients(ys=v, xs=x)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = tf.gradients(ys=u, xs=t)[0]
        u_xx = tf.gradients(ys=u_x, xs=x)[0]

        v_t = tf.gradients(ys=v, xs=t)[0]
        v_xx = tf.gradients(ys=v_x, xs=x)[0]

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return f_u, f_v

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):
        self.save_weights('schrod_start.h5', overwrite=True)
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
             # , save_format='tf')
            # Print
            best_loss = 10
            if it % 10 == 0:
                #save every 100
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
                if np.abs(loss_value) < best_loss:
                    self.save_weights('checkpoints/schrod_best.h5', overwrite=True)
                    best_loss = np.abs(loss_value)


        self.save_weights('schrod_final.h5', overwrite=True)

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    noise = 0.0

    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat("Data/schrod/NLS.mat")

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    do_train = 'train'
    if do_train =='train':
        start_time = time.time()
        # approx 12k overnight
        model.train(60)
        #model.build(tf.shape(X_star))
        u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
        model.save('checkpoints/my_model100', save_format='tf')
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
    else:
        #model.built = True
        #model.load_weights('my_model100.h5')
        model = tf.keras.models.load_model('checkpoints/my_model100.h5')
        u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)


    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = plt.subplots()
    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[:, 75], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[:, 100], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact_h[:, 125], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[125, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    fig.savefig('figures/NLS_120')