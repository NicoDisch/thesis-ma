import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt



def build_model(config):
    nn_width = config['width']
    nn_depth = config['depth']
    if config['initialiser'] == 'glorot':
        initializer = tf.keras.initializers.glorot_normal()
    else:
        return
    encoder_input = tf.keras.Input(shape=(2))
    for i in range(nn_depth):
        if i== 0:
            x = layers.Dense(nn_width, activation="tanh", kernel_initializer=initializer)(encoder_input)
        else:
            x = layers.Dense(nn_width, activation="tanh",kernel_initializer=initializer)(x)
    encoder_output = layers.Dense(1)(x)
    model = tf.keras.Model(encoder_input, encoder_output, name="encoder")
    return model


def train(model,config):
    training_steps = config['steps']
    nn_width = config['width']
    nn_depth = config['depth']
    nn_func = config['activation_function']
    problem_setting = config['pde']
    try:
        error_loss = np.load('error_curves/{}_{}_{}_{}.txt'.format(problem_setting,nn_width, nn_depth, nn_func))
    except:
        error_loss = np.array([1])
    ################
    # This is 2d now:
    ################
    N = 100
    x_len = np.linspace(0, 1, N).reshape((N, 1))
    x_grid = np.meshgrid(x_len, x_len)
    xx, yy = np.meshgrid(x_len, x_len)
    x_tensor = tf.convert_to_tensor(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32)

    if config['load_model']:
        print('load model!')
    best_loss = 100
    for step in range(0, training_steps):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:

            # Create tensor that you will watch
            tape.watch(x_tensor)
            # Feed forward
            output = model(x_tensor, training=True)
            y_x = tape.gradient(output, x_tensor)
            y_xx = tape.gradient(y_x,x_tensor)
            # y_y = tape.gradient(output, x_tensor[:,1])
            # y_yy = tape.gradient(y_y, x_tensor[:, 1])

            # Gradient and the corresponding loss function
            loss_direct = (tf.reduce_mean(input_tensor=(y_xx + 1) ** 2)
                           # + 100*tf.square(y_x[0]-1)
                           + tf.reduce_mean(input_tensor= tf.square(output[0, :]))
                           + tf.reduce_mean(input_tensor=tf.square(output[-1, :]))
                           # + 100*tf.square(output[0]-output[-1])
                           )
        # pick optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        grads_d = tape.gradient(loss_direct, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads_d, model.trainable_weights))

        # Log every 200 batches
        if step % 50 == 0:
            arr_loc = np.array([float(loss_direct)])
            error_loss = np.append([error_loss], [arr_loc])
            print(
                "Training loss in  step:",
                step, float(loss_direct)
            )
            if best_loss > float(loss_direct):
                # model.save_weights('checkpoints/lap_energy_best.h5')
                best_loss = float(loss_direct)
    print(best_loss)
    np.save('error_curves/{}_{}_{}_{}.txt'.format(problem_setting,nn_width,nn_depth, nn_func), error_loss)
    return best_loss, float(loss_direct)


def write_results(model, config, accuracies):
    nn_width = config['width']
    nn_depth = config['depth']
    nn_func = config['activation_function']
    problem_setting = config['pde']
    training_steps = config['steps']
    with open('saves/{}_{}_{}_{}.txt'.format(problem_setting,nn_width,nn_depth, nn_func), 'w') as f:
        f.write("DGL: Laplace")
        f.write("\n")
        f.write("f: 1")
        f.write("\n")
        f.write("Width:" + str(nn_width))
        f.write("\n")
        f.write("Depth:" + str(nn_depth))
        f.write("\n")
        f.write("Activation function:" + str(nn_func))
        f.write("\n")
        f.write("Training steps:" + str(training_steps))
        f.write("\n")
        f.write("Best loss:" + str(accuracies[0]))
        f.write("\n")
        f.write("Last loss:" + str(accuracies[1]))
        # stepsize = 50

    model.save('checkpoints2d/{}_{}_{}_{}.h5'.format(problem_setting,nn_width,nn_depth, nn_func))


def load_model(config):
    nn_width = config['width']
    nn_depth = config['depth']
    nn_func = config['activation_function']
    problem_setting = config['pde']
    return tf.keras.models.load_model('checkpoints2d/{}_{}_{}_{}.h5'.format(problem_setting,nn_width,nn_depth, nn_func))




def plot_decision_boundary(model, steps=100, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cmap = plt.get_cmap(cmap)
    #steps = 1000
    x_span = np.linspace(0, 1, steps)
    print(len(x_span))
    y_span = np.linspace(0, 1, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # boundary = np.outer(bounday_fct(x_span),bounday_fct(y_span))
    #print(boundary.shape)
    # Make predictions across region of interest
    func_values = model.predict(np.c_[xx.ravel(), yy.ravel()])#*\
             #
    print(func_values.shape)
    # Plot decision boundary in region of interest
    z = func_values.reshape(xx.shape)#*boundary

    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.7)
    # contourf

    # Get predicted labels on training data and plot
    #ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    plt.show()
    return fig, ax



