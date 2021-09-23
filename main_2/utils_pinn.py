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
            border = tf.reshape(output,(N,N))
            #print(border.shape)
            y_x = tape.gradient(output, x_tensor)
            y_xx = tape.gradient(y_x,x_tensor)
            #function_output = tf.reshape(y_xx,(N,N))


            # y_y = tape.gradient(output, x_tensor[:,1])
            # y_yy = tape.gradient(y_y, x_tensor[:, 1])
            #print(output.shape)
            # Gradient and the corresponding loss function
            loss_direct = (tf.reduce_mean(input_tensor=(y_xx + 1) ** 2)
                           # + 100*tf.square(y_x[0]-1)
                           + tf.reduce_mean(input_tensor= tf.square(border[0, 0]))
                           + tf.reduce_mean(input_tensor=tf.square(border[-1, 0]))
                           + tf.reduce_mean(input_tensor=tf.square(border[0, -1]))
                           + tf.reduce_mean(input_tensor= tf.square(border[-1, -1]))
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
    # Make predictions across region of interest
    func_values = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(func_values.shape)
    # Plot decision boundary in region of interest
    z = func_values.reshape(xx.shape)#*boundary
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.7)
    #ax.contourf(xx, yy, z, alpha=0.7)
    # contourf
    plt.show()
    return fig, ax


def fill_max_1(input_size):
    """
    Fills a matrix on the diagonal with the max matrix (see proof)
    alternatively use np.kron the Kronecker product
    :param input_size:the amount of max_arrays needed
    :return: the Correct weight matrix for max ReLU network, where max_arr is on the diagonal
    i.e. [max_arr,..,0],... ,[0,...,max_arr]
    """
    max_arr = np.array([[1, -1], [0, 1], [0, -1]])
    matrix = np.zeros((3*input_size, 2*input_size))
    half_length = matrix.shape[1]/2
    for i in range(int(half_length)):
        matrix[3*i:3*i+3, 2*i:2*i+2] = max_arr
    return matrix.T


def bias_1(input_size):
    """
    bias vector
    :param input_size:
    :return: returns zero vector of correct size
    """
    return np.zeros((3*input_size))

def fill_max_2(input_size):
    """
    creates the 2nd matrix for max ReLU, similar to fill_max_1
    :param input_size: amount of arrays needed
    :return: weight matrix
    """
    matrix = np.zeros((input_size, 3*input_size))
    max_array2 = np.array([1, 1, -1])
    half_length = matrix.shape[0]
    for i in range(int(half_length)):
        matrix[i:i+1, 3*i:3*i+3] = max_array2
    return matrix.T


def bias_2(input_size):
    """
    bias vector for the 2nd layer, see bias_1
    :param input_size: correct shape
    :return: just a zero vector
    """
    return np.zeros(( input_size ))


def model_init(entry_size):
    """
    set the model weights of a ReLU max unit with N inputs
    !! for now entry_size == 2^n for some n, pad with zero !!
    :param entry_size: 2^n input vector size
    :return: returns a ReLU model, with set weights
    """
    mat_array = np.array([])
    if entry_size == 1:
        print('already min')
    else:
        in_net = tf.keras.Input(shape=(entry_size))
        entry_size = int(entry_size/2)
        mat_array = np.append(mat_array, [[fill_max_1(entry_size),bias_1(entry_size)],
                                          [fill_max_2(entry_size),bias_2(entry_size)]])
        net = layers.Dense(3*entry_size, activation="relu",bias_initializer='zeros')(in_net)
        net = layers.Dense(entry_size,activation=lambda xy: xy,bias_initializer='zeros')(net)
    while entry_size !=1:
        print(entry_size)
        entry_size = int(entry_size/2)
        mat_array = np.append(mat_array, [[fill_max_1(entry_size), bias_1(entry_size)],
                                          [fill_max_2(entry_size), bias_2(entry_size)]])
        net = layers.Dense(3*entry_size, activation="relu",bias_initializer='zeros')(net)
        net = layers.Dense(entry_size,activation=lambda xy: xy,bias_initializer='zeros')(net)
    model_max = tf.keras.Model(in_net, net, name="min_max")
    model_max.set_weights(mat_array)
    return model_max


def make_arr(model_in):
    """
    this function reads each input shape of a layer, and appends it to an array
    since the models instaciated had ([None, input_size]), the first layer had to be different
    :param model_in: put in the model
    :return: an array, which contains layer sizes
    """
    arr = np.array([])
    for i, layer in enumerate(model_in.layers):
        if i == 0:
            arr = np.append(arr, layer.output_shape[0][1])
        if i > 0:
            arr = np.append(arr, layer.output_shape[1])
    return arr


def paral_arr(*arg):
    """
    adds layer depths of multiple models into one array
    !!! This works now if all networks have the same depth !!!
    :param arg: multiple keras models
    :return: an array containing layer sizes
    """
    for i, mod in enumerate(arg):
        # add all arrays together
        if i == 0:
            arr_return = make_arr(mod)
        else:
            arr_return = arr_return + make_arr(mod)
    # make sure the inputs are all integers
    return [int(a) for a in arr_return]


def paral_model(*arg):
    """
    builds a keras model, that has the correct shape for the Parallelisation of several Keras models
    !!! Keras models need to be the same depth !!!
    -> Pad other layers as alternative with Identity layers #TODO
    :param arg: multiple keras models
    :return: a Keras model that has the correct layer shapes
    """
    arr = paral_arr(*arg)
    for i, layer_array in enumerate(arr):
        if i == 0:
            # first layer, needs inout
            begin = tf.keras.Input(shape=(layer_array))
        elif i == 1:
            # chain the first layer (have at least depth 2)
            net = layers.Dense(layer_array, activation="relu", bias_initializer='zeros')(begin)
        else:
            # all the other layers
            net = layers.Dense(layer_array, activation="relu", bias_initializer='zeros')(net)
    # return the built model
    return tf.keras.Model(begin, net, name="parallel_model")


def parallel_weight_matrices_old(*arg):
    """
    This is the core function to make several models in parallel.
    This has no Keras functionality, it merely looks at the weight matrices
    !!!Same depth needed, may also only work with MAX-ReLU models!!!
    !! DEPRECATED - use updated  !!
    :param arg: several Keras models
    :return: a solution matrix
    """
    max_len = len(arg[0].get_weights())
    amount_mats = len(arg)
    solution = np.array([])
    # iterate over the depth
    for j in range(max_len):
        # iterate over all models
        for i, model in enumerate(arg):

            if j % 2 == 0:
                # the Weight matrices
                try:
                    loc_sol1 = np.insert(loc_sol1, [model.get_weights()[j]])
                except:
                    loc_sol1 = model.get_weights()[j]
                result1 = np.kron(np.eye(amount_mats, dtype=int), loc_sol1)
            else:
                # the bias vectors
                try:
                    loc_sol2 = np.insert(loc_sol2, [model.get_weights()[j]])
                except:
                    loc_sol2 = model.get_weights()[j]
                result2 = np.kron(np.eye(int(amount_mats), dtype=int), loc_sol2)
                # for some reason the bias vector gets doubled
                result2 = result2[0]
                solution = np.append(solution, [result1, result2])
    return solution


def add_matrices_general(*arg):
    max_len = len(arg[0].get_weights())
    print("max length of weights:", max_len)
    amount_mats = len(arg)
    solution = []
    for j in range(max_len):
        for i, model in enumerate(arg):
            if j % 2 == 0:
                try:
                    loc_sol1 = np.insert(loc_sol1, [model.get_weights()[j]])
                except:
                    loc_sol1 = model.get_weights()[j]
            else:
                try:
                    loc_sol1 = np.insert(loc_sol1, [model.get_weights()[j]])
                except:
                    loc_sol1 = model.get_weights()[j]

        result1 = np.kron(np.eye(amount_mats, dtype=int), loc_sol1)
        if j % 2 != 0:
            result1 = result1[0]
        try:
            solution.insert(0, result1)
        except:
            solution = result1
    solution = solution[::-1]
    return solution

# add identity after testing!

