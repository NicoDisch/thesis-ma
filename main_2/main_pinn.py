import numpy as np
import yaml
import time
import tensorflow as tf
import utils_pinn
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = time.time()
    print('run PINN!')
    # read config
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config['string'])

    # load the NN model
    try:
        model = utils_pinn.load_model(config)
    except:
        model = utils_pinn.build_model(config)
    model.summary()

    # receive trained accuracy
    if config['mode']=='train':
        accs = utils_pinn.train(model,config)
        utils_pinn.write_results(model, config, accs)
    elif config['mode']=='plot':
        #print(model(0.5,0.5))
        utils_pinn.plot_decision_boundary(model, 100)
    # write the result down
    end = time.time()
    print('runtime:', end - start)


