import os
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
import cv2

from scipy.cluster.hierarchy import dendrogram, linkage
from keras.models import load_model
from tensorflow.python.framework import ops

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        model_file = 'my_model.h5'
        new_model = load_model(model_file)
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(model, input, category_index, layer_name):
    print(layer_name)

    input_image = input[0]

    inp = model.input
    y_c = model.output.op.inputs[0][0, category_index]
    A_k = model.get_layer(layer_name).output

    print(y_c, A_k)

    get_output = K.function(
        [inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output(input)
    print(model_output)

    conv_output = conv_output[0]
    grad_val = grad_val[0]

    print(conv_output.shape, grad_val.shape)

    weights = np.mean(grad_val, axis=(0, 1))

    print(weights.shape)

    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    print(np.percentile(grad_cam, [0, 25, 50, 75, 100]))
    heatmap = grad_cam = np.maximum(grad_cam, 0)
    print(np.percentile(grad_cam, [0, 25, 50, 75, 100]))

    image = input_image.squeeze()
    print(image.shape, heatmap.shape)
    image -= np.min(image)
    image = 255 * image / np.max(image)
    image1 = cv2.resize(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB), (240, 1460),
                        interpolation=cv2.INTER_NEAREST)

    print(image1.shape)
    #heatmap1 = cv2.resize(heatmap, (240,1460))
    print(np.percentile(heatmap/np.max(heatmap), [0, 25, 50, 75, 100]))
    heatmap1 = cv2.resize(heatmap/np.max(heatmap),
                          (240, 1460), interpolation=cv2.INTER_LINEAR)

    grad_cam = cv2.applyColorMap(np.uint8(255*heatmap1), cv2.COLORMAP_JET)
    grad_cam = np.float32(grad_cam) + np.float32(image1)
    grad_cam = 255 * grad_cam / np.max(grad_cam)
    return np.uint8(grad_cam), heatmap


pca_exp = pd.read_excel("data/PCA_EXP.xlsx", header=None)
pca_cnv = pd.read_excel("data/PCA_CNV.xlsx", header=None)
pca_mt = pd.read_excel("data/PCA_MT.xlsx", header=None)
clinical = pd.read_excel("data/Clinical.xlsx")

# data size: 146 pathways, 5 PCs
n = len(pca_exp)  # sample size: number of Pts
path_n = 146  # number of pathways
pc = 5  # number of PCs

# data creation-EXP
pca_exp = pca_exp.to_numpy()
print(pca_exp.shape)
exp_data = np.zeros((n, path_n, pc))
for i in range(n):
    for j in range(path_n):
        exp_data[i, j, :] = pca_exp[i, j * pc:(j + 1) * pc]

# data creation-CNV
pca_cnv = pca_cnv.to_numpy()
cnv_data = np.zeros((n, path_n, pc))
for i in range(n):
    for j in range(path_n):
        cnv_data[i, j, :] = pca_cnv[i, j * pc:(j + 1) * pc]

# data creation-MT
pca_mt = pca_mt.to_numpy()
mt_data = np.zeros((n, path_n, pc))
for i in range(n):
    for j in range(path_n):
        mt_data[i, j, :] = pca_mt[i, j * pc:(j + 1) * pc]

# data merge: mRNA expression, CNV, and MT with a specific number of PCs
no_pc = 2  # use the first 2 PCs among 5 PCs
all_data = np.zeros((n, path_n, no_pc * 3))
for i in range(n):
    all_data[i, :, :] = np.concatenate((exp_data[i, :, 0:no_pc], cnv_data[i, :, 0:no_pc], mt_data[i, :, 0:no_pc]),
                                       axis=1)

index = np.arange(0, pca_exp.shape[1], pc)

clinical = clinical.to_numpy()
survival = clinical[:, 5]
os_months = clinical[:, 6]
idx0 = np.where((survival == 1) & (os_months <= 24))  # class0
idx1 = np.where(os_months > 24)                       # class1

all_data = all_data[:, :, :]

data_0 = all_data[idx0, :, :]
data_0 = data_0[0, :, :, :]
data_1 = all_data[idx1, :, :]
data_1 = data_1[0, :, :, :]

outcomes_0 = np.zeros(len(idx0[0]))
outcomes_1 = np.ones(len(idx1[0]))

# data merge
data = np.concatenate((data_0, data_1))
outcomes = np.concatenate((outcomes_0, outcomes_1))

print('test')
event_rate = np.array(
    [outcomes_0.shape[0], outcomes_1.shape[0]])/outcomes_0.shape[0]
print(event_rate)

vmin = -10
vmax = 10
m = 1

model_file = 'PathCNN_model.h5'

model = load_model(model_file)
layers = model.layers
print(model.summary())

orig_weights = model.get_weights().copy()

output_layer = model.get_layer('dense_2')
output_weights = output_layer.get_weights()
print(output_weights[1])

col = ['outcome', 'prediction', 'target_class',
       'probability']+list(range(1, no_pc*path_n*3+1))
print(col)

layer = layers[4]  # the last layer before flatten
print(layer.name)
weights = layer.get_weights()

os.makedirs('output/'+layer.name, exist_ok=True)
for i in range(0, data.shape[0], 1):
    oimg = data[i, :, :]

    print(f'\ncase {i:03d}_{outcomes[i]:.0f}')
    print(oimg.shape)

    img = ((np.clip(oimg, vmin, vmax) - vmin) /
           (vmax - vmin) * 255).astype('uint8')

    timg = oimg.reshape(1, oimg.shape[0], oimg.shape[1], 1)

    preprocessed_input = [timg]

    model.set_weights(orig_weights)
    predictions = model.predict(preprocessed_input)
    print('Predicted class:')
    print(predictions)
    predicted_class = np.argmax(predictions)
    print(predicted_class)

    for target_class in range(2):
        gradcam, heatmap = grad_cam(
            model, preprocessed_input, target_class, layer.name)
        filename = f'output/{layer.name}/gradcam{i:03d}_{outcomes[i]:.0f}_{predicted_class:d}_{target_class:d}_{predictions[0][target_class]:0.2f}.png'
        cv2.imwrite(filename, gradcam)

        heatmap1 = cv2.resize(heatmap, (no_pc*3, path_n))
        print(heatmap.shape)
        data_row = [[outcomes[i], predicted_class, target_class,
                     predictions[0][target_class]]+heatmap1.flatten().tolist()]
        gradcam_pd = pd.DataFrame(data_row, columns=col, index=[i])
        if i == 0 and target_class == 0:
            gradcam_pd.to_csv(
                f'output/{layer.name}/pathcnn_gradcam.csv', mode='w')
        else:
            gradcam_pd.to_csv(f'output/{layer.name}/pathcnn_gradcam.csv',
                              mode='a', header=False)
