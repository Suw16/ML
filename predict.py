import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from utils import *
import itertools


def cnn_model_fn(features, labels, mode, params):
    """
    Model function for CNN
    :param features: images features with shape (batch_size, height, width, channels)
    :param labels: images category with shape (batch_size)
    :param mode: Specifies if this training, evaluation or
                 prediction. See `model_fn_lib.ModeKey`
    :param params: dict of hyperparameters
    :return: predictions, loss, train_op, Optional(eval_op). See `model_fn_lib.ModelFnOps`
    """
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    # Dense Layer
    pool_flat = tf.reshape(pool3, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=params['drop_out_rate']
                                , training=mode == learn.ModeKeys.TRAIN)
    
    # Logits Layer, a final layer before applying softmax
    logits = tf.layers.dense(inputs=dropout, units=17)
    
    loss = None
    train_op = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=17, name="onehot")
        #cross entropy loss
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params['learning_rate'],
            summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
            ])
    
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss, 
                                   train_op=train_op)


def feature_engineering_fn(features, labels):
    """
    feature_engineering_fn: Feature engineering function. Takes features and
                              labels which are the output of `input_fn` and
                              returns features and labels which will be fed
                              into `model_fn`
    """
    
    features = tf.to_float(features)
    
    # Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation

    # Example
    # Subtract off the mean and divide by the variance of the pixels.
    features = tf.map_fn(tf.image.per_image_standardization, features)
    
    return features, labels

run_config = RunConfig(save_summary_steps=10, keep_checkpoint_max=2, save_checkpoints_secs=30)
#drop_out_rate = 0.2, learning_rate = 0.0001
params = {'drop_out_rate': 0.2, 'learning_rate': 0.0001}
#use "model/plain_cnn" as model_dir
cnn_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="_model/plain_cnn",
        config=run_config,
        feature_engineering_fn=feature_engineering_fn, params=params)

predict_input_fn = read_img(data_dir='data/predict', batch_size=1, shuffle=False)
cnn_result = cnn_classifier.predict(input_fn=predict_input_fn)

predictions = list(itertools.islice(cnn_result, 1))
result = predictions[0]


print("result = " + str(result['classes']))
loop = result['probabilities']
c = 0
for i in loop:
    print(str(c)+" probability: " + format(i, '.8f'))
    c += 1
    