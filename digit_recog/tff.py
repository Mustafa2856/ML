import warnings
warnings.simplefilter('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(32,5,activation='relu')
        self.conv2 = Conv2D(32,15,activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(10)

    def call(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions= model(images)
        loss = loss_object(labels,predictions)
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels,predictions)

@tf.function
def test_step(images,labels):
    predictions = model(images)
    t_loss = loss_object(labels,predictions)
    test_loss(t_loss)
    test_accuracy(labels,predictions)

EPOCHS=10
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images,labels in train_ds:
        train_step(images,labels)
    for images,labels in test_ds:
        test_step(images,labels)
    
    print('Epoch:',epoch+1)
    print('Train loss: {}'.format(train_loss.result()),'Training Accuracy: {}'.format(train_accuracy.result()*100))
    print('Test loss: {}'.format(test_loss.result()),'Test Accuracy: {}'.format(test_accuracy.result()*100))
