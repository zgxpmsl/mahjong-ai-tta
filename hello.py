<<<<<<< HEAD
#save this file as hello.py in your repo
import tensorflow as tf

# Simple hello world using TensorFlow
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
=======
#save this file as hello.py in your repo
import tensorflow as tf

# Simple hello world using TensorFlow
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
>>>>>>> 5d47f9f7ae26043ea790fd63c45df2baa17ac527
print(sess.run(hello))