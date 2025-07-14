# single_predictor.py
import os
import numpy as np
import tensorflow as tf
import base64
import cv2

# ✅ Limit TensorFlow threading for shared hosting (like cPanel)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

# Load model
model = tf.keras.models.load_model("single_model.h5")

class_names = [
    'Chandramallika', 'Cosmos Phul', 'Gada', 'Golap', 'Jaba', 'Kagoj Phul',
    'Noyontara', 'Radhachura', 'Rangan', 'Salvia', 'Sandhyamani',
    'Surjomukhi', 'Zinnia'
]
IMAGE_SIZE = (156, 156)

def predict_single(image_b64):
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])

    return {
        "predicted_class": class_names[class_idx],
        "confidence": confidence
    }
