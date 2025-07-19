# bulk_predictor.py
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
model = tf.keras.models.load_model("bulk_model.h5")

class_names = [
    'Chandramallika', 'Cosmos Phul', 'Gada', 'Golap', 'Jaba', 'Kagoj Phul',
    'Noyontara', 'Radhachura', 'Rangan', 'Salvia', 'Sandhyamani',
    'Surjomukhi', 'Zinnia'
]
IMAGE_SIZE = (156, 156)

def predict_bulk(image_b64):
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

    flower_info = {
        'Chandramallika': {
            'scientific_name': 'Chrysanthemum morifolium',
            'region': 'Widely cultivated in South Asia, especially in winter'
        },
        'Cosmos Phul': {
            'scientific_name': 'Cosmos bipinnatus',
            'region': 'Native to Mexico, but common in gardens across South Asia'
        },
        'Gada': {
            'scientific_name': 'Tagetes erecta',
            'region': 'Native to Mexico, widely grown in Bangladesh and India'
        },
        'Golap': {
            'scientific_name': 'Rosa spp.',
            'region': 'Globally cultivated, popular in Bangladesh'
        },
        'Jaba': {
            'scientific_name': 'Hibiscus rosa-sinensis',
            'region': 'Native to East Asia, very common in Bangladesh'
        },
        'Kagoj Phul': {
            'scientific_name': 'Bougainvillea glabra',
            'region': 'Native to South America, widely grown in South Asia'
        },
        'Noyontara': {
            'scientific_name': 'Catharanthus roseus',
            'region': 'Native to Madagascar, naturalized in tropical Asia'
        },
        'Radhachura': {
            'scientific_name': 'Caesalpinia pulcherrima',
            'region': 'Tropical and subtropical regions of Asia and Americas'
        },
        'Rangan': {
            'scientific_name': 'Ixora coccinea',
            'region': 'Native to Southern India and Sri Lanka'
        },
        'Salvia': {
            'scientific_name': 'Salvia splendens',
            'region': 'Native to Brazil, commonly grown in tropical gardens'
        },
        'Sandhyamani': {
            'scientific_name': 'Mirabilis jalapa',
            'region': 'Native to Peru, grown in tropical Asia as an ornamental'
        },
        'Surjomukhi': {
            'scientific_name': 'Helianthus annuus',
            'region': 'Native to North America, cultivated worldwide'
        },
        'Zinnia': {
            'scientific_name': 'Zinnia elegans',
            'region': 'Native to Mexico, popular in gardens across the world'
        }
    }


    return {
        "predicted_class": class_names[class_idx],
        "confidence": confidence,
        "sci_name": flower_info[class_names[class_idx]]['scientific_name'],
        "region": flower_info[class_names[class_idx]]['region']
    }
