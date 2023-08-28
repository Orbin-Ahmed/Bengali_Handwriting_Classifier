import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model_path = './accuracy_89.h5'
loaded_model = load_model(model_path)
class_mapping = {
    0: 'অ',
    1: 'আ',
    2: 'ই',
    3: 'ঈ',
    4: 'উ',
    5: 'ঊ',
    6: 'ঋ',
    7: 'এ',
    8: 'ঐ',
    9: 'ও',
    10: 'ঔ',
    11: 'ক',
    12: 'খ',
    13: 'গ',
    14: 'ঘ',
    15: 'ঙ',
    16: 'চ',
    17: 'ছ',
    18: 'জ',
    19: 'ঝ',
    20: 'ঞ',
    21: 'ট',
    22: 'ঠ',
    23: 'ড',
    24: 'ঢ',
    25: 'ণ',
    26: 'ত',
    27: 'থ',
    28: 'দ',
    29: 'ধ',
    30: 'ন',
    31: 'প',
    32: 'ফ',
    33: 'ব',
    34: 'ভ',
    35: 'ম',
    36: 'য',
    37: 'র',
    38: 'ল',
    39: 'শ',
    40: 'ষ',
    41: 'স',
    42: 'হ',
    43: 'ড়',
    44: 'ঢ়',
    45: 'য়',
    46: 'া',
    47: 'ি',
    48: 'ী',
    49: 'ু',
    50: 'ক্ব',
    51: 'ণ্ব',
    52: 'ধ্ন',
    53: 'স্ব',
    169: 'র-ফলা'
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


image_path = './test_Image/bcc000019.bmp'
preprocessed_image = preprocess_image(image_path)
predictions = loaded_model.predict(preprocessed_image)
top_classes = np.argsort(predictions[0])[::-1][:5]
for i, class_idx in enumerate(top_classes):
    class_name = class_mapping[class_idx]  # Replace with your class mapping
    probability = predictions[0][class_idx]
    print(f"Top {i+1}: {class_name} - Probability: {probability:.4f}")
predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)
