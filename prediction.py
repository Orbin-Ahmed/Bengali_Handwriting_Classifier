from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

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
    54: 'গ্ধ',
    55: 'গ্ন',
    56: 'স্ল',
    57: 'গ্ব',
    58: 'গ্ম',
    59: 'হ্ণ',
    60: 'হ্ব',
    61: 'গ্ল',
    62: 'হ্ম',
    63: 'ঙ্ক',
    64: 'ঙ্ঘ',
    65: 'ঙ্ক্ষ',
    66: 'চ্ছ্ব',
    67: 'হ্ল',
    68: 'জ্জ্ব',
    69: 'ল্ক',
    70: 'ল্ট',
    71: 'স্প',
    72: 'ল্ত',
    73: 'ঞ্ছ',
    74: 'স্ত্ব',
    75: 'স্ট',
    76: 'স্খ',
    77: 'ণ্ট',
    78: 'ণ্ড',
    79: 'ণ্ঢ',
    80: 'ণ্ণ',
    81: 'ণ্ম',
    82: 'ষ্ম',
    83: 'ত্ত্ব',
    84: 'ত্ব',
    85: 'ত্ম',
    86: 'স্ত্র',
    87: 'ষ্ব',
    88: 'ত্র',
    89: 'ষ্ফ',
    90: 'থ্ব',
    91: 'স্ক্র',
    92: 'শ্ল',
    93: 'দ্গ',
    94: 'দ্ঘ',
    95: 'দ্দ্ব',
    96: 'দ্ভ্র',
    97: 'দ্ম',
    98: 'ষ্ক',
    99: 'ষ্ক্র',
    100: 'ক্ষ্ণ',
    101: 'ধ্ব',
    102: 'জ্ঝ',
    103: 'ন্ট',
    104: 'ন্ট্র',
    105: 'ন্ঠ',
    106: 'শ্ম',
    107: 'ন্ত্ব',
    108: 'ক্ম',
    109: 'ক্ষ্ম',
    110: 'ন্থ',
    111: 'ঙ্ম',
    112: 'ন্দ্ব',
    113: 'চ্ঞ',
    114: 'ড়্গ',
    115: 'শ্ব',
    116: 'ন্ব',
    117: 'ধ্ম',
    118: 'প্ট',
    119: 'প্ন',
    120: 'শ্ন',
    121: 'শ্ছ',
    122: 'ল্ল',
    123: 'প্স',
    124: 'ল্ব',
    125: 'ফ্ল',
    126: 'ব্জ',
    127: 'ব্ধ',
    128: 'ব্ব',
    129: 'ল্ম',
    130: 'ল্ড',
    131: 'ব্ল',
    132: 'ল্গ',
    133: 'প্ল',
    134: 'ম্ন',
    135: 'প্প',
    136: 'ম্ভ',
    137: 'ম্ভ্র',
    138: 'ম্ম',
    139: 'ম্ফ',
    140: 'ট্ব',
    141: 'ম্ল',
    142: 'ঞ্ঝ',
    143: 'ট্র',
    144: 'প্র',
    145: 'গ্র',
    146: 'জ্র',
    147: 'ণ্ড্র',
    148: 'দ্র',
    149: 'ন্ড্র',
    150: 'ন্ত্র',
    151: 'ন্দ্র',
    152: 'ম্প্র',
    153: 'ম্র',
    154: 'ভ্র',
    155: 'র্গ',
    156: ' জ্র',
    157: 'র্চ',
    158: 'থ্র',
    159: 'হ্র',
    160: 'র্থ',
    161: 'র্ব্য',
    162: 'র্শ',
    163: ' শ্রু',
    164: 'র্ছ',
    165: 'র্ত',
    166: 'র্থ',
    167: 'রেফ',
    168: 'য-ফলা',
    169: 'র-ফলা'
}

model_path = './accuracy_86.h5'
loaded_model = load_model(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28, 3))
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


image_path = './Dataset/Balanced_train/2/bcc000000.bmp'
preprocessed_image = preprocess_image(image_path)
predictions = loaded_model.predict(preprocessed_image)
top_classes = np.argsort(predictions[0])[::-1][:5]
for i, class_idx in enumerate(top_classes):
    class_name = class_mapping[class_idx]
    probability = predictions[0][class_idx]
    print(f"Top {i+1}: {class_name} - Probability: {probability:.4f}")
predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)