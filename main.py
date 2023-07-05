import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import load_model
from keras.optimizers.legacy.nadam import learning_rate_schedule
from keras.optimizers import Adam


BASE_DIR = ''
WORKING_DIR = ''
CAP_DIR = ''



def find_max_length():
    with open(os.path.join(CAP_DIR, 'captions10k.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()

    # create mapping of image to captions
    mapping = {}
    # process lines
    for line in tqdm(captions_doc.split('\n')):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        # remove extension from image ID
        image_id = image_id.split('.')[0]
        # convert caption list to string
        caption = " ".join(caption)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
            # store the caption
            mapping[image_id].append(caption)


    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                # take one caption at a time
                caption = captions[i]
                # preprocessing steps
                # convert to lowercase
                caption = caption.lower()
                # delete digits, special chars, etc.,
                caption = caption.replace('[^A-Za-z]', '')
                # delete additional spaces
                caption = caption.replace('\s+', ' ')
                # add start and end tags to the caption
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
                captions[i] = caption

    clean(mapping)

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    # tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1

    # get maximum length of the caption available
    max_length = max(len(caption.split()) for caption in all_captions)

    return max_length,tokenizer




def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break
        in_text += " " + word

        if word == 'endseq':
            break

    return in_text


def predict(image_path):
    image_path = os.path.join("static",image_path)
    img = Image.open(image_path)
    # plt.imshow(img)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    feature = vgg_model.predict(image, verbose=0)


    # predict from the trained model
    max_length,tokenizer = find_max_length()
    model = load_model("best_model10k005.h5")
    pred = predict_caption(model, feature, tokenizer, max_length)
    test_pred = pred.replace("startseq", "").replace("endseq", "")
    print("---------------------Generated caption---------------------\n",test_pred)
    print("---------------------Actual image---------------------")

    return test_pred


# image_path = "images (1).jpeg"
# predict(image_path)