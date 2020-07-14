import re

import numpy as np
# import tensorflow as tf
import tensorflow_hub as hub
from GoogleNews import GoogleNews
from newspaper import Article
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack, vstack
from tensorflow import keras

global embed, feat_vec, feat_svec


def initialize():
    # Allowing use of Multiple GPU's
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Create 2 virtual GPUs with 1GB memory each
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
    #              tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    global embed
    embed = hub.load(module_url)

    global feat_vec, feat_svec
    feat_vec = np.empty((1025))
    feat_vec[:] = np.nan
    feat_svec = csr_matrix(feat_vec)
    feat_svec, feat_svec.toarray()


def pre_process(text):
    lst_stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
                      "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                      "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
                      "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
                      "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
                      "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor",
                      "of", "on", "once", "only", "or", "other",
                      "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
                      "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                      "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                      "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                      "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                      "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd",
                      "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    punctuations = '''!()-[\’\“\”]{—};:'"\,<>./?@#$%^&*_~'''
    ps = PorterStemmer()
    re.sub(r'http\S+', '', text)
    word_lst = text.lower().split()
    new_word_lst = []
    for word in word_lst:
        mod_word = ''.join([char for char in word if char not in punctuations])
        mod_word = ps.stem(mod_word)
        new_word_lst.append(mod_word)
        if mod_word in lst_stop_words:
            new_word_lst.remove(mod_word)
    mod_text = ' '.join(new_word_lst)

    return mod_text


def get_user_data(user_link):
    article = Article(user_link, language='en')
    article.download()
    article.parse()
    article.nlp()
    headline = article.title
    content = article.summary
    image = article.top_image
    user_d = {'link': user_link, 'headline': headline, 'content': content, 'image': image}
    return user_d


def get_admin_data(user_headline, user_img):
    admin_data = {'link': None, 'headline': None, 'content': None, 'image': None}
    google_news = GoogleNews(lang='en')
    google_news.search(user_headline)
    links = google_news.get__links()
    print('No. of links found: ', len(links))
    if len(links) == 0:
        return admin_data
    elif len(links) == 1:
        link_used = links[0]
    else:
        link_used = links[1]

    admin_data['link'] = link_used
    print(link_used)
    article = Article(link_used)
    article.download()
    article.parse()
    article.nlp()
    admin_data['headline'] = article.title
    admin_data['content'] = article.summary
    if article.top_image is None:
        admin_data['image'] = user_img
    else:
        admin_data['image'] = article.top_image

    return admin_data


def get_feature_vector(user_data, admin_data):
    user_emb = embed([user_data])
    admin_emb = embed([admin_data])
    sim = np.inner(user_emb, admin_emb)
    global feat_vec, feat_svec
    feat_svec = vstack(
        (feat_svec, hstack((admin_emb[0], sim[0], user_emb[0]))))
    feat_svec = feat_svec.tocsr()
    feat_svec = feat_svec[1:]
    feat_vec = feat_svec.toarray()
    print(feat_vec.shape)
    print('Vector:', feat_vec)
    print('similarity:', sim[0])
    return feat_vec, sim[0][0]


def predict(feature_vec):
    model_keras = keras.models.load_model('model_1.keras')
    meaning = {0: 'Agree', 1: 'Disagree', 2: 'Discuss', 3: 'Unrelated'}
    label = np.argmax(model_keras.predict(feature_vec), axis=-1)
    print(label[0])
    result = meaning[label[0]]
    print('Result:', result)
    return result


def build(user_link):
    # Getting Headline and article from User's Link
    user_data = get_user_data(user_link)

    # Pre-Processing user data
    user_data['content'] = pre_process(user_data['content'])

    # Getting our own data
    admin_data = get_admin_data(user_data['headline'], user_data['image'])

    # If no related data is found
    if admin_data['link'] is None:
        r = {'result': 'Possibly_Fake', 'similarity': 0, 'user_data': user_data, 'admin_data': admin_data}
        return r

    # If data available
    # Pre-Processing our data
    admin_data['content'] = pre_process(admin_data['content'])

    # Getting Feature Vector
    features, sim = get_feature_vector(user_data['content'], admin_data['content'])

    # Prediction on Feature Vector
    result = predict(features)
    r = {'result': result, 'similarity': sim * 100, 'user_data': user_data, 'admin_data': admin_data}
    return r

    # time.sleep(1)
