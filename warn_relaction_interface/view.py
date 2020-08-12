from django.http import HttpResponse
from . import settings

from .core import systen_unit as su
import threading
import os
import re
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import numpy as np
from tensorflow import keras
from numpy import array
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tqdm import tqdm
from tensorflow import keras
import json
import logging

logger = logging.getLogger('log')


def start(req):
    return HttpResponse(json.dumps({
        "success": True,
        "msg": "心跳检测成功..."
    }))


def check_rule(req):
    if req.method == "POST" and req.body != None:
        try:
            rule_data = json.loads(req.body)
            rules = json.loads(rule_data['resu'])
            resu_data=do_check_rule(rules)

            return HttpResponse(json.dumps({
                "code": 200,
                "msg": "no other",
                "body": {
                    "resu": json.dumps(resu_data)
                }
            }))
        except Exception as e:
            logger.error(str(e))
            return HttpResponse(json.dumps({
                "code": 500,
                "msg": "unknown exception",
                "body": {
                    "resu": ""
                }
            }))
    else:
        return HttpResponse(json.dumps({
            "code": 500,
            "msg": "only accept http post",
            "body": {
                "resu": ""
            }
        }))
def do_check_rule(data):
    all_title=[]
    z=[all_title.extend(i) for i in data]
    all_title = list(set(all_title))
    label_encoder = LabelEncoder()
    integer_encoded = to_categorical(label_encoder.fit_transform(all_title))
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    model = keras.models.load_model(os.path.join(settings.MODEL_URL,'model_20200812_104239.h'))
    x = np.c_[
        np.mat(onehot_encoded), np.mat(np.zeros([onehot_encoded.shape[0], 1000 - onehot_encoded.shape[1]]))]
    value = model.predict(x)
    new_data = []
    for key, val in enumerate([1 if i[0] >= 0.5 else 0 for i in value]):
        if val == 1:
            new_data.append(data[key])
    return new_data
