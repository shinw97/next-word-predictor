from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
# Create your views here.
import os
from pickle import load
from predictormodel.model import NextWordModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

MODEL_PATH = BASE_DIR + '/predictormodel/256-100.h5'
TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizer.pkl'

print('Loading tokenizer...')
tokenizer = load(open(TOKENIZER_PATH, 'rb'))
tokenizer.oov_token = None
print('\tDONE.')

model = NextWordModel()
model.load_model_and_tokenizer(tokenizer, MODEL_PATH)

@api_view(['GET', 'POST'])
def predict(request):
	if request.method == 'POST':
			pretext = request.data['text']
			predicted_words = model.generate_seq(50, pretext, 1)
			return Response({"predicted_words": predicted_words})
	return Response({"text": "<TEXT HERE>"})
