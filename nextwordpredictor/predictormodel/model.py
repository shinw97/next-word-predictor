from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class NextWordModel(object):
	"""docstring for NextWordModel"""
	def __init__(self):
		super(NextWordModel, self).__init__()
	
	def load_model_and_tokenizer(self, tokenizer, model_path):
		self.tokenizer = tokenizer
		print('Loading model...')
		self.model = load_model(model_path)
		self.model._make_predict_function()

	def generate_seq(self, seq_length, seed_text, n_words):
		result = list()
		in_text = seed_text
		# generate a fixed number of words
		for _ in range(n_words):
			# encode the text as integer
			encoded = self.tokenizer.texts_to_sequences([in_text])[0]
			# truncate sequences to a fixed length
			encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
			# predict probabilities for each word
			yhat = self.model.predict_classes(encoded, verbose=0)
			# map predicted word index to word
			out_word = ''
			for word, index in self.tokenizer.word_index.items():
				if index == yhat:
					out_word = word
					break
			# append to input
			in_text += ' ' + out_word
			result.append(out_word)
		return ' '.join(result)