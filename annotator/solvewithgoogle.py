from os import listdir, path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import codecs, sys, pdb

from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from multiprocessing import cpu_count

## Imports from Google Speech code:
import io, os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

BASE_DIR = '/home/vandna/wav/books/chunks/' #for al
book_dirs = [BASE_DIR + book_dir + '/' for book_dir in listdir(BASE_DIR)]
files = sum([[book_dir + f for f in listdir(book_dir)] for book_dir in book_dirs], [])
sorted_files_with_lens = sorted([(float(f.split('__')[1]), f) for f in files], reverse=True)  

clip_at = 10.

while sorted_files_with_lens[0][0] > clip_at:
	sorted_files_with_lens = sorted_files_with_lens[1:]

assert sorted_files_with_lens[0][0] <= clip_at # no file should be longer than 1 min

print('Beginning transcription of {} files, will skip already finished files'.format(len(sorted_files_with_lens)))
print('Files contain {} hours of audio'.format(sum([c for c, f in sorted_files_with_lens]) / 3600))


# Instantiates a client
client = speech.SpeechClient()

sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

files = [f for c, f in sorted_files_with_lens]

def recognize(file_name):
	output_filename = file_name.replace('chunks', 'texts').replace('__.wav', '__.txt')
	print(output_filename)
	if path.exists(output_filename):
		return

	try:
		# Loads the audio into memory
		with io.open(file_name, 'rb') as audio_file:
			content = audio_file.read()
			audio = types.RecognitionAudio(content=content)

		config = types.RecognitionConfig(
					encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
					sample_rate_hertz=48000,
					language_code='hi-IN')

		# Detects speech in the audio file
		response = client.recognize(config, audio)

		for result in response.results:
			text = result.alternatives[0].transcript
			with open(output_filename, 'wb':q) as out:
				out.write(text.encode('utf8'))
	#			out.write(text)
	except Exception as e:
		print(e)
		with open(output_filename.replace('/texts/', '/failures/'), 'w') as out:
			out.write(e)

pool = ThreadPoolExecutor(4)
futures = [pool.submit(recognize, file) for file in files]
_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
