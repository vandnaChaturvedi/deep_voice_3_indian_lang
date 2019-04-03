from os import listdir, path
from numpy.random import shuffle
from tqdm import tqdm
from hparams import hparams

BASE_DIR = '../data/saved_chunks/'
book_dirs = [BASE_DIR + book_dir + '/' for book_dir in listdir(BASE_DIR)]

wavfiles = sum([[book_dir + f for f in listdir(book_dir)] for book_dir in book_dirs], [])
all_samples = []

for f in tqdm(wavfiles):
	txtpath = f.replace('/saved_chunks/', '/texts/').replace('__.wav', '__.txt')
	if float(f.split('__')[1]) > 10.:
		continue
	if path.exists(txtpath):
		text = open(txtpath, encoding='utf8').read()
		if len(text) < hparams.min_text:
			continue

		sample = (f, text)
		all_samples.append(sample)

print('Total number of samples: {}'.format(len(all_samples)))

shuffle(all_samples) # in-place

train_split = len(all_samples) - 100

train_samples, test_samples = all_samples[:train_split], all_samples[train_split:]

def write_to_file(samples, split):
	with open('filelists/{}_metadata.txt'.format(split), 'w', encoding='utf-8') as f:
		for wavpath, text in samples:
			f.write('{}|{}\n'.format(wavpath, text))

write_to_file(train_samples, 'train')
write_to_file(test_samples, 'test')

print('Training files: {}, Testing files: {}'.format(len(train_samples), len(test_samples)))
