# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
import sys
import os
from os.path import dirname, join, basename, splitext

from tts import audio

import torch
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string
import train
from train import plot_alignment, build_model
from tqdm import tqdm
from io import StringIO

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later


def _tts(model, text, p=0, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def get_model(checkpoint_path):    
    max_decoder_steps = 500

    assert hparams.name == "deepvoice3"

    preset = join(dirname(__file__), 'presets/nyanko_ljspeech.json')
    with open(preset) as f:
        hparams.parse_json(f.read())

    global _frontend
    _frontend = getattr(frontend, hparams.frontend)

    train._frontend = _frontend
    # Model
    model = build_model()

    checkpoint = _load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    return model

def tts(model, text, output_file='output.wav'):
    waveform, alignment, _, _ = _tts(model, text, fast=True)

    # plot_alignment(alignment.T, file.replace('.wav', '.png'))
    audio.save_wav(waveform, output_file)

def tts_of_file(model, fname, output_wav_dir):
    with open(fname, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            tts(model, text, join(output_wav_dir, '{}.wav'))
