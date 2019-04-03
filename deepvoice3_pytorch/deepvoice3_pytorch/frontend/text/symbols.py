'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from .cmudict import valid_symbols

_pad = '_'
_eos = '~'
# _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list("d-i$3+t&7 6qf=Aw%RH5sCzxMhaBe8G.n9jp',2rgmvklc1o4J:uXN0*y/PTb")
