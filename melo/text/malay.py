import os
from functools import cache
from transformers import AutoTokenizer
from .japanese import distribute_phone

model_id = os.environ.get('MODEL_ID', 'mesolitica/bert-base-standard-bahasa-cased')

@cache
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

@cache
def get_normalizer():
    from malaya_speech.tts import load_text_ids

    normalizer = load_text_ids(pad_to = None, understand_punct = True, is_lower = False)
    return normalizer

@cache
def get_phonemizer():
    import phonemizer
    from phonemizer.separator import Separator

    global_phonemizer = phonemizer.backend.EspeakBackend(language='ms', preserve_punctuation=True,  with_stress=True)
    separator = Separator(phone='-', word='|')

    return global_phonemizer, separator

def text_normalize(text):
    normalizer = get_normalizer()
    t, ids = normalizer.normalize(text, add_fullstop = True)
    return t

def g2p(text, pad_start_end=True, tokenized=None):
    global_phonemizer, separator = get_phonemizer()
    if tokenized is None:
        tokenizer = get_tokenizer()
        tokenized = tokenizer.tokenize(text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        r = global_phonemizer.phonemize([w], separator = separator)[0].replace('|', '')
        splitted = r.split('-')
        for s in splitted:
            if len(s):
                phones.append(s)
                if 'ˈ' in s:
                    t = 1
                else:
                    t = 0
                tones.append(t)
                phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    try:
        from text import malay_bert
    except:
        from melo.text import malay_bert

    return malay_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    text = 'hello nama saya.'
    text = text_normalize(text)
    phones, tones, word2ph = g2p(text)
    """
    (['_',
    'h',
    'ˈɛ',
    'l',
    'o',
    'n',
    'ˈa',
    'm',
    'ə',
    's',
    'ˈa',
    'j',
    'ə',
    '.',
    '_'],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 4, 4, 4, 1, 1])
    """