import re

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(t):
    '''map raw text to normalized text.'''
    t = fix_apastrophe(t)
    t = fix_star(t)
    t = remove_words(t)
    t = normalize_text(t)
    return t

def fix_apostrophe_dict():
    '''maps apostrophe words e.g. 't to full words -> het.'''
    d ={}
    d["'t"] = 'het'
    d["'n"] = 'een'
    d["d'r"] = 'der'
    d["zo'n"] = 'zo een'
    d["da's"] = 'dat is'
    d["z'n"] = 'zijn'
    d["m'n"] = 'mijn'
    d["'m"] = 'hem'
    d["'r"] = 'er'
    d["'s"] = 'is'
    d["'ns"] = 'eens'
    return d

def fix_apastrophe(t):
    '''map words as 't and 'n to het and een, remove apostrophe for plural s.'''
    words = t.split(' ')
    d = fix_apostrophe_dict()
    output = []
    for word in words:
        if word in d.keys():
            o = d[word]
        else: o = word
        output.append(o)
    output = ' '.join(output)
    return output.replace("'",'')

def fix_star(t):
    '''retains first part of a word before the *.
    for example grappi*a maps to grappi
    '''
    words = t.split(' ')
    output = []
    for word in words:
        if '*' in word:
            o = word.split('*')[0]
            if len(o) == 1: continue
        else: o = word
        output.append(o)
    return ' '.join(output)

def remove_words(t, remove = None):
    if not remove:
        remove = ['gg','ggg','gggg','xx','xxx','xxxx','um','uhm','kch']
        remove += ['mm','mmm']
    words = t.split(' ')
    output = []
    for word in words:
        if word in remove: continue
        if len(word) == 1: continue
        output.append(word)
    return ' '.join(output)
