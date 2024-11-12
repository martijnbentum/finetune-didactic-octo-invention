'''
Copyright 2021, Martijn Bentum, Humanities Lab, Radboud University Nijmegen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from pathlib import Path
import sys
import re
'''whether to fail if a number cannot be mapped to int'''
hard_fail_global = None

def ten_words(nd = None):
    if not nd: nd = number_dict()
    ten_words = [nd[x] for x in range(20,91,10)]
    return ten_words

def number_dict_to_number_words(nd = None):
    if not nd: nd = number_dict()
    w = list(nd.values())  
    if type(w[0]) == int: w = list(nd.keys())
    return w

def milion_multipliers(nd = None):
    number_words = number_dict_to_number_words(nd)
    return number_words[1:-3]

def thousand_multipliers(nd = None):
    number_words = number_dict_to_number_words(nd)
    return number_words[2:-3]
    
def hunderd_multipliers(nd = None):
    number_words = number_dict_to_number_words(nd)
    return number_words[2:10] + number_words[11:20]

def hunderd_addition(nd = None):
    number_words = number_dict_to_number_words(nd)
    return number_words[1:-4]

def ten_addition(nd = None):
    number_words = number_dict_to_number_words(nd)
    return number_words[1:10]
    

def map_all_numbers_to_words(text, language = 'dutch'):
    '''maps all numbers in a text to words'''
    n2w = Number2word()
    output = re.sub(r'\d+', lambda x: n2w.toword(x.group(),language), text)
    return output


def frisian_numbers():
    p = Path('../NUMBERS/frisian_numbers')
    if not p.exists(): return
    return p.open().read().split('\n')

def dutch_numbers():
    p = Path('../NUMBERS/dutch_numbers')
    if not p.exists(): return
    return p.open().read().split('\n')

def number_dict(language = 'dutch', mirror = False):
    '''a number dict contains a digit and a word column; map digits to words.'''
    if language == 'frisian':t = frisian_numbers()
    elif language == 'dutch':t=dutch_numbers()
    else: 
        print(language, 'unknown using default dutch language')
        t = dutch_numbers()
    d = dict([[int(x.split(' ')[0]),x.split(' ')[1]] for x in t if x])
    if mirror: return mirror_dict(d)
    return d
    

def mirror_dict(d):
    '''mirror the key and value of a dict'''
    return dict([[v,k] for k,v in d.items()])

class Number2word:
    '''converts a number into the word representing to number.
    by using this class the number_dict does not have to be reloaded.
    you can also use handle_number
    '''

    def __init__(self,spaces = False,hard_fail = False):
        '''spaces       whether to separate number word e.g. vier en twintig / vierentwintig
        hard_fail       whether to raise an error if a number can be converted to int
                        mainly for debugging, if false will return empty if error occurs
        '''
        global hard_fail_global
        hard_fail_global = hard_fail
        self.spaces = spaces
        self.frisian_number_dict = number_dict('frisian')
        if not self.frisian_number_dict: print('could not load frisian number dict')
        self.dutch_number_dict = number_dict(language = 'dutch')
        if not self.dutch_number_dict: print('could not load dutch number dict')
        if not self.frisian_number_dict and not self.dutch_number_dict:
            raise ValueError('could not load any language number dict')

    def toword(self,number,language = 'dutch', spaces = None):
        '''map digit number to word number'''
        if spaces != None and type(spaces) == bool: self.spaces = spaces
        if language.lower() == 'frisian':nd = self.frisian_number_dict 
        if language.lower() == 'dutch':nd = self.dutch_number_dict
        return handle_number(number,nd,spaces)

class Word2number:
    '''converts a word into a number.
    by using this class the number_dict does not have to be reloaded.
    '''

    def __init__(self,hard_fail = False):
        '''
        hard_fail       whether to raise an error if a word cannot be converted to int
                        mainly for debugging, if false will return empty if error occurs
        '''
        global hard_fail_global
        hard_fail_global = hard_fail
        self.frisian_number_dict = number_dict('frisian', mirror = True)
        if not self.frisian_number_dict: 
            print('could not load frisian number dict')
        self.dutch_number_dict = number_dict(language = 'dutch', mirror = True)
        if not self.dutch_number_dict: 
            print('could not load dutch number dict')
        if not self.frisian_number_dict and not self.dutch_number_dict:
            raise ValueError('could not load any language number dict')

    def to_number(self,word,language = 'dutch'):
        '''map digit number to word number'''
        if spaces != None and type(spaces) == bool: self.spaces = spaces
        if language.lower() == 'frisian':nd = self.frisian_number_dict 
        if language.lower() == 'dutch':nd = self.dutch_number_dict
        return handle_word(word,nd,spaces)

def handle_milion(number_words,nd, mnd):
    milion = nd[10**6]
    print('milion',milion)
    index = number_words.index(milion)
    if index == 0: 
        print('index 0',number_words)
        number = 10**6
    if index == 1:
        print('index 1',number_words)
        first_number = number_words[0]
        print('first number',first_number)
        if first_number in milion_multipliers(mnd): 
            number = mnd[first_number] * 10**6
    if index > 1:
        number, _= _handle_word(number_words[:index],nd, mnd,
            skip_milion = True, skip_thousand = True)
        print('index',index ,number_words, number)
        number = number * 10**6
    rest = len(number_words) - (index + 1)
    print('rest',rest)
    if rest == 0: return number, number_words
    extra_number, _ = _handle_word(number_words[index + 1:],nd, mnd,
        skip_milion = True)
    return number + extra_number, number_words

def handle_thousand(number_words,nd, mnd):
    thousand = nd[10**3]
    index = number_words.index(thousand)
    if index == 0: number = 10**3 
    if index == 1:
        first_number = number_words[0]
        if first_number in thousand_multipliers(mnd): 
            number = mnd[first_number] * 10**3 
    if index > 1:
        number, _= _handle_word(number_words[:index],nd, mnd,
            skip_milion = True, skip_thousand = True)
        # number, _ = handle_hunderd(number_words[:2],nd, mnd)
        number = number * 10**3
    rest = len(number_words) - (index + 1)
    print('rest',rest)
    if rest == 0: return number, number_words
    if rest == 1:
        number = number + mnd[number_words[index + 1]]
        return number, number_words
    if rest > 1:
        extra_number, _= _handle_word(number_words[index + 1:],nd, mnd,
            skip_milion = True, skip_thousand = True)
        return number + extra_number, number_words
    
def handle_hunderd(number_words,nd, mnd):
    hunderd = nd[100]
    index = number_words.index(hunderd)
    print(index)
    if index == 0: number = 100
    if index == 1:
        first_number = number_words[0]
        if first_number in hunderd_multipliers(mnd): 
            number =  mnd[first_number] * 100 
    if index == 2:
        extra_number, _= handle_ten(number_words[:2],nd, mnd)
        number = extra_number * 100
        print('extra number',extra_number, number)
    rest = len(number_words) - (index + 1)
    print('rest',rest)
    if rest == 0: return number, number_words
    if rest == 1:
        if number_words[index +1] in hunderd_addition(mnd):
            return number + mnd[number_words[index +1]], number_words
    if rest > 1:
        ten_number,_= handle_ten(number_words[index +1:],nd, mnd)
        return number + ten_number, number_words
    

def handle_ten(number_words,nd, mnd):
    tens = ten_words(nd)
    index = check_tens_in_number_words(number_words, tens)
    number_word = number_words[index]
    if index == False:
        raise ValueError('could not find tens', tens,number_words)
    if index == 0: return mnd[number_word], number_words
    if index == 1: 
        first_number = number_words[0]
        if first_number in ten_addition(mnd):
            return mnd[first_number] + mnd[number_word], number_words

def check_tens_in_number_words(number_words, tens):
    index = False
    for word in tens:
        if word in number_words:
            index = number_words.index(word)
    return index

def _handle_word(number_words, nd, mnd, skip_milion = False, 
    skip_thousand = False, skip_hundred = False, skip_ten = False):
    milion = nd[10**6]
    thousand= nd[10**3]
    hundred = nd[100]
    if not skip_milion and milion in number_words: 
        return handle_milion(number_words,nd, mnd)
    if not skip_thousand and thousand in number_words: 
        return handle_thousand(number_words,nd, mnd)
    if not skip_hundred and hundred in number_words: 
        return handle_hunderd(number_words,nd, mnd)
    tens = ten_words(nd)
    if not skip_ten and check_tens_in_number_words(number_words, tens): 
        return handle_ten(number_words,nd, mnd)
    
def handle_word(word,nd = None):
    if not nd: 
        nd = number_dict()
        mnd = number_dict(mirror = True)
    else: 
        mnd = mirror_dict(nd)
    if word.lower() in mnd.keys(): return mnd[word.lower()]
    all_number_words = number_dict_to_number_words(mnd)
    number_words = discover_word_order(word,all_number_words)
    print('number words',number_words)
    return _handle_word(number_words,nd, mnd)

def handle_number(number, nd = None,spaces =False):
    '''converts a number into the word representing to number.
    upto (not including) a billion
    '''
    if not nd: nd = number_dict()
    str_number = str(number)
    minus, number = _handle_minus(str_number)
    if type(number) == float or '.' in str(number) or ',' in str(number):
        return _handle_float(number,nd,spaces,minus)
    number = convert_number_to_int(number)
    len_number = len(str_number)
    if len_number == 1: return minus +_handle_single_digit(number,nd,spaces)
    if len_number == 2: return minus +_handle_two_digit(number,nd,spaces)
    if len_number == 3: return minus +_handle_three_digit(number,nd,spaces)
    if len_number == 4: return minus +_handle_four_digit(number,nd,spaces)
    if len_number == 5: return minus +_handle_five_digit(number,nd,spaces)
    if len_number == 6: return minus +_handle_six_digit(number,nd,spaces)
    if len_number == 7: return minus +_handle_seven_digit(number,nd,spaces)
    if len_number == 8: return minus +_handle_eight_digit(number,nd,spaces)
    if len_number == 9: return minus +_handle_nine_digit(number,nd,spaces)
    print('handles number upto length 9, return default value 42 ',
        _handle_two_digit('42',nd,spaces))
    return _handle_two_digit('42',nd,spaces)

def _handle_minus(str_number):
    '''checks whether the number start with a minus sign and prepends 
    the number word with min if it does.'''
    if len(str_number) == 0: return '',''
    if str_number[0] == '-':
        if len(str_number) == 1: return '',''
        str_number = str_number[1:]
        minus = 'min '
    else: minus = ''
    return minus , str_number

def _handle_float(number,nd = None,spaces = False, minus = ''):
    '''handles float numbers.'''
    sep = ' ' if spaces else ''
    number = str(number)
    number = number.replace(',','.')
    if not number.count('.') == 1: return handle_number(number.replace('.',''),nd,spaces)
    before_decimal, after_decimal = number.split('.')
    if before_decimal == '': before_decimal = 0
    if after_decimal == '': after_decimal = 0
    bdint = convert_number_to_int(before_decimal) 
    adint = convert_number_to_int(after_decimal) 
    if bdint == 0 and adint == 5: 
        if nd[2] == 'twa':return 'healwei'
        else: return 'half'
    elif bdint ==1 and adint == 5: 
        if nd[2] == 'twa': return 'oardel'
        else: return 'anderhalf'
    elif 0 < bdint <10 and adint == 5: 
        if nd[2] == 'twa' :return handle_number(before_decimal) + 'eninheal'
        else: return handle_number(before_decimal) + 'eneenhalf'
    before_decimal_word = handle_number(before_decimal,nd,spaces)
    after_decimal_word =  _handle_number_after_decimal(after_decimal,nd,spaces)
    return minus + before_decimal_word + ' komma ' + after_decimal_word


def _handle_number_after_decimal(after_decimal,nd,spaces):
    '''handles number after the decimal
    if there are more than two digits they are spelled out one by one
    e.g. 3.781 three comma seven eight one
    '''
    sep = ' ' if spaces else ''
    if len(after_decimal) < 3 and after_decimal[0] != '0': return handle_number(after_decimal,nd,spaces)
    output = []
    for digit in after_decimal:
        output.append( handle_number(digit) )
    return sep.join(output)

# helper function to handle digits of different lengths

def _handle_single_digit(number, nd= None,spaces = False):
    if not len(str(number)) == 1: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    number = convert_number_to_int(number)
    return nd[number]

def _handle_two_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 2: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    str_number = str(number)
    first_digit = _handle_two_digit(str_number[0] +'0',nd,spaces)
    last_digit = _handle_single_digit(str_number[-1],nd,spaces)
    if spaces: return last_digit + ' en ' + first_digit
    return last_digit + 'en' + first_digit

def _handle_three_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 3: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_two_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    if str_number[-2:] != '00':
        last_digits = _handle_two_digit(str_number[-2:],nd,spaces)
    else: last_digits = ''
    if str_number[0] != '1': 
        first_digit = _handle_single_digit(str_number[0],nd,spaces) + sep
    else: first_digit = ''
    first_digit += _handle_three_digit('100',nd,spaces)
    return first_digit + sep + last_digits

def _handle_four_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 4: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_three_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)

    if str_number[-2:] == '00': last_digits = ''
    else: last_digits = _handle_two_digit(str_number[-2:],nd,spaces)

    if str_number[1] != '0':
        first_digits = _handle_two_digit(str_number[:2],nd,spaces)
        return first_digits + sep + _handle_three_digit('100',nd,spaces)+ sep + last_digits
    if str_number[0] != '1': 
        first_digit = _handle_single_digit(str_number[0],nd,spaces) 
        first_digit += sep
    else: first_digit = ''
    first_digit += _handle_four_digit('1000',nd,spaces)
    return first_digit + sep + last_digits
        
def _handle_five_digit(number, nd = None,spaces=False):
    if not len(str(number)) == 5: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_four_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    first_digits = _handle_two_digit(str_number[:2],nd,spaces)
    if int(str_number[2:]) == 0: last_digits = ''
    else: last_digits = _handle_three_digit(str_number[2:],nd,spaces)
    return first_digits + sep + _handle_four_digit('1000',nd,spaces) + sep + last_digits

def _handle_six_digit(number, nd = None,spaces=False):
    if not len(str(number)) == 6: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_five_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    first_digits = _handle_three_digit(str_number[:3],nd,spaces)
    if int(str_number[3:]) == 0: last_digits = ''
    else: last_digits = _handle_three_digit(str_number[3:],nd,spaces)
    return first_digits + sep + _handle_four_digit('1000',nd) + sep + last_digits

def _handle_seven_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 7: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_six_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    first_digit = _handle_single_digit(str_number[0],nd,spaces)
    if int(str_number[1:]) == 0: last_digits = ''
    else: last_digits = _handle_six_digit(str_number[1:],nd,spaces)
    return first_digit + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def _handle_eight_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 8: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_seven_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    first_digits = _handle_two_digit(str_number[:2],nd,spaces)
    if int(str_number[2:]) == 0: last_digits = ''
    else: last_digits = _handle_six_digit(str_number[2:],nd,spaces)
    return first_digits + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def _handle_nine_digit(number, nd = None,spaces = False):
    if not len(str(number)) == 9: return handle_number(number,nd,spaces)
    if not nd: nd= number_dict()
    if type(number) == str and number[0] == '0': 
        return _handle_eight_digit(number[1:],nd,spaces)
    number = convert_number_to_int(number)
    if number in nd.keys(): return nd[number]
    sep = ' ' if spaces else ''
    str_number = str(number)
    first_digits = _handle_three_digit(str_number[:3],nd,spaces)
    if int(str_number[3:]) == 0: last_digits = ''
    else:last_digits = _handle_six_digit(str_number[3:],nd,spaces)
    return first_digits + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def convert_number_to_int(number, hard_fail = False):
    if hard_fail_global != None: hard_fail = hard_fail_global
    try:return int(number)
    except: 
        if hard_fail: 
            print(sys.exc_info(),number)
            raise ValueError('could not convert number to int') 
        print(sys.exc_info(),number, 'could not convert number returning empty string')
        return ''
    
    
    
def discover_word_order(text, words):
    positions = []
    
    for word in words:
        pos = -1
        while True:
            pos = text.find(word, pos+1)
            if pos == -1: break
            end = pos + len(word)
            print(text[pos+end: end+4], 'test', word)
            if text[pos + end: end+3] == 'tig': continue
            if text[pos + end: end+4] == 'tien': continue
            if word == 'tien':
                start = pos -5
                if start < 0: start = 0
                print(text[start:end], '<----')
                found = False
                for x in 'zes','zeven','acht','negen':
                    if x in text[start:end]: found = True
                if found: continue
            positions.append((pos, word))  # Store position and word

    positions.sort()
    
    ordered_words = [word for _, word in positions]
    return ordered_words
