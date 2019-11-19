__charSet = None


def get_charSet():
    '''
    character class return
    but must initialize first

    Returns
    -------
    CharSet class
    '''
    if __charSet:
        return __charSet
    else:
        raise ValueError


def init_charSet(language):
    '''
    character class init with language option

    Parameters
    ----------
    language : str ('kr' or 'en')
    '''
    global __charSet
    __charSet = CharSet(language)


class CharSet:
    '''

    Attributes
    ----------
    __char_list : tuple of characters(certain language)

    __index_of_str : dict
        char : index pairs

    __char_of_index : dict
        index : char pairs

    __total_number : int
        the number of characters
    '''

    def __init__(self, language):
        '''
        1. initialize __char_list with language
        2. initialize indexes

        Parameters
        ----------
        language : str ('en' or 'kr')
        '''
        self.__char_list = (
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'J',
            'K',
            'L',
            'M',
            'N',
            'O',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'U',
            'V',
            'W',
            'X',
            'Y',
            'Z',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '0',
            '<sos>',
            '<eos>',
            '<pad>',
            '\'',
        )
        self._init_index_dict()

    def _init_index_dict(self):
        '''
        1. make index_list that has same length of __char_list

        2. combine index_list and __char_list to index char and number

        3. compute the number of characters
        '''
        index_list = [i for i in range(len(self.__char_list))]
        self.__index_of_str = dict(zip(self.__char_list, index_list))
        self.__char_of_index = dict(zip(index_list, self.__char_list))
        self.__total_num = index_list[-1] + 1

    def get_index_of(self, character):
        if character == ' ':
            character = "<pad>"
        return self.__index_of_str[character]

    def get_char_of(self, index):
        return self.__char_of_index[index]

    def get_total_num(self):
        return self.__total_num
