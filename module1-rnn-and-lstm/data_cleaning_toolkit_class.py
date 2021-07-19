import re
import numpy as np


class data_cleaning_toolkit(object):
    
    def __init__(self):
        """
        This class host several data cleaning and preperation methods that are used to prepare text data 
        for a text generation model, specifically for the LSTM.
        """
        
        # TODO: provide descriptions of each variable
        self.sequences = None
        self.next_char = None
        self.n_features = None
        self.unique_chars = None
        self.maxlen = None
        self.char_int = None
        self.int_char = None
        self.sequences = None
        self.next_char = None
    
    def clean_data(self, doc):
        """
        Accepts a single text document and performs several regex substitutions in order to clean the document. 

        Note
        ----
        Don't forget about online regex editors such as this one -  https://regex101.com/

        Parameters
        ----------
        doc: string or object 

        Returns
        -------
        doc: string or object
        """

        # order of operations - apply the expression from top to bottom
        date_regex = r"\d+/\d+/\d+" # remove dates in the format 00/00/0000
        punct_regex = r"[^0-9a-zA-Z\s]" # any non-alphanumeric chars
        special_chars_regex = r"[\$\%\&\@\n+]" # any speical chars
        numerical_regex = r"\d+" # any remianing digits 
        multiple_whitespace = " {2,}" # any 2 or more consecutive white spaces (don't strip single white spaces!)

        doc = re.sub(date_regex, "", doc)
        doc = re.sub(punct_regex, "", doc)
        doc = re.sub(special_chars_regex, " ", doc)
        doc = re.sub(numerical_regex, "", doc)
        doc = re.sub(multiple_whitespace, "", doc)

        # apply case normalization 
        return doc.lower()   
    
    def create_char_sequences(self, data, maxlen = 20, step = 5):
        """
        Creates numerically encoded text sequences for model input and encoded chars 
        for what the model should predict next. 
        
        This method needs to be used prior to calling def create_X_and_Y()
        
        Parameters
        ----------
        data: list of strings
            This is our list of documents
            
        maxlen: int 
            This is the maximum length for the numerically encoded documents
            
        step: int
            Determines how many characters to skip before picking a starting index 
            to generate the next input sequence. 
            
            Example
            -------
            If the sequence is "I love big and fluffy dogs!"
            Then maxlen = 6 step = 5 would chop up the following sequences 
            
            "I love", "ve big", " and f", "fluffy", and so on ... 
            
            Notice that <maxlen> is the size of char seqeunce
            Notice that <step> is the starting index for creating the next char sequence
            
        Returns
        -------
        None
        """
        
        # this valueof maxlen will be used in def create_X_and_Y() method
        self.maxlen = maxlen
        
        # Encode Data as Chars

        # join all text data into a single string 
        text = " ".join(data)

        # get unique characters
        self.unique_chars = list(set(text))
        
        # our text gen model will treat every unique char as a possible feature to predict
        self.n_features = len(self.unique_chars)

        # Lookup Tables
        # keys are chars
        # vals are integers
        self.char_int = {c:i for i, c in enumerate(self.unique_chars)}

        # keys are integers
        # vals are chars
        self.int_char = {i:c for i, c in enumerate(self.unique_chars)} 

        # we will encore our text by taking a character and representing it by 
        # the index that we have assigned to it in our char_int dictionary 
        # we are transforming natural language into a numerical representation (similar to countvectorizer and tfidf) 
        encoded = [self.char_int[char] for char in text]

        total_num_chars_in_text = len(encoded)

        sequences = [] # Each sequence in this list is maxlen chars long
        next_char = [] # One element for each sequence

        for i in range(0, total_num_chars_in_text - maxlen, step):

            # input sequence
            sequences.append(encoded[i : i + maxlen])
            # the very next char that a model should predict will follow the input sequence
            next_char.append(encoded[i + maxlen])

        # we know we have this many samples 
        print('Created {0} sequences.'.format(len(sequences)))
        
        self.sequences = sequences
        self.next_char = next_char

    def create_X_and_Y(self):
        """
        Takes a sequence of chars and creates an input/output split (i.e. X and Y)
        
        Paremeters
        ----------
        None
        
        Returns
        -------
        x: array of Booleans (i.e. True and False)
        y: array of Booleans (i.e. True and False)
        """
        # this is the number of rows in the doc-term matrix that we are about to create (i.e. x) 
        n_seqs = len(self.sequences)
        
        # this is the number of features in the doc-term matrix that we are about to create 
        n_unique_chars = len(self.unique_chars) 
        
        # Create shape for x and y 
        x_dims = (len(self.sequences), self.maxlen, len(self.unique_chars))
        y_dims = (len(self.sequences),len(self.unique_chars))

        # create data containers for x and y 
        # default values will all be zero ( i.e. look up docs for np.zeros() )
        # recall that a value of zero is equivalent to False in Python 
        x = np.zeros(x_dims, dtype=np.bool)
        y = np.zeros(y_dims, dtype=np.bool)

        # populate x and y with 1 (from a Boolean perspective, 1 and True are the same thing)
        # iterative through the index and sequence
        for i, sequence in enumerate(self.sequences):
            # take tha sequence and iterate through the chars in the sequence 
            for t, char in enumerate(sequence):
                # for row i, location in time series t, and feature char
                # assign a value of 1 
                # recall we are using encoded chars from def create_char_sequenes()
                # meaning characters are now represented by a numerical value 
                x[i,t,char] = 1

            # follow similar for the char that should be predicted by the model 
            # given the corresponding sequence of chars in x
            y[i, self.next_char[i]] = 1

        return x, y