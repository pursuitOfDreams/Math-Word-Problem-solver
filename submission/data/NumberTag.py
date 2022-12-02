import re


class NumberTag():
    def __init__(self, sentence, equation):
        int_regex = r"\.0(\s|\D+|$)"
        self._original_sentence = self._ints(re.sub(int_regex, ' ', sentence))
        self._original_equation = self._ints(re.sub(int_regex, ' ', equation))
        self._lookup_table = {}
        self._unmasked = []

        self._tags = ['j', 'q', 'z',
                      'y', 'x', 'v',
                      'w', 'r', 'p',
                      'm', 't', 'f',
                      'c', 'b', 'd']

        self._tagged_sentence, self._tagged_equation, self._lookup_table = self._map_numbers()

        self._check()
        self._clean_output()

    def _map_numbers(self):
        # Replaces numbers in a sentence with keyed tags
        splitput = self._original_sentence.split(' ')
        spliteq = self._original_equation.split(' ')

        for i, word in enumerate(splitput):
            try:
                if "[[" in word and "]]" in word:
                    # Number that is not relevant and should not be tagged
                    n = word[2:-2]
                    splitput[i] = n
                    self._unmasked.append(n)
                else:
                    maybe_number = float(word)
                    index = len(self._lookup_table)

                    key = f"<{self._tags[index]}>"

                    self._lookup_table[key] = word
                    splitput[i] = key
            except:
                pass

        for i, word in enumerate(spliteq):
            try:
                if not word in self._lookup_table.values():
                    maybe_number = float(word)
                    index = len(self._lookup_table)

                    key = f"<{self._tags[index]}>"

                    self._lookup_table[key] = word
            except:
                pass

        adjust_dict = self._lookup_table.copy()

        for i, word in enumerate(spliteq):
            try:
                for k, v in adjust_dict.items():
                    if word == v:
                        spliteq[i] = k
                        break
            except:
                pass

        return " ".join(splitput), " ".join(spliteq), self._lookup_table

    def mapped_correctly(self):
        test = True
        for word in self._tagged_sentence.split(' '):
            try:
                x = float(word)
                if not word in self._unmasked:
                    test = False
            except:
                pass

        for word in self._tagged_equation.split(' '):
            try:
                x = float(word)
                test = False
            except:
                pass
        return test

    def _ints(self, sentence):
        # For example here, change 132.0 to 132, but leave 2.03 as is
        temp = []
        for word in sentence.split(' '):
            if word == "dozen":
                word = "12"

            try:
                temp.append(self._format(word))
            except:
                temp.append(word)

        return ' '.join(temp)

    def _format(num):
        if float(num) % 1 == 0:
            return str(int(num))
        else:
            return str(num)

    def get_originals(self):
        return self._original_sentence, self._original_equation

    def get_masked(self):
        return self._tagged_sentence, self._tagged_equation, self._lookup_table

    def _space_eliminator(self, s):
        s = re.sub(r"^\s+", '', s)
        s = re.sub(r"\s+$", '', s)
        s = re.sub(r"\s+", ' ', s)
        return s

    def _clean_output(self):
        self._tagged_sentence = self._space_eliminator(self._tagged_sentence)
        self._tagged_equation = self._space_eliminator(self._tagged_equation)

    def apply_map(self, sentence, lookup):
        splitput = sentence.split(' ')

        for i, word in enumerate(splitput):
            try:
                if word in lookup:
                    splitput[i] = lookup[word]
            except:
                pass

        return " ".join(splitput)

    def _check(self):
        rev_table = {v: k for k,
                     v in self._lookup_table.items()}

        for k, _ in rev_table.items():
            self._tagged_equation = re.sub(k, f" {k} ", self._tagged_equation)

        split = self._tagged_equation.split(' ')

        for i, word in enumerate(split):
            try:
                maybe_number = float(word)
                try_num = []
                for c in word:
                    try_num.append(c)
                    s_try_num = "".join(try_num)

                    if s_try_num in rev_table:
                        self._tagged_equation = re.sub(s_try_num,
                                                       f"{s_try_num} ",
                                                       self._tagged_equation)

                split_new = self._tagged_equation.split(' ')
                for j, word in enumerate(split_new):
                    if word in rev_table:
                        split_new[j] = rev_table[word]

                self._tagged_equation = " ".join(split_new)
            except:
                pass