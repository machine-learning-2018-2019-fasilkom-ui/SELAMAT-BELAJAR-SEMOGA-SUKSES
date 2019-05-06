
class StopwordRemover():

    def __init__(self):
        self.stoplist = list()
        with open('assets/stoplist.txt', 'r') as f:
            for line in f.readlines():
                self.stoplist.append(line.strip('\n'))

    def remove_stopwords(self, text):
        return ' '.join(token for token in text.split(' ') if token not in self.stoplist)
