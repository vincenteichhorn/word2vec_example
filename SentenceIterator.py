import re

class SentenceIterator:

    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        for line in open(self.filepath, encoding="utf-8"):
            line = line.split()
            line.pop(0)
            for index, word in enumerate(list(reversed(line))):
                i = len(line) - 1 - index
                line[i] = re.sub('[^a-zA-Z0-9]', '', word) #[^a-zA-Z0-9_äöüÄÖÜß]
                if len(line[i]) == 0:
                    line.pop(i)
            yield line