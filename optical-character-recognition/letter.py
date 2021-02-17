class Letter(object):

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w*h


    def getArg(self):
        return self.x, self.y, self.w, self.h