class A():
    def __init__(self, **kw):
        func = kw["func"]
        if func == "cal_b":
            self.cal_b(**kw)
    
    @classmethod
    def cal_a(self, **kw):
        x = kw["x"]
        print(x)
    
    @staticmethod
    def cal_b(**kw):
        x = kw["x"]
        print(x)

A().cal_b(x = 10)
a = A(func = "cal_b", x = 12)