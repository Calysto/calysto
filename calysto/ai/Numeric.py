## Pure-Python Numeric replacement
## (c) 2013, Doug Blank <doug.blank@gmail.com>
## GPL, version 3.0

import math, operator, copy
import array as python_array
import random
from functools import reduce

#class python_array(parray.array):
#    def __new__(cls, *args, **kwargs):
#        return parray.array.__new__(cls, *args, **kwargs)

def ndim(n, *args, **kwargs):
    """
    Makes a multi-dimensional array of random floats. (Replaces
    RandomArray).
    """
    thunk = kwargs.get("thunk", lambda: random.random())
    if not args: 
        return [thunk() for i in range(n)]
    A = [] 
    for i in range(n):
        A.append( ndim(*args, thunk=thunk) ) 
    return A 

class array:
    """
    Replaces Numeric's ndarray.
    """
    def __init__(self, data, typecode='f'):
        ## array([1, 2])
        ## array([[1, 2], [3, 4]])
        if type(data) == array:
            self.array = data[:]
        elif type(data[0]) in [int, float, int, bool]:
            # array.array of floats
            self.array = python_array.array('d', data)
        else:
            # list of Arrays
            self.array = list(map(array, data))

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, item, value):
        self.array[item] = value

    def __len__(self):
        return len(self.array)

    def copy(self):
        if type(self.array) is list:
            return copy.deepcopy(self.array)
        else:
            return self.array[:] # vector only!

    def __repr__(self):
        if type(self.array) is list:
            return str([str(v) for v in self.array])
        else:
            return str(self.array.tolist())

    def __lt__(self, other):
        if type(self.array) is list:
            return array([v < other for v in self.array])
        return array([f < other for f in self.array])

    def __gt__(self, other):
        if type(self.array) is list:
            return array([v > other for v in self.array])
        return array([f > other for f in self.array])

    def __mul__(self, other):
        if type(other) in [int, float, int]:
            return array([v * other for v in self.array])
        else: # array * [0, 1] 
            return array(list(map(lambda a,b: a * b, self.array, other)))

    def __div__(self, other):
        if type(other) in [int, float, int]:
            return array([v / other for v in self.array])
        else:
            raise Exception("not implemented yet")
        #else: # array * [0, 1] 
        #    return array(map(lambda a,b: a / b, self.array, other))

    def __sub__(self, other):
        if type(other) in [int, float, int]:
            return array([v - other for v in self.array])
        else: # array - [0, 1]
            ##print("-", self.array, other)
            return array(list(map(lambda a,b: a - b, self.array, other)))

    def __rsub__(self, other):
        if type(other) in [int, float, int]:
            return array([other - v for v in self.array])
        else: # array - [0, 1]
            return array(list(map(lambda a,b: b - a, self.array, other)))

    def __add__(self, other):
        if type(other) in [int, float, int]:
            return array([v + other for v in self.array])
        else: # array + [0, 1]
            #print "add a", self.array
            #print "add b", other
            return array(list(map(lambda a,b: a + b, self.array, other)))

    def __pow__(self, other):
        if type(other) in [int, float, int]:
            return array([v ** other for v in self.array])
        else: # array ** [0, 1]
            return array(list(map(lambda a,b: a ** b, self.array, other)))

    def __abs__(self):
        return array([abs(v) for v in self.array])

    def getShape(self):
        if type(self.array) == list:
            return (len(self.array), len(self.array[0]))
        else:
            return (len(self.array), )

    shape = property(getShape)
    __rmul__ = __mul__
    __rgt__ = __gt__
    __rlt__ = __lt__
    __radd__ = __add__

fabs = abs
exp = math.exp
argmin = lambda vector: vector.index(min(vector))

def put(toArray, arange, fromArray):
    for i in arange:
        if type(fromArray) in [int, float, int]:
            toArray[i] = fromArray
        else:
            toArray[i] = fromArray[i]

def arange(size):
    return list(range(size))

def zeros(dims, typecode='f'):
    if type(dims) == type(1):
        dims = (dims,)
    return array(ndim(*dims, thunk=lambda: 0.0))

def ones(dims, typecode='f'):
    if type(dims) == type(1):
        dims = (dims,)
    return array(ndim(*dims, thunk=lambda: 1.0))

class add:
    @staticmethod
    def reduce(vector):
        """
        Can be a vector or matrix. If data are bool, sum Trues.
        """
        if type(vector) is list: # matrix
            return array(list(map(add.reduce, vector)))
        else:
            return sum(vector) # Numeric_array, return scalar

class multiply:
    @staticmethod
    def reduce(vector):
        """
        Can be a vector or matrix. If data are bool, sum Trues.
        """
        if type(vector) is list: # matrix
            return array(list(map(multiply.reduce, vector)))
        else:
            return reduce(operator.mul, vector) # Numeric.array, return scalar

def outerproduct(a, b):
    """Numeric.outerproduct([0.46474895,  0.46348238,  0.53923529,  0.46428344,  0.50223047], 
    [-0.16049719,  0.17086812,  0.1692107 ,  0.17433657,  0.1738235 ,
    0.17292975,  0.17553493,  0.17222987, -0.17038313,  0.17725782,
    0.18428386]) =>
         [[-0.0745909,   0.07941078,  0.07864049,  0.08102274,  0.08078429,  0.08036892,
            0.08157967,  0.08004365, -0.07918538,  0.08238038,  0.08564573]
          [-0.07438762,  0.07919436,  0.07842618,  0.08080193,  0.08056413,  0.08014989,
            0.08135735,  0.07982551, -0.07896958,  0.08215587,  0.08541232]
          [-0.08654575,  0.09213812,  0.09124438,  0.09400843,  0.09373177,  0.09324982,
            0.09465463,  0.09287243, -0.09187659,  0.09558367,  0.09937236]
          [-0.07451619,  0.07933124,  0.07856172,  0.08094158,  0.08070337,  0.08028842,
            0.08149796,  0.07996348, -0.07910606,  0.08229787,  0.08555994]
          [-0.08060658,  0.08581518,  0.08498277,  0.08755714,  0.08729946,  0.08685059,
            0.08815899,  0.08649909, -0.0855716,   0.08902428,  0.09255297]])"""
    result = zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            result[i][j] = a[i] * b[j]
    return result

def matrixmultiply(a, b):
    #print "x a:", a
    #print "x b:", b
    # a = [[0, 1], [2, 3], [4, 5]], b = [0, 1]
    if type(b[0]) in [float, int, int]:
        retval = zeros(len(a))
        for i in range(len(a)):
            for j in range(len(a[0])):
                retval[i] = retval[i] + (a[i][j] * b[j])
    else:
        retval = zeros(len(b[0]))
        for i in range(len(a)):
            retval = retval + (a[i] * b[i])
    return retval

###====================================================

def test():
    count = 0
    ocount = 0
    rcount = 0

    def compare(answer, solution, question):
        correct = True
        try:
            if type(solution[0]) == list:
                for i in range(len(solution)):
                    for j in range(len(solution[0])):
                        if abs(answer[i][j] - solution[i][j]) > .00001:
                            correct = False
            else:
                for i in range(len(solution)):
                    if abs(answer[i] - solution[i]) > .00001:
                        correct = False
        except:
            traceback.print_exc()
            correct = False
        return int(correct)

    def testexpr(expr, solution, count):
        print(("Test %s: %s" % (count, expr)))
        # Testing original Numeric versus this version
        # Define one, or both of these:
        #orig = expr.replace("Numeric.", "MyNumeric.")
        #repl = expr.replace("Numeric.", "Numeric.")
        try:
            a = eval(orig)
        except:
            traceback.print_exc()
            a = "ERROR"
        print(("   Numeric:", a))
        try:
            b = eval(repl)
        except:
            traceback.print_exc()
            b = "ERROR"
        print(("   Replace:", b))
        return compare(a, solution, expr), compare(b, solution, expr)

    for expr, result in [
        ("Numeric.array([1, 2, 3])", [1, 2, 3]),
        ("Numeric.array([1, 2, 3]) * .1", [.1, .2, .3]),
        ("Numeric.array([1, 2, 3]) + .1", [1.1, 2.1, 3.1]),
        ("Numeric.array([1, 2, 3]) - .1", [0.9, 1.9, 2.9]),
        ("1 - Numeric.array([1, 2, 3])", [0, -1, -2]),
        ("Numeric.array([1, 2, 3]) * (1 - Numeric.array([1, 2, 3]))", [0, -2, -6]),
        ("Numeric.array([1, 2, 3]) + Numeric.array([1, 2, 3])", [2, 4, 6]),
        ("Numeric.array([1, 2, 3]) - Numeric.array([1, 2, 3])", [0, 0, 0]),
        ("Numeric.array([1, 2, 3]) * Numeric.array([1, 2, 3])", [1, 4, 9]),
        ("Numeric.matrixmultiply([0, 1], Numeric.array([[1, 2, 3], [4, 5, 6]]))", [4, 5, 6]),
        ("""Numeric.matrixmultiply(Numeric.array([ 0.46474895,  0.46348238,  0.53923529,  0.46428344,  0.50223047]), 
      Numeric.array([[ 0.04412408, -0.075404  , -0.06693672, -0.02464404,  0.02612747,
         0.02212752,  0.07636238,  0.072556  , -0.06134244,  0.02546186,
        -0.01727653],
       [-0.08557264, -0.00232236,  0.07388691, -0.00192655,  0.01600296,
         0.06193477, -0.02097884,  0.04044046, -0.05679244, -0.09011306,
        -0.0977003 ],
       [-0.06398591,  0.08277569, -0.09862752, -0.06175452, -0.09487095,
         0.0585492 ,  0.00494566,  0.0130769 ,  0.02689676, -0.05318661,
         0.00723755],
       [-0.02247093,  0.08601486,  0.09500633, -0.01985373, -0.06269768,
        -0.01716621, -0.0814789 , -0.09739735, -0.03084963,  0.09573449,
         0.0864492 ],
       [-0.06170642,  0.06903436,  0.05886129,  0.04076389, -0.02530141,
        -0.0604482 , -0.02426675, -0.013736  ,  0.00506414,  0.02097294,
        -0.05462886]]))""", [-0.09508197,  0.0831217,   0.02362488, -0.03439132, -0.07341459,  0.03223229,
 -0.02158392,  0.00739668, -0.05210706, -0.00363136, -0.03670823]),
        ("""Numeric.matrixmultiply(Numeric.array([[0, 1], [2, 3], [4, 5]]), Numeric.array([0, 1]))""", [1, 3, 5]),
        ("""Numeric.outerproduct([0.46474895,  0.46348238,  0.53923529,  0.46428344,  0.50223047], 
       [-0.16049719,  0.17086812,  0.1692107 ,  0.17433657,  0.1738235 ,
        0.17292975,  0.17553493,  0.17222987, -0.17038313,  0.17725782,
        0.18428386])""", 
         [[-0.0745909,   0.07941078,  0.07864049,  0.08102274,  0.08078429,  0.08036892,
            0.08157967,  0.08004365, -0.07918538,  0.08238038,  0.08564573],
          [-0.07438762,  0.07919436,  0.07842618,  0.08080193,  0.08056413,  0.08014989,
            0.08135735,  0.07982551, -0.07896958,  0.08215587,  0.08541232],
          [-0.08654575,  0.09213812,  0.09124438,  0.09400843,  0.09373177,  0.09324982,
            0.09465463,  0.09287243, -0.09187659,  0.09558367,  0.09937236],
          [-0.07451619,  0.07933124,  0.07856172,  0.08094158,  0.08070337,  0.08028842,
            0.08149796,  0.07996348, -0.07910606,  0.08229787,  0.08555994],
          [-0.08060658,  0.08581518,  0.08498277,  0.08755714,  0.08729946,  0.08685059,
            0.08815899,  0.08649909, -0.0855716,   0.08902428,  0.09255297]]),
        ]:
        count += 1
        o, r = testexpr(expr, result, count)
        print(("-" * 60))
        print((count, o, r))
        ocount += o
        rcount += r
    print(("%s expressions tested: %s originals passed; %s replacements passed" % (count, ocount, rcount)))
