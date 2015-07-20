from __future__ import print_function

"""
----------------------------------------------------
An Artificial Neural Network System Implementing
Backprop. Part of the Pyrobot Robotics Project.
Provided under the GNU General Public License.
----------------------------------------------------
(c) 2001-2013, Developmental Robotics Research Group
----------------------------------------------------

This file implements the major classes and functions for
making artificial neural networks in Python.
"""

__author__ = "Douglas Blank <dblank@brynmawr.edu>"
__version__ = "$Revision$"

# Includes newConx version 4/24/2009
# Jim Marshall

import math, random, time, sys, operator, copy, array, traceback
from functools import reduce
try:
    import numpy.oldnumeric as Numeric
except:
    from . import Numeric ## pure-python version

def reverse(lyst):
    """ Returns a reversed list. """
    return [lyst[p] for p in range(len(lyst) - 1, -1, -1)]

def pad(s, n, p = " ", sep = "|", align = "left"):
    """
    Returns a padded string.
    s = string to pad
    n = width of string to return
    sep = separator (on end of string)
    align = text alignment, "left", "center", or "right"
    """
    if align == "left":
        return (s + (p * n))[:n] + sep
    elif align == "center":
        pos = n + len(s)/2 - n/2
        return ((p * n) + s + (p * n))[pos:pos + n] + sep
    elif align == "right":
        return ((p * n) + s)[-n:] + sep

def sumMerge(dict1, dict2):
    """
    Adds two dictionaries together, and merges into the first, dict1.
    Returns first dict.
    """
    for key in dict2:
        dict1[key] = list(map(lambda a,b: a + b, dict1.get(key, [0,0,0,0]), dict2[key]))
    return dict1 # and also returns it, in case you want to do something to it

def loadNetwork(filename, mode = 'pickle'):
    """
    Loads network from a file using pickle. See Network.saveNetwork()
    """
    return loadNetworkFromFile(filename, mode)

def loadNetworkFromFile(filename, mode = 'pickle'):
    """
    Deprecated. Use loadNetwork instead.
    """
    if mode == 'pickle':
        import pickle
        fp = open(filename)
        network = pickle.load(fp)
        fp.close()
        return network
    elif mode in ['plain', 'conx']:
        fp = open(filename, "r")
        line = fp.readline()
        network = None
        while line:
            if line.startswith("layer,"):
                # layer, name, size
                temp, name, sizeStr = line.split(",")
                name = name.strip()
                size = int(sizeStr)
                network.addLayer(name, size)
                line = fp.readline()
                weights = [float(f) for f in line.split()]
                for i in range(network[name].size):
                    network[name].weight[i] = weights[i]
            elif line.startswith("connection,"):
                # connection, fromLayer, toLayer
                temp, nameFrom, nameTo = line.split(",")
                nameFrom, nameTo = nameFrom.strip(), nameTo.strip()
                network.connect(nameFrom, nameTo)
                for i in range(network[nameFrom].size):
                    line = fp.readline()
                    weights = [float(f) for f in line.split()]
                    for j in range(network[nameTo].size):
                        network[nameFrom, nameTo].weight[i][j] = weights[j]
            elif line.startswith("parameter,"):
                temp, exp = line.split(",")
                exec(exp) # network is the neural network object
            elif line.startswith("network,"):
                temp, netType = line.split(",")
                netType = netType.strip().lower()
                if netType == "cascornetwork":
                    from pyrobot.brain.cascor import CascorNetwork
                    network = CascorNetwork()
                elif netType == "network":
                    network = Network()
                elif netType == "srn":
                    network = SRN()
                else:
                    raise AttributeError("unknown network type: '%s'" % netType)
            line = fp.readline()
        return network

def ndim(n, *args, **kwargs):
    """
    Makes a multi-dimensional array of random floats. (Replaces RandomArray).
    """
    thunk = kwargs.get("thunk", lambda: random.random())
    if not args: 
        return [thunk() for i in range(n)]
    A = [] 
    for i in range(n):
        A.append( ndim(*args, thunk=thunk) ) 
    return A 

def randomArray2(size, bound):
    """
    Returns an array initialized to random values between -bound and
    bound distributed in a gaussian probability distribution more
    appropriate for a Tanh activation function.
    """
    if type(size) == type(1):
        size = (size,)
    temp = Numeric.array( ndim(*size), thunk=lambda: random.gauss(0, 1)) * (2.0 * bound)
    return temp - bound

def randomArray(size, bound):
    """
    Returns an array initialized to random values between -max and max.
    """
    if type(size) == type(1):
        size = (size,)
    temp = Numeric.array( ndim(*size) ) * (2.0 * bound)
    return temp - bound

def displayArray(name, a, width = 0):
    """
    Prints an array (any sequence of floats, really) to the screen.
    """
    print(name + ": ", end=" ")
    cnt = 0
    for i in a:
        print("%4.2f" % i, end=" ")
        if width > 0 and (cnt + 1) % width == 0:
            print('')
        cnt += 1

def toStringArray(name, a, width = 0):
    """
    Returns an array (any sequence of floats, really) as a string.
    """
    string = name + ": "
    cnt = 0
    for i in a:
        string += "%4.2f  " % i 
        if width > 0 and (cnt + 1) % width == 0:
            string += '\n'
        cnt += 1
    return string

def writeArray(fp, a, delim = " ", nl = 1):
    """
    Writes a sequence a of floats to file pointed to by file pointer.
    """
    for i in a:
        fp.write("%f%s" % (i, delim))
    if nl:
        fp.write("\n")

class LayerError(AttributeError):
    """
    Used to indicate that a layer has some improper attribute (size,
    type, etc.).
    """
    
class NetworkError(AttributeError):
    """
    Used to indicate that a network has some improper attribute (no
    layers, no connections, etc.).
    """

class SRNError(NetworkError):
    """
    Used to indicate that SRN specific attributes are improper.
    """

class Node:
    """
    A temp place to hold values for reference. If given a layer and position, then
    the node can be used to set values, and can update itself.
    """
    def __init__(self, layer = None, position = None, **keywords):
        self.__dict__["_layer"] = layer
        self.__dict__["_position"] = position
        self.__dict__["_attributes"] = list(keywords.keys())
        for key in keywords:
            self.__dict__[key] = keywords[key]
    def __repr__(self):
        return str(self)
    def __str__(self):
        retval = "Node:"
        if self._layer != None:
            retval += "\n   layer = '%s'" % self._layer.name
            retval += "\n   position = %d" % self._position
        for key in self._attributes:
            retval += "\n   %s = %s" % (key, self.__dict__[key])
        return retval
    def update(self):
        if self._layer != None:
            for key in self._attributes:
                self.__dict__[key] = self._layer.__dict__[key][self._position]
        else:
            raise AttributeError("node is read-only")
    def __setattr__(self, attribute, value):
        if self._layer != None:
            if attribute in self._layer.__dict__:
                self._layer.__dict__[attribute][self._position] = value
            else:
                raise AttributeError("no such property '%s'" % attribute)
        else:
            raise AttributeError("node is read-only: use net['layerName'].%s[NUM] = %s" % (attribute, value))

class Layer:
    """
    Class which contains arrays of node elements (ie, activation,
    error, bias, etc).
    """
    # constructor
    def __init__(self, name, size, maxRandom = 0.1):
        """
        Constructor for Layer class. A name and the number of nodes
        for the instance are passed as arguments.
        """
        if size <= 0:
            raise LayerError('Layer was initialized with size zero.' , size)
        self.patternReport = 0 # flag to determine if layer should be in report
        # patternReport is set to 1 automatically for output layers (in .connect())
        self.name = name
        self.size = size
        self.displayWidth = size
        self.type = 'Undefined' # determined later by connectivity
        self.kind = 'Undefined' # mirrors self.type but will include 'Context'
        self._verbosity = 0
        self.log = 0
        self.logFile = ''
        self._logPtr = 0
        self._namePtr = 0
        self.active = 1 # takes layer out of the processing
        self.frozen = 0 # freezes weights (dbiases), if layer still active
        self._maxRandom = maxRandom
        self.initialize()
        
    def initialize(self):
        """
        Initializes important node values to zero for each node in the
        layer (target, error, activation, dbias, delta, netinput, bed).
        """
        self.randomize()
        self.dweight = Numeric.zeros(self.size, 'f')
        self.delta = Numeric.zeros(self.size, 'f')
        self.wed = Numeric.zeros(self.size, 'f')
        self.wedLast = Numeric.zeros(self.size, 'f')
        self.target = Numeric.zeros(self.size, 'f')
        self.error = Numeric.zeros(self.size, 'f')
        self.activation = Numeric.zeros(self.size, 'f')
        self.netinput = Numeric.zeros(self.size, 'f')
        self.targetSet = 0
        self.activationSet = 0
        self.verify = 1
        # layer report of stats:
        self.pcorrect = 0
        self.ptotal = 0
        self.correct = 0
        # misc:
        self.minTarget = 0.0
        self.maxTarget = 1.0
        self.minActivation = 0.0
        self.maxActivation = 1.0

    def randomize(self, force = 0):
        """
        Initialize node biases to random values in the range [-max, max].
        """
        if force or not self.frozen:
            self.weight = randomArray(self.size, self._maxRandom)

    # general methods
    def __len__(self):
        """
        Returns the number of nodes in the layer.
        """
        return self.size
    def __repr__(self):
        return str(self)
    def __str__(self):
        return self.toString()
    def __getitem__(self, i):
        """ Return a temp object with some properties set """
        if type(i) == int:
            return Node(
                layer = self,
                position = i,
                activation = self.activation[i],
                error = self.error[i],
                target = self.target[i],
                netinput = self.netinput[i],
                weight = self.weight[i]
                )
        else:
            raise AttributeError("expected integer instead of '%s'" % i)
    # layer methods
    def setActive(self, value):
        """
        Sets layer to active or inactive. Layers must be active to propagate activations.
        """
        self.active = value
    def getActive(self):
        """
        Used to determine if a layer is active or inactive.
        """
        return self.active
    def changeSize(self, newsize):
        """
        Changes the size of the layer. Should only be called through
        Network.changeLayerSize().
        """
        # overwrites current data
        if newsize <= 0:
            raise LayerError('Layer size changed to zero.', newsize)
        minSize = min(self.size, newsize)
        bias = randomArray(newsize, self._maxRandom)
        Numeric.put(bias, Numeric.arange(minSize), self.weight)
        self.weight = bias
        self.size = newsize
        self.displayWidth = newsize
        self.targetSet = 0
        self.activationSet = 0
        self.target = Numeric.zeros(self.size, 'f')
        self.error = Numeric.zeros(self.size, 'f')
        self.activation = Numeric.zeros(self.size, 'f')
        self.dweight = Numeric.zeros(self.size, 'f')
        self.delta = Numeric.zeros(self.size, 'f')
        self.netinput = Numeric.zeros(self.size, 'f')
        self.wed = Numeric.zeros(self.size, 'f')
        self.wedLast = Numeric.zeros(self.size, 'f')

    # error and report methods
    def TSSError(self):
        """
        Returns Total Sum Squared Error for this layer's pattern.
        """
        return Numeric.add.reduce((self.target - self.activation) ** 2)
    def RMSError(self):
        """
        Returns Root Mean Squared Error for this layer's pattern.
        """
        tss = self.TSSError()
        return math.sqrt(tss / self.size)
    def getCorrect(self, tolerance):
        """
        Returns the number of nodes within tolerance of the target.
        """
        return Numeric.add.reduce(Numeric.fabs(self.target - self.activation) < tolerance)
    def getWinner(self, type = 'activation'):
        """
        Returns the winner of the type specified {'activation' or
        'target'}.
        """
        maxvalue = -10000
        maxpos = -1
        ttlvalue = 0
        if type == 'activation':
            ttlvalue = Numeric.add.reduce(self.activation)
            maxpos = Numeric.argmax(self.activation)
            maxvalue = self.activation[maxpos]
        elif type == 'target':
            # note that backprop() resets self.targetSet flag
            if self.verify and self.targetSet == 0:
                raise LayerError('getWinner() called with \'target\' but target has not been set.', \
                       self.targetSet)
            ttlvalue = Numeric.add.reduce(self.target)
            maxpos = Numeric.argmax(self.target)
            maxvalue = self.target[maxpos]
        else:
            raise LayerError('getWinner() called with unknown layer attribute.', \
                               type)
        if self.size > 0:
            avgvalue = ttlvalue / float(self.size)
        else:
            raise LayerError('getWinner() called for layer of size zero.', \
                               self.size)
        return maxpos, maxvalue, avgvalue

    # used so pickle will work with log file pointer
    def __getstate__(self):
        odict = self.__dict__.copy() 
        del odict['_logPtr']
        return odict
    def __setstate__(self,dict):
        if dict['log']:
            self._logPtr = open(dict['logFile'], 'a') 
        else:
            self._logPtr = 0
        self.__dict__.update(dict)
        
    # log methods
    def setLog(self, fileName, writeName=False):
        """
        Opens a log file with name fileName.
        """
        self.log = 1
        self.logFile = fileName
        self._logPtr = open(fileName, "w")
        if writeName:
            self._namePtr = open(fileName + ".name", "w")
    def logMsg(self, msg):
        """
        Logs a message.
        """
        self._logPtr.write(msg)
    def closeLog(self):
        """
        Closes the log file.
        """
        self._logPtr.close()
        if self._namePtr:
            self._namePtr.close()
        self.log = 0
    def writeLog(self, network):
        """
        Writes to the log file.
        """
        if self.log:
            writeArray(self._logPtr, self.activation)
            if self._namePtr:
                self._namePtr.write(network.getWord(self.activation))
                self._namePtr.write("\n")

    # string and display methods
    def setDisplayWidth(self, val):
        """
        Sets self.displayWidth the the argument value.
        """
        self.displayWidth = val
    def toString(self):
        """
        Returns a string representation of Layer instance.
        """
        string = "Layer '%s': (Kind: %s, Size: %d, Active: %d, Frozen: %d)\n" % (
            self.name, self.kind, self.size, self.active, self.frozen) 
        if (self.type == 'Output'):
            string += toStringArray('Target    ', self.target, self.displayWidth)
        string += toStringArray('Activation', self.activation, self.displayWidth) 
        if (self.type != 'Input' and self._verbosity > 1):
            string += toStringArray('Error     ', self.error, self.displayWidth)
        if (self._verbosity > 4 and self.type != 'Input'):
            string +=  toStringArray('weight    ', self.weight, self.displayWidth)
            string +=  toStringArray('dweight   ', self.dweight, self.displayWidth)
            string +=  toStringArray('delta     ', self.delta, self.displayWidth)
            string +=  toStringArray('netinput  ', self.netinput, self.displayWidth)
            string +=  toStringArray('wed       ', self.wed, self.displayWidth)
        return string 
    def display(self):
        """
        Displays the Layer instance to the screen.
        """
        if self.displayWidth == 0: return
        print("=============================")
        print("Layer '%s': (Kind: %s, Size: %d, Active: %d, Frozen: %d)" % (
            self.name, self.kind, self.size, self.active, self.frozen))
        if (self.type == 'Output'):
            displayArray('Target    ', self.target, self.displayWidth)
        displayArray('Activation', self.activation, self.displayWidth)
        if (self.type != 'Input' and self._verbosity > 1):
            displayArray('Error     ', self.error, self.displayWidth)
        if (self._verbosity > 4 and self.type != 'Input'):
            print("    ", end=" "); displayArray('weight', self.weight)
            print("    ", end=" "); displayArray('dweight', self.dweight)
            print("    ", end=" "); displayArray('delta', self.delta)
            print("    ", end=" "); displayArray('netinput', self.netinput)
            print("    ", end=" "); displayArray('wed', self.wed)

    # activation methods
    def getActivationsList(self):
        """
        Returns node activations in list (copy) form.
        """
        return self.activation.copy()
    def getActivations(self):
        """
        Returns node activations in (Numeric) array (pointer) form.
        """
        return self.activation
    def setActivations(self, value):
        """
        Sets all activations to the value of the argument. Value should be in the range [0,1].
        """
        #if self.verify and not self.activationSet == 0:
        #    raise LayerError, \
        #          ('Activation flag not reset. Activations may have been set multiple times without any intervening call to propagate().', self.activationSet)
        Numeric.put(self.activation, Numeric.arange(len(self.activation)), value)
        self.activationSet = 1
    def copyActivations(self, arr, reckless = 0):
        """
        Copies activations from the argument array into
        layer activations.
        """
        array = Numeric.array(arr)
        if not len(array) == self.size:
            raise LayerError('Mismatched activation size and layer size in call to copyActivations()', \
                   (len(array), self.size))
        if self.verify and not self.activationSet == 0:
            if not reckless:
                raise LayerError('Activation flag not reset before call to copyActivations()', \
                       self.activationSet) 
        self.activation = array
        self.activationSet = 1

    # target methods
    def getTargetsList(self):
        """
        Returns targets in list form.
        """
        return self.target.copy()
    def getTargets(self):
        """
        Return targets in (Numeric) array form.
        """
        return self.target
    def setTargets(self, value):
        """
        Sets all targets the the value of the argument. This value must be in the range [min,max].
        """
        # Removed this because both propagate and backprop (via compute_error) set targets
        #if self.verify and not self.targetSet == 0:
        #    if not self.warningIssued:
        #        print 'Warning! Targets have already been set and no intervening backprop() was called.', \
        #              (self.name, self.targetSet)
        #        print "(Warning will not be issued again)"
        #        self.warningIssued = 1
        if value > self.maxActivation or value < self.minActivation:
            raise LayerError('Targets for this layer are out of the proper interval.', (self.name, value))
        Numeric.put(self.target, Numeric.arange(len(self.target)), value)
        self.targetSet = 1
    def copyTargets(self, arr):
        """
        Copies the targets of the argument array into the self.target attribute.
        """
        array = Numeric.array(arr)
        if not len(array) == self.size:
            raise LayerError('Mismatched target size and layer size in call to copyTargets()', \
                   (len(array), self.size))
        # Removed this because both propagate and backprop (via compute_error) set targets
        #if self.verify and not self.targetSet == 0:
        #    if not self.warningIssued:
        #        print 'Warning! Targets have already been set and no intervening backprop() was called.', \
        #              (self.name, self.targetSet)
        #        print "(Warning will not be issued again)"
        #        self.warningIssued = 1
        if Numeric.add.reduce(array < self.minTarget) or Numeric.add.reduce(array > self.maxTarget):
            print(self.name, self.minTarget, self.maxTarget)
            raise LayerError('Targets for this layer are out of range.', (self.name, array))
        self.target = array
        self.targetSet = 1

    # flag methods
    def resetFlags(self):
        """
        Resets self.targetSet and self.activationSet flags.
        """
        self.targetSet = 0
        self.activationSet = 0
    def resetTargetFlag(self):
        """
        Resets self.targetSet flag.
        """
        self.targetSet = 0
    def resetActivationFlag(self):
        """
        Resets self.activationSet flag.
        """
        self.activationSet = 0

class Connection:
    """
    Class which contains references to two layers (from and to) and the
    weights between them.
    """
    # constructor and initialization methods
    def __init__(self, fromLayer, toLayer):
        """
        Constructor for Connection class. Takes instances of source and
        destination layers as arguments.
        """
        self.active = 1 # propagates and backrops if active
        self.frozen = 0 # freezes weights
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.initialize()
    def __getitem__(self, i):
        return self.weight[i]
    def initialize(self):
        """
        Initializes self.dweight and self.wed to zero matrices.
        """
        self.randomize()
        self.dweight = Numeric.zeros((self.fromLayer.size, \
                                      self.toLayer.size), 'f')
        self.wed = Numeric.zeros((self.fromLayer.size, \
                                  self.toLayer.size), 'f')
        self.wedLast = Numeric.zeros((self.fromLayer.size, \
                                      self.toLayer.size), 'f')
    def randomize(self, force = 0):
        """
        Sets weights to initial random values in the range [-max, max].
        """
        if force or not self.frozen:
            self.weight = randomArray((self.fromLayer.size, \
                                       self.toLayer.size),
                                      self.toLayer._maxRandom)
    def __str__(self):
        return self.toString()
    
    def __repr__(self):
        return str(self)

    # connection modification methods
    def changeSize(self, fromLayerSize, toLayerSize):
        """
        Changes the size of the connection depending on the size
        change of either source or destination layer. Should only be
        called through Network.changeLayerSize().
        """
        if toLayerSize <= 0 or fromLayerSize <= 0:
            raise LayerError('changeSize() called with invalid layer size.', \
                               (fromLayerSize, toLayerSize))
        dweight = Numeric.zeros((fromLayerSize, toLayerSize), 'f')
        wed = Numeric.zeros((fromLayerSize, toLayerSize), 'f')
        wedLast = Numeric.zeros((fromLayerSize, toLayerSize), 'f')
        weight = randomArray((fromLayerSize, toLayerSize),
                             self.toLayer._maxRandom)
        # copy from old to new, considering one is smaller
        minFromLayerSize = min( fromLayerSize, self.fromLayer.size)
        minToLayerSize = min( toLayerSize, self.toLayer.size)
        for i in range(minFromLayerSize):
            for j in range(minToLayerSize):
                wed[i][j] = self.wed[i][j]
                wedLast[i][j] = self.wedLast[i][j]
                dweight[i][j] = self.dweight[i][j]
                weight[i][j] = self.weight[i][j]
        self.dweight = dweight
        self.wed = wed
        self.wedLast = wedLast
        self.weight = weight

    # display methods
    def display(self):
        """
        Displays connection information to the screen.
        """
        if self.toLayer._verbosity > 4:
            print("wed: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'")
            for j in range(self.toLayer.size):
                print(self.toLayer.name, "[", j, "]", end=" ")
            print('')
            for i in range(self.fromLayer.size):
                print(self.fromLayer.name, "[", i, "]", ": ", end=" ")
                for j in range(self.toLayer.size):
                    print(self.wed[i][j], end=" ")
                print('')
            print('')
            print("dweight: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'")
            for j in range(self.toLayer.size):
                print(self.toLayer.name, "[", j, "]", end=" ")
            print('')
            for i in range(self.fromLayer.size):
                print(self.fromLayer.name, "[", i, "]", ": ", end=" ")
                for j in range(self.toLayer.size):
                    print(self.dweight[i][j], end=" ")
                print('')
            print('')
        if self.toLayer._verbosity > 2:
            print("Weights: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'")
            print("             ", end=" ")
            for j in range(self.toLayer.size):
                print(self.toLayer.name, "[", j, "]", end=" ")
            print('')
            for i in range(self.fromLayer.size):
                print(self.fromLayer.name, "[", i, "]", ": ", end=" ")
                for j in range(self.toLayer.size):
                    print(self.weight[i][j], end=" ")
                print('')
            print('')
    # string method
    def toString(self):
        """
        Connection information as a string.
        """
        string = ""
        if self.toLayer._verbosity > 4:
            string += "wed: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'\n" 
            string += "            "
            for j in range(self.toLayer.size):
                string += "        " + self.toLayer.name + "[" + str(j) + "]"
            string += '\n'
            for i in range(self.fromLayer.size):
                string += self.fromLayer.name+ "["+ str(i)+ "]"+ ": "
                for j in range(self.toLayer.size):
                    string += "              " + str(self.wed[i][j])
                string += '\n'
            string += '\n'
            string += "dweight: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'\n"
            string += "            "
            for j in range(self.toLayer.size):
                string += "        " + self.toLayer.name+ "["+ str(j)+ "]"
            string += '\n'
            for i in range(self.fromLayer.size):
                string += self.fromLayer.name+ "["+ str(i)+ "]"+ ": "
                for j in range(self.toLayer.size):
                    string +=  "              " + str(self.dweight[i][j])
                string += '\n'
            string += '\n'
        if self.toLayer._verbosity > 2:
            string += "Weights: from '" + self.fromLayer.name + "' to '" + self.toLayer.name +"'\n"
            string += "            "
            for j in range(self.toLayer.size):
                string += "        "  + self.toLayer.name+ "["+ str(j)+ "]"
            string += '\n'
            for i in range(self.fromLayer.size):
                string += self.fromLayer.name+ "["+ str(i)+ "]"+ ": "
                for j in range(self.toLayer.size):
                    string +=  "    " + str(self.weight[i][j])
                string += '\n'
            string += '\n'
        return string
    
class Network(object):
    """
    Class which contains all of the parameters and methods needed to
    run a neural network.
    """
    # constructor
    def __init__(self, name = 'Backprop Network', verbosity = 0,
                 seed = None):
        """
        Constructor for the Network class. Takes optional name and
        verbosity arguments.
        """
        if seed == None:
            x = random.random() * 100000 + time.time()
        else:
            x = seed
        self.layers = []
        self.verbosity = verbosity
        self.setSeed(x)
        self.complete = 0
        self.name = name
        self.layersByName = {}
        self.connections = []
        self.inputMap = []
        self.targetMap = []
        self.association = []
        self.inputs = []
        self.targets = []
        self.orderedInputs = 0
        self.loadOrder = []
        self.learning = 1
        self.momentum = 0.9
        self.resetEpoch = 5000
        self.resetCount = 1
        self.resetLimit = 1
        self.batch = 0
        self.epoch = 0
        self.totalEpoch = 0
        self.count = 0 # number of times propagate is called
        self.stopPercent = 1.0
        self.sigmoid_prime_offset = 0.1
        self.tolerance = 0.4
        self.interactive = 0
        self.epsilon = 0.1
        self.reportRate = 25
        self.sweepReportRate = None #1000
        self.crossValidationCorpus = ()
        self.crossValidationReportLayers = []
        self.crossValidationSampleRate = 0
        self.crossValidationSampleFile = "sample.cv"
        self.patterns = {}
        self.patterned = 0 # used for file IO with inputs and targets
        self.sharedWeights = 0
        self.useCrossValidationToStop = 0
        self.saveResults = 0 # will save error, correct, total in sweep()
        self.results = []
        self.autoCrossValidation = 0
        self.autoSaveWeightsFile = None
        self.autoSaveWeightsFileFormat = "conx"
        self.lastAutoSaveWeightsFilename = None
        self.autoSaveNetworkFile = None
        self.autoSaveNetworkFileFormat = "conx"
        self.lastAutoSaveNetworkFilename = None
        self.lastLowestTSSError = sys.maxsize # some maximum value (not all pythons have Infinity)
        self._cv = False # set true when in cross validation
        self._sweeping = 0 # flag set when sweeping through corpus (as apposed to just stepping)
        self._maxRandom = 0.1
        self.currentSweepCount = None
        self.log = None # a pointer to a file-like object, like a Log object
        self.echo = False   # if going to a log file, echo it too, if true
        self.hyperbolicError = 0 # exaggerate error?
        # Quickprop settings:
        self._quickprop = 0
        self.mu = 1.75 # maximum growth factor
        self.splitEpsilon = 0
        self.decay = 0.0000
        self.cacheConnections = []
        self.cacheLayers = []
        self.setup()
    def setCache(self, val = 1):
        """ Sets cache on (or updates), or turns off """
        # first clear the old cached values
        self.cacheConnections = []
        self.cacheLayers = []
        if val:
            for layer in self.layers:
                if layer.active and not layer.frozen:
                    self.cacheLayers.append( layer )
            for connection in self.connections:
                if connection.active and not connection.frozen:
                    self.cacheConnections.append( connection )
    def setQuickprop(self, value):
        if value:
            self.batch = 1
            self._quickprop = 1
            self.mu = 1.75 # maximum growth factor
            self.splitEpsilon = 1
            self.decay = -0.0001
            self.epsilon = 4.0
            #set different activation function here?
            self.name = "Quickprop Network"
        else:
            self._quickprop = 0
            self.splitEpsilon = 0
            self.decay = 0.0000
            self.epsilon = 0.1
    def getQuickprop(self): return self._quickprop
    def setup(self):
        pass
    # general methods
    def path(self, startLayer, endLayer):
        """
        Used in error checking with verifyArchitecture() and in prop_from().
        """
        next = {startLayer.name : startLayer}
        visited = {}
        while next != {}:
            for item in list(next.items()):
                # item[0] : name, item[1] : layer reference
                # add layer to visited dict and del from next 
                visited[item[0]] = item[1]
                del next[item[0]]
                for connection in self.connections:
                    if connection.fromLayer.name == item[0]:
                        if connection.toLayer.name == endLayer.name:
                            return 1 # a path!
                        elif connection.toLayer.name in next:
                            pass # already in the list to be traversed
                        elif connection.toLayer.name in visited:
                            pass # already been there
                        else:
                            # add to next
                            next[connection.toLayer.name] = connection.toLayer
        return 0 # didn't find it and ran out of places to go
    def __str__(self):
        """
        Returns string representation of network.
        """
        return self.toString()
    def __repr__(self):
        """
        Returns string representation of network.
        """
        return str(self)
    def __iter__(self):
        for layer in self.layers:
            yield layer
    def __getitem__(self, name):
        """
        Returns the layer specified by name.
        """
        if type(name) == str:
            return self.layersByName[name]
        else:
            fromName, toName = name
            return self.getConnection(fromName, toName)
    def __len__(self):
        """
        Returns the number of layers in the network.
        """
        return len(self.layers)
    def getLayerIndex(self, layer):
        """
        Given a reference to a layer, returns the index of that layer in
        self.layers.
        """              
        for i in range(len(self.layers)):
            if layer == self.layers[i]: # shallow cmp
                return i
        return -1 # not in list
    def addLayer(self, name, size, verbosity = 0, position = None):
        assert type(name) == str, "first parameter (name) must be a string"
        assert type(size) == int, "second parameter (size) must be an integer"
        assert (name not in self.layersByName), ("duplicate layer name '%s' is not allowed" % name)
        layer = Layer(name, size, maxRandom=self._maxRandom)
        Network.add(self, layer, verbosity, position)
    # methods for constructing and modifying a network
    def add(self, layer, verbosity = 0, position = None):
        """
        Adds a layer. Layer verbosity is optional (default 0).
        """
        layer._verbosity = verbosity
        layer._maxRandom = self._maxRandom
        layer.minTarget = 0.0
        layer.maxTarget = 1.0
        layer.minActivation = 0.0
        layer.maxActivation = 1.0
        if position == None:
            self.layers.append(layer)
        else:
            self.layers.insert(position, layer)
        self.layersByName[layer.name] = layer
    def isConnected(self, fromName, toName):
        """ Are these two layers connected this way? """
        for c in self.connections:
            if (c.fromLayer.name == fromName and
                c.toLayer.name == toName):
                return 1
        return 0
    def connect(self, *names):
        """
        Connects a list of names, one to the next.
        """
        fromName, toName, rest = names[0], names[1], names[2:]
        self.connectAt(fromName, toName)
        if len(rest) != 0:
            self.connect(toName, *rest)
    def connectAt(self, fromName, toName, position = None):
        """
        Connects two layers by instantiating an instance of Connection
        class. Allows a position number, indicating the ordering of
        the connection.
        """
        fromLayer = self.getLayer(fromName)
        toLayer   = self.getLayer(toName)
        if self.getLayerIndex(fromLayer) >= self.getLayerIndex(toLayer):
            raise NetworkError('Layers out of order.', (fromLayer.name, toLayer.name))
        if (fromLayer.type == 'Output'):
            fromLayer.type = 'Hidden'
            fromLayer.patternReport = 0 # automatically turned off for hidden layers
            if fromLayer.kind == 'Output':
                fromLayer.kind = 'Hidden'
        elif (fromLayer.type == 'Undefined'):
            fromLayer.type = 'Input'
            fromLayer.patternReport = 0 # automatically turned off for input layers
            if fromLayer.kind == 'Undefined':
                fromLayer.kind = 'Input'
        if (toLayer.type == 'Input'):
            raise NetworkError('Connections out of order', (fromLayer.name, toLayer.name))
        elif (toLayer.type == 'Undefined'):
            toLayer.type = 'Output'
            toLayer.patternReport = 1 # automatically turned on for output layers
            if toLayer.kind == 'Undefined':
                toLayer.kind = 'Output'
        if position == None:
            self.connections.append(Connection(fromLayer, toLayer))
        else:
            self.connections.insert(position, Connection(fromLayer, toLayer))
    def addThreeLayers(self, inc, hidc, outc):
        """
        Creates a three layer network with 'input', 'hidden', and
        'output' layers.
        """
        self.addLayer('input', inc)
        self.addLayer('hidden', hidc)
        self.addLayer('output', outc)
        self.connect('input', 'hidden')
        self.connect('hidden', 'output')
    def addLayers(self, *arg, **kw):
        """
        Creates an N layer network with 'input', 'hidden1', 'hidden2',...
        and 'output' layers. Keyword type indicates "parallel" or
        "serial". If only one hidden layer, it is called "hidden".
        """
        netType = "serial"
        if "type" in kw:
            netType = kw["type"]
        self.addLayer('input', arg[0])
        hiddens = []
        if len(arg) > 3:
            hcount = 0
            for hidc in arg[1:-1]:
                name = 'hidden%d' % hcount
                self.addLayer(name, hidc)
                hiddens.append(name)
                hcount += 1
        elif len(arg) == 3:
                name = 'hidden'
                self.addLayer(name, arg[1])
                hiddens.append(name)
        elif len(arg) == 2:
            pass
        else:
            raise AttributeError("not enough layers! need >= 2")
        self.addLayer('output', arg[-1])
        lastName = "input"
        for name in hiddens:
            if netType == "parallel":
                self.connect('input', name)
                self.connect(name, 'output')
            else: # serial
                self.connect(lastName, name)
                lastName = name
        if netType == "serial" or lastName == "input":
            self.connect(lastName, "output")
    def setLayerVerification(self, value):
        for layer in self.layers:
            layer.verify = value
    def deleteLayerNode(self, layername, nodeNum):
        """
        Removes a particular unit/node from a layer.
        """
        # first, construct an array of all of the weights
        # that won't be deleted:
        gene = []
        for layer in self.layers:
            if layer.type != 'Input':
                for i in range(layer.size):
                    if layer.name == layername and i == nodeNum:
                        pass # skip it
                    else:
                        gene.append(layer.weight[i])
        for connection in self.connections:
            for i in range(connection.fromLayer.size):
                for j in range(connection.toLayer.size):
                    if ((connection.fromLayer.name == layername and i == nodeNum) or
                        (connection.toLayer.name == layername and j == nodeNum)):
                        pass # skip weights from/to nodeNum
                    else:
                        gene.append(connection.weight[i][j])
        # now, change the size (removes rightmost node):
        self.changeLayerSize(layername, self[layername].size - 1)
        # and put the good weights where they go:
        self.unArrayify(gene)
    def addLayerNode(self, layerName, bias = None, weights = {}):
        """
        Adds a new node to a layer, and puts in new weights. Adds node on the end.
        Weights will be random, unless specified.

        bias    = the new node's bias weight
        weights = dict of {connectedLayerName: [weights], ...}

        Example:
        >>> net = Network() # doctest: +ELLIPSIS
        Conx using seed: ...
        >>> net.addLayers(2, 5, 1)
        >>> net.addLayerNode("hidden", bias = -0.12, weights = {"input": [1, 0], "output": [0]})
        """
        self.changeLayerSize(layerName, self[layerName].size + 1)
        if bias != None:
            self[layerName].weight[-1] = bias
        for name in list(weights.keys()):
            for c in self.connections:
                if c.fromLayer.name == name and c.toLayer.name == layerName:
                    for i in range(self[name].size):
                        self[name, layerName].weight[i][-1] = weights[name][i]
                elif c.toLayer.name == name and c.fromLayer.name == layerName:
                    for j in range(self[name].size):
                        self[layerName, name].weight[-1][j] = weights[name][j]
    def changeLayerSize(self, layername, newsize):
        """
        Changes layer size. Newsize must be greater than zero.
        """
        # for all connection from to this layer, change matrix:
        if self.sharedWeights:
            raise AttributeError("shared weights broken")
        for connection in self.connections:
            if connection.fromLayer.name == layername:
                connection.changeSize(  newsize, connection.toLayer.size )
            if connection.toLayer.name == layername:
                connection.changeSize( connection.fromLayer.size, newsize )
        # then, change the actual layer size:
        self.getLayer(layername).changeSize(newsize)
    # reset and intialization
    def reset(self):
        """
        Resets seed values.
        """
        random.seed(self.seed)
        self.initialize()
    def initialize(self):
        """
        Initializes network by calling Connection.initialize() and
        Layer.initialize(). self.count is set to zero.
        """
        print("Initializing '%s' weights..." % self.name, end=" ", file=sys.stderr)
        if self.sharedWeights:
            raise AttributeError("shared weights broken")
        self.count = 0
        for connection in self.connections:
            connection.initialize()
        for layer in self.layers:
            layer.initialize()
    def resetFlags(self):
        """
        Resets layer flags for activation and target.
        """
        for layer in self.layers:
            layer.resetFlags()

    # set and get methods for attributes
    def putActivations(self, dict):
        """
        Puts a dict of name: activations into their respective layers.
        """
        for name in dict:
            self.layersByName[name].copyActivations( dict[name] )
    def getActivationsDict(self, nameList):
        """
        Returns a dictionary of layer names that map to a list of activations.
        """
        retval = {}
        for name in nameList:
            retval[name] = self.layersByName[name].getActivationsList()
        return retval
    def getLayer(self, name):
        """
        Returns the layer with the argument (string) name.
        """
        return self.layersByName[name]
    def setAutoCrossValidation(self, value):
        self.autoCrossValidation = value
    def setAutoSaveWeightsFile(self, filename, format = "conx"):
        self.autoSaveWeightsFile = filename
        self.autoSaveWeightsFileFormat = format
    def setAutoSaveNetworkFile(self, filename, format = "conx"):
        self.autoSaveNetworkFile = filename
        self.autoSaveNetworkFileFormat = format
    def setPatterned(self, value):
        """
        Sets the network to use patterns for inputs and targets.
        """
        self.patterned = value
    def setEpsilon(self, value): self.epsilon = value
    def setInteractive(self, value):
        """
        Sets interactive to value. Specifies if an interactive prompt
        should accompany sweep() or step().
        """
        self.interactive = value
    def setTolerance(self, value):
        """
        Sets tolerance to value. This specifies how close acceptable
        outputs must be to targets.
        """
        self.tolerance = value
    def setActive(self, layerName, value):
        """
        Sets a layer to active. Affects ce_init(), compute_error(),
        propagate(), change_weights(), display().
        """
        self.getLayer(layerName).setActive(value)
    def getActive(self, layerName):
        """
        Returns the value of the active flag for the layer specified
        by layerName.
        """
        return self.getLayer(layerName).getActive()
    def setLearning(self, value):
        """
        Sets learning to value. Determines if the network learns,
        ie. changes connection weights.
        """
        self.learning = value
    def setMomentum(self, value):
        """
        Sets self.momentum to value. Used in change_weights().
        """
        self.momentum = value
    def setResetLimit(self, value):
        """
        Sets self.resetLimit to value. When the number of resets
        reaches the reset limit the network quits.
        """
        self.resetLimit = value
    def setResetEpoch(self, value):
        """
        Sets self.resetEpoch to value. When the number of epochs
        reaches self.resetEpoch the network restarts.
        """
        self.resetEpoch = value
    def setBatch(self, value):
        """
        Sets self.batch to value. Accumulates error over the entire
        training set and only changes weights after each training
        batch is complete.
        """
        self.batch = value
    def setSeed(self, value):
        """
        Sets the seed to value.
        """
        self.seed = value
        random.seed(self.seed)
        if self.verbosity >= 0:
            print("Conx using seed:", self.seed)
    def getConnection(self, lfrom, lto):
        """
        Returns the connection instance connecting the specified (string)
        layer names.
        """
        for connection in self.connections:
            if connection.fromLayer.name == lfrom and \
               connection.toLayer.name == lto:
                return connection
        raise NetworkError('Connection was not found.', (lfrom, lto))
    def setVerbosity(self, value):
        """
        Sets network self._verbosity and each layer._verbosity to value.
        """
        self._verbosity = value
        for layer in self.layers:
            layer._verbosity = value
    def getVerbosity(self): return self._verbosity
    def setStopPercent(self, value):
        """
        Sets self.stopPercent to value. train() will stop if the percent
        correct surpasses the stop percent.
        """
        self.stopPercent = value
    def setUseCrossValidationToStop(self, value):
        """
        Sets flag so that self.stopPercent is compared to cross validation
        percent rather than the regular training data percentage correct.
        """
        self.useCrossValidationToStop = value
    def setSigmoid_prime_offset(self, value):
        """
        Sets self.sigmoid_prime_offset to value.
        """
        self.sigmoid_prime_offset = value
    def setReportRate(self, value):
        """
        Sets self.reportRate to value.  train() will report when
        epoch % reportRate == 0. Cross validation will only report if
        there is a cross validation data set.
        """
        self.reportRate = value
    def setSweepReportRate(self, value):
        """
        Sets self.sweepReportRate to value.  sweep() will report
        when epoch % sweepReportRate == 0.  there is a cross
        validation data set.
        """
        self.sweepReportRate = value
    def getMaxRandom(self): return self._maxRandom
    def setMaxRandom(self, value):
        """
        Sets the maxRandom Layer attribute for each layer to value.Specifies
        the global range for randomly initialized values, [-max, max].
        """
        self._maxRandom = value
        for layer in self.layers:
            layer._maxRandom = value
    def getWeights(self, fromName, toName):
        """
        Gets the weights of the connection between two layers (argument strings).
        """
        for connection in self.connections:
            if connection.fromLayer.name == fromName and \
               connection.toLayer.name == toName:
                return connection.weight
        raise NetworkError('Connection was not found.', (fromName, toName))
    def setWeight(self, fromName, fromPos, toName, toPos, value):
        """
        Sets the weight of the connection between two layers (argument strings).
        """
        for connection in self.connections:
            if connection.fromLayer.name == fromName and \
               connection.toLayer.name == toName:
                connection.weight[fromPos][toPos] = value
                return value
        raise NetworkError('Connection was not found.', (fromName, toName))
    def setOrderedInputs(self, value):
        """
        Sets self.orderedInputs to value. Specifies if inputs
        should be ordered and if so orders the inputs.
        """
        self.orderedInputs = value
        if self.orderedInputs:
            self.loadOrder = [0] * len(self.inputs)
            for i in range(len(self.inputs)):
                self.loadOrder[i] = i
    def verifyArguments(self, arg):
        """
        Verifies that arguments to setInputs and setTargets are appropriately formatted.
        """
        for l in arg:
            if not type(l) == list and \
               not type(l) == type(Numeric.array([0.0])) and \
               not type(l) == tuple and \
               not type(l) == dict:
                return 0
            if type(l) == dict:
                for i in l:
                    if not type(i) == str and i not in list(self.layers.keys()):
                        return 0
            else:
                for i in l:
                    if not type(i) == float and not type(i) == int:
                        return 0
        return 1
    def setInputs(self, inputs):
        """
        Sets self.input to inputs. Load order is by default random. Use setOrderedInputs() to order inputs.
        """
        if not self.verifyArguments(inputs) and not self.patterned:
            raise NetworkError('setInputs() requires [[...],[...],...] or [{"layerName": [...]}, ...].', inputs)
        self.inputs = inputs
        self.loadOrder = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            self.loadOrder[i] = i
        # will randomize later, if need be
    def setOutputs(self, outputs):
        """
        For compatiblity.
        """
        self.setTargets(outputs)
    def setTargets(self, targets):
        """
        Sets the targets.
        """
        if not self.verifyArguments(targets) and not self.patterned:
            raise NetworkError('setTargets() requires [[...],[...],...] or [{"layerName": [...]}, ...].', targets)
        self.targets = targets
    def setInputsAndTargets(self, data1, data2=None):
        """
        Network.setInputsAndTargets()
        Sets the corpus of data for training. Can be in one of two formats:

        Format 1: setInputsAndTargets([[input0, target0], [input1, target1]...])
        Network.setInputsAndTargets([[[i00, i01, ...], [t00, t01, ...]],
                                     [[i10, i11, ...], [t10, t11, ...]],
                                     ...])
        Format 2: setInputsAndTargets([input0, input1, ...], [target0, target1, ...])
        Network.setInputsAndTargets([[i00, i01, ...], [i10, i11, ...],...],
                                    [[t00, t01, ...], [t10, t11, ...],...] )
        
        """
        if data2 == None: # format #1
            inputs = [x[0] for x in data1]
            targets = [x[1] for x in data1]
        else: # format 2
            inputs = data1
            targets = data2
        self.setInputs(inputs)
        self.setTargets(targets)
    def associate(self, inName, outName):
        """
        inName layer and outName layer will be auto-associating.
        """
        self.association.append((inName, outName))
    def mapInput(self, layerName, offset = 0):
        """
        Adds layerName and offset to inputMap.
        """
        self.inputMap.append((layerName, offset))
    def mapInputs(self, nameOffsetPairs):
        for pair in nameOffsetPairs:
            self.mapInput(pair[0], pair[1])
    def mapTarget(self, layerName, offset = 0):
        """
        Adds layerName and offset to targetMap.
        """
        self.targetMap.append((layerName, offset))
    def mapTargets(self, nameOffsetPairs):
        for pair in nameOffsetPairs:
            self.mapTarget(pair[0], pair[1])

    # input and target methods
    def randomizeOrder(self):
        """
        Randomizes self.loadOrder, the order in which inputs set with
        self.setInputs() are presented.
        """
        flag = [0] * len(self.inputs)
        self.loadOrder = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            pos = int(random.random() * len(self.inputs))
            while (flag[pos] == 1):
                pos = int(random.random() * len(self.inputs))
            flag[pos] = 1
            self.loadOrder[pos] = i

    def copyVector(self, vector1, vec2, start):
        """
        Copies vec2 into vector1 being sure to replace patterns if
        necessary. Use self.copyActivations() or self.copyTargets()
        instead.
        """
        vector2 = self.replacePatterns(vec2)
        length = min(len(vector1), len(vector2))
        if self.verbosity > 4:
            print("Copying Vector: ", vector2[start:start+length])
        p = 0
        for i in range(start, start + length):
            vector1[p] = vector2[i]
            p += 1
    def copyActivations(self, layer, vec, start = 0):
        """
        Copies activations in vec to the specified layer, replacing
        patterns if necessary.
        """
        vector = self.replacePatterns(vec, layer.name)
        if self.verbosity > 4:
            print("Copying Activations: ", vector[start:start+layer.size])
        layer.copyActivations(vector[start:start+layer.size])
    def copyTargets(self, layer, vec, start = 0):
        """
        Copies targets in vec to specified layer, replacing patterns
        if necessary.
        """
        vector = self.replacePatterns(vec, layer.name)
        if self.verbosity > 4:
            print("Copying Target: ", vector[start:start+layer.size])
        layer.copyTargets(vector[start:start+layer.size])
    def getDataCrossValidation(self, pos):
        """
        Returns the inputs/targets for a pattern pos, or assumes that
        the layers are called input and output and uses the lists
        in self.inputs and self.targets.
        """
        set = {}
        if type(self.inputs[pos]) == dict:
            set.update(self.inputs[pos])
        else:
            set["input"] = self.inputs[pos]
        if self.targets:
            if type(self.targets[pos]) == dict:
                set.update(self.targets[pos])
            else:
                set["output"] = self.targets[pos]
        return set
    def getDataMap(self, intype, pos, name, offset = 0):
        """
        Hook defined to lookup a name, and get it from a vector.
        Can be overloaded to get it from somewhere else.
        """
        if intype == "input":
            vector = self.inputs
        elif intype == "target":
            vector = self.targets
        else:
            raise AttributeError("invalid map type '%s'" % intype)
        return vector[pos][offset:offset+self[name].size]
    def getData(self, pos):
        """
        Returns dictionary with input and target given pos. 
        """
        retval = {}
        if pos >= len(self.inputs):
            raise IndexError('getData() pattern beyond range.', pos)
        if self.verbosity >= 1: print("Getting input", pos, "...")
        if len(self.inputMap) == 0:
            if type(self.inputs[pos]) == dict: # allow inputs to be a dict
                retval.update(self.inputs[pos])
            else:
                retval[self.layers[0].name] = self.inputs[pos]
        else: # mapInput set manually
            for vals in self.inputMap:
                (name, offset) = vals
                retval[name] = self.getDataMap("input", pos, name, offset)
        if self.verbosity > 1: print("Loading target", pos, "...")
        if len(self.targets) == 0:
            pass # ok, no targets
        elif len(self.targetMap) == 0:
            if type(self.targets[pos]) == dict: # allow targets to be a dict
                retval.update(self.targets[pos])
            else:
                retval[self.layers[len(self.layers)-1].name] = self.targets[pos]
        else: # set manually
            for vals in self.targetMap:
                (name, offset) = vals
                retval[name] = self.getDataMap("target", pos, name, offset)
        return retval

    # input, architecture, and target verification
    def verifyArchitecture(self):
        """
        Check for orphaned layers or connections. Assure that network
        architecture is feed-forward (no-cycles). Check connectivity. Check naming.
        """
        if len(self.cacheLayers) != 0 or len(self.cacheConnections) != 0: return
        # flags for layer type tests
        hiddenInput = 1
        hiddenOutput = 1       
        outputLayerFlag = 1  
        inputLayerFlag = 1
        # must have layers and connections
        if len(self.layers) == 0:
            raise NetworkError('No network layers.', \
                                 self.layers)
        if len(self.connections) == 0:
            raise NetworkError('No network connections.', \
                                 self.connections)
        # layers should not have the same name
        for x, y in [(x, y) for x in range(len(self.layers)) for y in range(len(self.layers))]:
            if x == y:
                pass # same layer so same name
            else:
                # different layers same name
                if self.layers[x].name == self.layers[y].name:
                    raise NetworkError('Two layers have the same name.', (x,y))
        # no multiple connections between layers
        for x, y in [(x,y) for x in range(len(self.connections)) for y in range(len(self.connections))]:
            if x == y:
                pass # same connection
            else:
                # multiple connections between fromLayer and toLayer
                if self.connections[x].fromLayer.name == self.connections[y].fromLayer.name and \
                   self.connections[x].toLayer.name == self.connections[y].toLayer.name:
                    raise NetworkError('Multiple connections between two layers.', \
                                         (self.connections[x].fromLayer.name, \
                                          self.connections[x].toLayer.name))
        # layer type tests
        for layer in self.layers:
            # no undefined layers
            if layer.type == 'Undefined':
                raise NetworkError('There is an unconnected layer.', layer.name)
            elif layer.type == 'Input':
                for connection in self.connections:
                    # input layers must have outgoing connections 
                    if connection.fromLayer.name == layer.name:
                        inputLayerFlag = 0
                    # input layers must have no incoming connections
                    if connection.toLayer.name == layer.name:
                        raise NetworkError('Layer has type \'Input\' and an incoming connection.', layer.name)
                if inputLayerFlag:
                    raise NetworkError('Layer has type \'Input\' but no outgoing connections', layer.name)
            elif layer.type == 'Output':
                for connection in self.connections:
                    # output layers must have no outgoing connections`
                    if connection.fromLayer.name == layer.name:
                        raise NetworkError('Layer has type \'Output\' and an outgoing connections.',layer.name)
                    # output layers must have an incoming connection
                    if connection.toLayer.name == layer.name:
                        outputLayerFlag = 0
                if outputLayerFlag:
                    raise NetworkError('Layer has type \'Output\' and no incoming connections.', layer.name)           
            elif layer.type == 'Hidden':
                for connection in self.connections:
                    # hidden layers must have incoming and outgoing connections.
                    if connection.toLayer.name == layer.name:
                        hiddenInput = 0
                    if connection.fromLayer.name == layer.name:
                        hiddenOutput = 0
                if hiddenInput or hiddenOutput:
                    raise NetworkError('Layer has type \'Hidden\' but does not have both input and output connections.',\
                           layer.name)
            else:
                raise LayerError('Unknown layer encountered in verifyArchitecture().', layer.name)
        # network should not have unconnected sub networks
        # every input layer should have a path to every output layer
        for inLayer in self.layers:
            if inLayer.type == 'Input':
                for outLayer in self.layers:
                    if outLayer.type == 'Output':
                        if not self.path(inLayer, outLayer):
                            raise NetworkError('Network contains disconnected sub networks.', \
                                                 (inLayer.name, outLayer.name))
        # network should not have directed cycles
        for layer in self.layers:
            if self.path(layer, layer):
                raise NetworkError('Network contains a cycle.', layer.name)   
    def verifyInputs(self):
        """
        Used in propagate() to verify that the network input
        activations have been set.
        """
        for layer in self.layers:
            if (layer.verify and
                layer.type == 'Input' and
                layer.kind != 'Context' and
                layer.active and
                not layer.activationSet):
                raise LayerError("Inputs are not set and verifyInputs() was called on layer '%s'." % layer.name)
            else:
                layer.resetActivationFlag()
    def verifyTargets(self):
        """
        Used in backprop() to verify that the network targets have
        been set.
        """
        for layer in self.layers:
            if layer.verify and layer.type == 'Output' and layer.active and not layer.targetSet:
                raise LayerError('Targets are not set and verifyTargets() was called.',\
                                   (layer.name, layer.type))
            else:
                layer.resetTargetFlag()

    # error reporting methods
    def getCorrect(self, layerName):
        """
        Returns the number of correct activation within tolerance of a
        layer.
        """
        return self.getLayer(layerName).getCorrect(self.tolerance)
    def RMSError(self):
        """
        Returns Root Mean Squared Error for all output layers in this network.
        """
        tss = 0.0
        size = 0
        for layer in self.layers:
            if layer.type == 'Output':
                tss += layer.TSSError()
                size += layer.size
        return math.sqrt( tss / size )
    def TSSError(self, layerName):
        """
        Returns Total Sum Squared error for the specified layer's pattern.
        """
        return self.getLayer(layerName).TSSError()
    def Print(self, msg):
        if self.log:
            self.log.write(msg + "\n")
        else:
            print(msg)
    def reportEpoch(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print("Epoch #%6d | TSS Error: %.4f | Correct: %.4f | RMS Error: %.4f" % \
                   (epoch, tssErr, totalCorrect * 1.0 / totalCount, rmsErr))
        for layerName in pcorrect:
            if self[layerName].active and self[layerName].size > 1:
                self.Print("   Epoch #%6d, Layer = %-12s | Units: %.4f | Patterns: %.4f" % \
                           (epoch, "'" + layerName + "'",
                            float(pcorrect[layerName][0]) / pcorrect[layerName][1],
                            float(pcorrect[layerName][2]) / pcorrect[layerName][3]))
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()
    def reportPattern(self): pass
    def reportStart(self): pass
    def reportFinal(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print("Final #%6d | TSS Error: %.4f | Correct: %.4f | RMS Error: %.4f" % \
                   (epoch-1, tssErr, totalCorrect * 1.0 / totalCount, rmsErr))
        for layerName in pcorrect:
            if self[layerName].active and self[layerName].size > 1:
                self.Print("   Final #%6d, Layer = %-12s | Units: %.4f | Patterns: %.4f" % \
                           (epoch-1, "'" + layerName + "'",
                            float(pcorrect[layerName][0]) / pcorrect[layerName][1],
                            float(pcorrect[layerName][2]) / pcorrect[layerName][3]))
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()
    # train and sweep methods
    def doWhile(self, totalCount, totalCorrect):
        return totalCount != 0 and ((totalCorrect * 1.0 / totalCount < self.stopPercent) or self.useCrossValidationToStop)
    def train(self, sweeps=None, cont=0):
        """
        Trains the network on the dataset till a stopping condition is
        met. This stopping condition can be a limiting epoch or a percentage correct requirement.
        """
        # check architecture
        self.complete = 0
        self.verifyArchitecture()
        tssErr = 0.0; rmsErr = 0.0; totalCorrect = 0; totalCount = 1; totalPCorrect = {}
        if not cont: # starting afresh
            self.resetFlags()
            self.epoch = 0
            self.reportStart()
            self.resetCount = 1
            self.epoch = 1
            self.lastLowestTSSError = sys.maxsize # some maximum value (not all pythons have Infinity)
            if sweeps != None:
                self.resetEpoch = sweeps
        else:
            if sweeps != None:
                self.resetEpoch = self.epoch + sweeps - 1
        while self.doWhile(totalCount, totalCorrect):
            (tssErr, totalCorrect, totalCount, totalPCorrect) = self.sweep()
            self.complete = 1
            if totalCount != 0:
                rmsErr = math.sqrt(tssErr / totalCount)
            else:
                self.Print("Warning: sweep didn't do anything!")
            if self.epoch % self.reportRate == 0:
                self.reportEpoch(self.epoch, tssErr, totalCorrect, totalCount, rmsErr, totalPCorrect)
                if len(self.crossValidationCorpus) > 0 or self.autoCrossValidation:
                    (tssCVErr, totalCVCorrect, totalCVCount, totalCVPCorrect) = self.sweepCrossValidation()
                    rmsCVErr = math.sqrt(tssCVErr / totalCVCount)
                    self.Print("CV    #%6d | TSS Error: %.4f | Correct: %.4f | RMS Error: %.4f" % \
                               (self.epoch, tssCVErr, totalCVCorrect * 1.0 / totalCVCount, rmsCVErr))
                    if self.autoSaveWeightsFile != None and tssCVErr < self.lastLowestTSSError:
                        self.lastLowestTSSError = tssCVErr
                        self.saveWeights(self.autoSaveWeightsFile, mode = self.autoSaveWeightsFileFormat)
                        self.Print("auto saving weights to '%s'..." % self.lastAutoSaveWeightsFilename)
                    if self.autoSaveNetworkFile != None:
                        self.saveNetwork(self.autoSaveNetworkFile, mode = self.autoSaveNetworkFileFormat)
                        self.Print("auto saving network to '%s'..." % self.lastAutoSaveNetworkFilename)
                    if totalCVCorrect * 1.0 / totalCVCount >= self.stopPercent and self.useCrossValidationToStop:
                        self.epoch += 1
                        break
            if self.resetEpoch == self.epoch:
                if self.resetCount == self.resetLimit:
                    self.Print("Reset limit reached; ending without reaching goal")
                    self.epoch += 1
                    self.complete = 0
                    break
                self.resetCount += 1
                self.Print("RESET! resetEpoch reached; starting over...")
                self.initialize()
                tssErr = 0.0; rmsErr = 0.0; self.epoch = 1; totalCorrect = 0; totalPCorrect = {}
                continue
            self.epoch += 1
        if totalCount > 0:
            self.reportFinal(self.epoch, tssErr, totalCorrect, totalCount, rmsErr, totalPCorrect)
            if len(self.crossValidationCorpus) > 0 or self.autoCrossValidation:
                (tssCVErr, totalCVCorrect, totalCVCount, totalCVPCorrect) = self.sweepCrossValidation()
                rmsCVErr = math.sqrt(tssCVErr / totalCVCount)
                self.Print("CV    #%6d | TSS Error: %.4f | Correct: %.4f | RMS Error: %.4f" % \
                           (self.epoch-1, tssCVErr, totalCVCorrect * 1.0 / totalCVCount, rmsCVErr))
                if self.autoSaveWeightsFile != None and tssCVErr < self.lastLowestTSSError:
                    self.lastLowestTSSError = tssCVErr
                    self.saveWeights(self.autoSaveWeightsFile, mode = self.autoSaveWeightsFileFormat)
                    self.Print("auto saving weights to '%s'..." % self.lastAutoSaveWeightsFilename)
                if self.autoSaveNetworkFile != None:
                    self.saveNetwork(self.autoSaveNetworkFile, mode = self.autoSaveNetworkFileFormat)
                    self.Print("auto saving network to '%s'..." % self.lastAutoSaveNetworkFilename)
        else:
            print("Final: nothing done")

    def trainOne(self, **args):
        """
        Network.trainOne()
        Does a single step, but just returns error.
        Calls propagate(), backprop(), and change_weights() if learning is set.
        Format for parameters: <layer name> = <activation/target list>
        """
        return self.step(**args)[0]

    def step(self, **args):
        """
        Network.step()
        Does a single step. Calls propagate(), backprop(), and
        change_weights() if learning is set.
        Format for parameters: <layer name> = <activation/target list>
        
        """
        if self.verbosity > 0:
            print("Network.step() called with:", args)
        # First, copy the values into either activations or targets:
        retargs = self.preStep(**args)
        if retargs: args = retargs # replace the args
        # Propagate activation through network:
        self.propagate(**args)
        retargs = self.postPropagate(**args)
        if retargs: args = retargs # replace the args
        # Next, take care of any Auto-association, and copy
        # activations to targets
        for aa in self.association:
            (inName, outName) = aa
            inLayer = self.getLayer(inName)
            if inLayer.type not in ['Input', "Hidden"]:
                raise LayerError('Associated input layer not type \'Input\' or \'Hidden\'.', \
                                   inLayer.type)
            outLayer = self.getLayer(outName)
            if not outLayer.type == 'Output':
                raise LayerError('Associated output layer not type \'Output\'.', \
                                   outLayer.type)
            outLayer.copyTargets(inLayer.activation)
        # Compute error, and back prop it:
        retargs = self.preBackprop(**args)
        if retargs: args = retargs # replace the args
        (error, correct, total, pcorrect) = self.backprop(**args) # compute_error()
        if self.verbosity > 2 or self.interactive:
            self.display()
            if self.interactive:
                self.prompt()
        retargs = self.postBackprop(**args)
        if retargs: args = retargs # replace the args
        # if learning is true, and need to update weights here:
        if self.learning and not self.batch:
            self.change_weights() # else change weights in sweep
        retargs = self.postStep(**args)
        if retargs: args = retargs # replace the args
        self.reportPattern()
        return (error, correct, total, pcorrect)
    # Hooks for adding bits without having to breakup step(), train(), and sweep()
    def preBackprop(self, **args):   return None
    def postBackprop(self, **args):  return None
    def prePropagate(self, **args):  pass
    def postPropagate(self, **args): return None
    def preStep(self, **args):       return None
    def postStep(self, **args):      return None
    def preSweep(self):      pass
    def postSweep(self):     pass
    def sweep(self):
        """
        Runs through entire dataset. 
        Returns TSS error, total correct, total count, pcorrect (a dict of layer data)
        """
        self.preSweep()
        if self.loadOrder == []:
            raise NetworkError('No loadOrder for the inputs. Make sure inputs are properly set.', self.loadOrder)
        if len(self.targets) != 0 and len(self.targets) != len(self.inputs):
            raise NetworkError("Number of inputs does not equal number of targets (inputs=%d, targets=%d)" % (len(self.targets), len(self.inputs)))
        if self.verbosity >= 1: print("Epoch #", self.epoch, "Cycle...")
        if not self.orderedInputs:
            self.randomizeOrder()
        tssError = 0.0; totalCorrect = 0; totalCount = 0; totalPCorrect = {}
        cnt = 0
        if self.saveResults:
            self.results = [(0,0,0) for x in self.loadOrder]
        for i in self.loadOrder:
            if self.verbosity >= 1 or self.interactive:
                print("-----------------------------------Pattern #", self.loadOrder[i] + 1)
            datum = self.getData(i) # creates a dictionary of input/targets from self.inputs, self.targets
            if cnt < len(self.loadOrder) - 1:
                self.currentSweepCount = cnt
            else:
                self.currentSweepCount = None
            self._sweeping = 1
            (error, correct, total, pcorrect) = self.step( **datum )
            self._sweeping = 0
            if self.saveResults:
                self.results[i] = (error, correct, total, pcorrect)
            tssError += error
            totalCorrect += correct
            totalCount += total
            sumMerge(totalPCorrect, pcorrect)
            if self.sweepReportRate and (cnt + 1) % self.sweepReportRate == 0:
                print("   Step # %6d | TSS Error: %.4f | Correct: %.4f" % \
                      (cnt + 1, tssError, totalCorrect * 1.0 / totalCount))
            if self.crossValidationSampleRate and self.epoch % self.crossValidationSampleRate == 0:
                self.saveNetworkForCrossValidation(self.crossValidationSampleFile)
            cnt += 1
        if self.learning and self.batch:
            self.change_weights() # batch mode, otherwise change weights in step
        self.postSweep()
        return (tssError, totalCorrect, totalCount, totalPCorrect)
    def sweepCrossValidation(self):
        """
        sweepCrossValidation() will go through each of the crossvalidation input/targets.
        The crossValidationCorpus is a list of dictionaries of input/targets
        referenced by layername.
        Example: ({"input": [0.0, 0.1], "output": [1.0]}, {"input": [0.5, 0.9], "output": [0.0]})
        """
        # get learning value and then turn it off
        oldLearning = self.learning
        self.learning = 0
        tssError = 0.0; totalCorrect = 0; totalCount = 0; totalPCorrect = {}
        self._cv = True # in cross validation
        if self.autoCrossValidation:
            for i in range(len(self.inputs)):
                set = self.getDataCrossValidation(i)
                self._sweeping = 1
                (error, correct, total, pcorrect) = self.step( **set )
                self._sweeping = 0
                if self.crossValidationReportLayers != []:
                    (error, correct, total, pcorrect) = self.getError( *self.crossValidationReportLayers )
                tssError += error
                totalCorrect += correct
                totalCount += total
                sumMerge(totalPCorrect, pcorrect)
        else:
            for set in self.crossValidationCorpus:
                self._sweeping = 1
                (error, correct, total, pcorrect) = self.step( **set )
                self._sweeping = 0
                if self.crossValidationReportLayers != []:
                    (error, correct, total, pcorrect) = self.getError( *self.crossValidationReportLayers )
                tssError += error
                totalCorrect += correct
                totalCount += total
                sumMerge(totalPCorrect, pcorrect)
        self.learning = oldLearning
        self._cv = False
        return (tssError, totalCorrect, totalCount, totalPCorrect)
    def saveNetworkForCrossValidation(self, filename, mode = 'a'):
        fp = open(filename, mode)
        # for each layer, if type is Input or Output, save it with name
        # one per line
        fp.write("{")
        for layer in self.layers:
            if layer.type == 'Input':
                fp.write('"' + layer.name + '": [')
                writeArray(fp, layer.activation, delim = ", ", nl = 0)
                fp.write('], ')
            elif layer.type == 'Output':
                fp.write('"' + layer.name + '": [')
                writeArray(fp, layer.target, delim = ", ", nl = 0)
                fp.write('], ')
        fp.write("}\n")
        fp.close()
    def loadCrossValidation(self, filename):
        self.crossValidationCorpus = []
        fp = open(filename, "r")
        for line in fp:
            dict = eval( line )
            self.crossValidationCorpus.append( dict )
        fp.close()
    def cycle(self):
        """
        Alternate to sweep().
        """
        return self.sweep()

    def numConnects(self, layerName):
        """ Number of incoming weights, including bias. Assumes fully connected. """
        count = 0
        if self[layerName].active: 
            count += 1 # 1 = bias
            for connection in self.connections:
                if connection.active and connection.fromLayer.active and connection.toLayer.name == layerName:
                    count += connection.fromLayer.size
        return count

    # propagation methods
    def prop_from(self, startLayers):
        """
        Start propagation from the layers in the list
        startLayers. Make sure startLayers are initialized with the
        desired activations. NO ERROR CHECKING.
        """
        if self.verbosity > 2: print("Partially propagating network:")
        # find all the layers involved in the propagation
        propagateLayers = []
        # propagateLayers should not include startLayers (no loops)
        for startLayer in startLayers:
            for layer in self.layers:
                if self.path(startLayer, layer):
                    propagateLayers.append(layer)
        for layer in propagateLayers:
            if layer.active: 
                layer.netinput = (layer.weight).copy() 
        for layer in propagateLayers:
            if layer.active:
                for connection in self.connections:
                    if connection.active and connection.toLayer.name == layer.name:
                        connection.toLayer.netinput = connection.toLayer.netinput + \
                                                      Numeric.matrixmultiply(connection.fromLayer.activation,\
                                                                             connection.weight) # propagate!
                if layer.type != 'Input':
                    layer.activation = self.activationFunction(layer.netinput)
        for layer in propagateLayers:
            if layer.log and layer.active:
                layer.writeLog(self)
    def propagate(self, **args):
        """
        Propagates activation through the network. Optionally, takes input layer names
        as keywords, and their associated activations. If input layer(s) are given, then
        propagate() will return the output layer's activation. If there is more than
        one output layer, then a dictionary is returned.

        Examples:

        >>> net = Network() # doctest: +ELLIPSIS
        Conx using seed: ...
        >>> net.addLayers(2, 5, 1)
        >>> len(net.propagate(input = [0, .5]))
        1
        """
        self.prePropagate(**args)
        for key in args:
            layer = self.getLayer(key)
            if layer.kind == 'Input':
                if self[key].verify and not self[key].activationSet == 0: 
                    raise AttributeError("attempt to set activations on input layer '%s' without reset" % key)
                self.copyActivations(layer, args[key])
            elif layer.kind == 'Context':
                self.copyActivations(layer, args[key])
            elif layer.kind == 'Output' and len(args[key]) == layer.size: # in case you expect propagate to handle the outputs
                self.copyTargets(layer, args[key])
        self.verifyInputs() # better have inputs set
        if self.verbosity > 2: print("Propagate Network '" + self.name + "':")
        # initialize netinput:
        for layer in self.layers:
            if layer.type != 'Input' and layer.active:
                layer.netinput = (layer.weight).copy() 
        # for each connection, in order:
        for layer in self.layers:
            if layer.active:
                for connection in self.connections:
                    if (connection.toLayer.name == layer.name
                        and connection.fromLayer.active
                        and connection.active):

                        a = connection.fromLayer.activation
                        w = connection.weight
                        m = Numeric.matrixmultiply(a, w)
                        ni = m + connection.toLayer.netinput

                        connection.toLayer.netinput = ni

                        #connection.toLayer.netinput = \
                        #    (connection.toLayer.netinput + 
                        #     Numeric.matrixmultiply(connection.fromLayer.activation,
                        #                            connection.weight)) # propagate!

                if layer.type != 'Input':
                    layer.activation = self.activationFunction(layer.netinput)
        for layer in self.layers:
            if layer.log and layer.active:
                layer.writeLog(self)
        self.count += 1 # counts number of times propagate() is called
        if len(args) != 0:
            dict = {}
            for layer in self.layers:
                if layer.type == "Output":
                    dict[layer.name] = layer.activation.copy()
            if len(dict) == 1:
                return dict[list(dict.keys())[0]]
            else:
                return dict
    def propagateTo(self, toLayer, **args):
        """
        Propagates activation to a layer. Optionally, takes input layer names
        as keywords, and their associated activations. Returns the toLayer's activation.

        Examples:

        >>> net = Network() # doctest: +ELLIPSIS
        Conx using seed: ...
        >>> net.addLayers(2, 5, 1)
        >>> len(net.propagateTo("output"))
        1
        >>> len(net.propagateTo("hidden"))
        5
        >>> len(net.propagateTo("hidden", input = [0, 0]))
        5
        """
        for layerName in args:
            self[layerName].activationSet = 0 # force it to be ok
            self[layerName].copyActivations(args[layerName])
        # init toLayer:
        self[toLayer].netinput = (self[toLayer].weight).copy() 
        # for each connection, in order:
        for connection in self.connections:
            if connection.active and connection.toLayer.name == toLayer and connection.fromLayer.active:
                connection.toLayer.netinput = connection.toLayer.netinput + \
                                              Numeric.matrixmultiply(connection.fromLayer.activation,\
                                                                     connection.weight) # propagate!
                if self[toLayer].type != 'Input':
                    self[toLayer].activation = self.activationFunction(self[toLayer].netinput)
        return self[toLayer].activation.copy()
    def propagateFrom(self, startLayer, **args):
        """
        Propagates activation through the network. Optionally, takes input layer names
        as keywords, and their associated activations. If input layer(s) are given, then
        propagate() will return the output layer's activation. If there is more than
        one output layer, then a dictionary is returned.

        Examples:

        >>> net = Network() # doctest: +ELLIPSIS
        Conx using seed: ...
        >>> net.addLayers(2, 5, 1)
        >>> len(net.propagate(input = [1, .5]))
        1
        """
        for layerName in args:
            self[layerName].copyActivations(args[layerName])
        # initialize netinput:
        started = 0
        for layer in self.layers:
            if layer.name == startLayer:
                started = 1
                continue # don't set this one
            if not started: continue
            if layer.type != 'Input' and layer.active:
                layer.netinput = (layer.weight).copy() 
        # for each connection, in order:
        started = 0
        for layer in self.layers:
            if layer.name == startLayer:
                started = 1
                continue # don't get inputs into this one
            if not started: continue
            if layer.active:
                for connection in self.connections:
                    if connection.active and connection.toLayer.name == layer.name and connection.fromLayer.active:
                        connection.toLayer.netinput = connection.toLayer.netinput + \
                                                      Numeric.matrixmultiply(connection.fromLayer.activation,\
                                                                             connection.weight) # propagate!
                if layer.type != 'Input':
                    layer.activation = self.activationFunction(layer.netinput)
        for layer in self.layers:
            if layer.log and layer.active:
                layer.writeLog(self)
        self.count += 1 # counts number of times propagate() is called
        if len(args) != 0:
            dict = {}
            for layer in self.layers:
                if layer.type == "Output":
                    dict[layer.name] = layer.activation.copy()
            if len(dict) == 1:
                return dict[list(dict.keys())[0]]
            else:
                return dict
    #Perhaps these activation functions could be encapsulated in objects?
    #  We have various activation function classes that also hold their min and max values and the three or
    #  four functions required to define them.
    def activationFunctionASIG(self, x):
        """
        Determine the activation of a node based on that nodes net input.
        """
        def act(v):
            if   v < -15.0: return 0.0
            elif v >  15.0: return 1.0
            else: return 1.0 / (1.0 + Numeric.exp(-v))
        return Numeric.array(list(map(act, x)), 'f')
    def ACTPRIMEASIG(self, act):
        """
        Used in compute_error.
        """
        retval = (act * (1.0 - act)) + self.sigmoid_prime_offset
        return retval
    def actDerivASIG(self, x):
        """
        Only works on scalars.
        """
        def act(v):
            if   v < -15.0: return 0.0
            elif v >  15.0: return 1.0
            else: return 1.0 / (1.0 + Numeric.exp(-v))
        return (act(x) * (1.0 - act(x))) + self.sigmoid_prime_offset
    #here are three functions that define a symmetric sigmoid activation function identical to the one
    # fahlman uses in his code
    def activationFunctionFahlman(self, x):
        def act(v):
            if   v < -15.0: return -0.5
            elif v >  15.0: return 0.5
            else: return 1.0 / (1.0 + Numeric.exp(-v)) - 0.5
        return Numeric.array(list(map(act, x)), 'f')
    def ACTPRIME_Fahlman(self, act):
        return self.sigmoid_prime_offset+0.25 - act*act
    def actDerivFahlman(self, x):
        def act(v):
            if   v < -15.0: return -0.5
            elif v >  15.0: return 0.5
            else: return 1.0 / (1.0 + Numeric.exp(-v)) - 0.5
        return self.ACTPRIME_Fahlman( act(x) )
    
    #here are three functions that define an alternative node activation function
    def activationFunctionTANH(self, x):
        def act(v):
            if   v < -15.0: return -1.0
            elif v >  15.0: return 1.0
            else: return 1.7159 * Numeric.tanh(0.66666 * v)
        return Numeric.array(list(map(act, x)), 'f')
    def ACTPRIMETANH(self, act):
        return 1 - act*act + self.sigmoid_prime_offset
    def actDerivTANH(self, x):
        #print x, 1/Numeric.cosh(x)
        #return 1/Numeric.cosh(x) + self.sigmoid_prime_offset
        def act(v):
            if   v < -15.0: return -1.0
            elif v >  15.0: return 1.0
            else: return 1.7159 * Numeric.tanh(0.66666 * v)
        return self.ACTPRIMETANH(act(x))
    def useTanhActivationFunction(self):
        """
        Change the network to use the hyperbolic tangent activation function for all layers.
        Must be called after all layers have been added.
        """
        self.activationFunction = self.activationFunctionTANH
        self.ACTPRIME = self.ACTPRIMETANH
        self.actDeriv = self.actDerivTANH
        for layer in self:
            layer.minTarget, layer.minActivation = -1.7159, -1.7159 
            layer.maxTarget, layer.maxActivation = 1.7159, 1.7159
    def useFahlmanActivationFunction(self):
        """
        Change the network to use Fahlman's default activation function for all layers.
        Must be called after all layers have been added.
        """
        self.activationFunction = self.activationFunctionFahlman
        self.ACTPRIME = self.ACTPRIME_Fahlman
        self.actDeriv = self.actDerivFahlman
        for layer in self:
            layer.minTarget, layer.minActivation = -0.5, -0.5 
            layer.maxTarget, layer.maxActivation = 0.5, 0.5
    #bind the default activation function and its related functions
    activationFunction = activationFunctionASIG
    ACTPRIME = ACTPRIMEASIG
    actDeriv = actDerivASIG
    # backpropagation
    def backprop(self, **args):
        """
        Computes error and wed for back propagation of error.
        """
        retval = self.compute_error(**args)
        if self.learning:
            self.compute_wed()
        return retval
    def deltaWeight(self, e, wed, m, dweightLast, wedLast, w, n):
        """
        e - learning rate
        wed - weight error delta vector (slope)
        m - momentum
        dweightLast - previous dweight vector (slope)
        wedLast - only used in quickprop; last weight error delta vector
        w - weight vector
        n - fan-in, number of connections coming in (counts bias, too)
        """
        #print "WEIGHT = ", w
        shrinkFactor = self.mu / (1.0 + self.mu)
        if self.splitEpsilon:
            e /= float(n)
        if self._quickprop:
            nextStep = Numeric.zeros(len(dweightLast), 'f')
            for i in range(len(dweightLast)):
                s = wed[i] 
                d = dweightLast[i]
                p = wedLast[i]
                #print self.mu
                #print "slopes[node] = %f  QP w=%s d=%f s=%f p=%f eps=%f" % (s, w, d, s, p, e)
                #print type(nextStep[i]), nextStep[i]
                if (d > 0.0):
                    if (s > 0.0):
                        #print "CASE A1"
                        nextStep[i] = nextStep[i] + (e * s)
                    if (s >= (shrinkFactor * p)):
                        #print "CASE B1"
                        nextStep[i] = nextStep[i] + (self.mu * d)
                    else:
                        #print "CASE C1"
                        nextStep[i] = nextStep[i] + (d * s / (p - s))
                elif (d < 0.0):
                    if (s < 0.0):
                        #print "CASE A2"
                        nextStep[i] = nextStep[i] + (e * s)
                    if (s <= (shrinkFactor * p)):
                        #print "CASE B2"
                        nextStep[i] = nextStep[i] + (self.mu * d)
                    else:
                        #print "CASE C2"
                        nextStep[i] = nextStep[i] + (d * s / (p - s))
                else:
                    #print "CASE D"
                    nextStep[i] = nextStep[i] + (e * s + m * d)
                    ##  Last step was zero, so only use linear   ##
            newDweight = nextStep
            #print "Next Step = ", nextStep[i]
        else: # backprop
            newDweight = e * wed + m * dweightLast # gradient descent
        return newDweight
    def change_weights(self):
        """
        Changes the weights according to the error values calculated
        during backprop(). Learning must be set.
        """
        dw_count, dw_sum = 0, 0.0
        if len(self.cacheLayers) != 0:
            changeLayers = self.cacheLayers
        else:
            changeLayers = self.layers
        for layer in changeLayers:
            if layer.active and layer.type != 'Input':
                if not layer.frozen:
                    if self._quickprop or self.splitEpsilon:
                        layer.dweight = self.deltaWeight(self.epsilon,
                                                         layer.wed,
                                                         self.momentum,
                                                         layer.dweight,
                                                         layer.wedLast,
                                                         layer.weight,
                                                         self.numConnects(layer.name))
                    else:
                        layer.dweight = self.epsilon * layer.wed + self.momentum * layer.dweight
                    layer.weight += layer.dweight
                    #print "layer.wed = ",layer.wed
                    #print "layer.weight = ",layer.weight," layer.dweight = ",layer.dweight
                    layer.wedLast = Numeric.array(layer.wed) # make copy
                    if self._quickprop:
                        layer.wed = layer.weight * self.decay # reset to last weight, with decay
                    else:
                        layer.wed = layer.wed * 0.0 # keep same numeric type, just zero it
                    dw_count += len(layer.dweight)
                    dw_sum += Numeric.add.reduce(abs(layer.dweight))
        if len(self.cacheConnections) != 0:
            changeConnections = self.cacheConnections
        else:
            changeConnections = self.connections
        for connection in changeConnections:
            if (connection.active
                and connection.fromLayer.active
                and connection.toLayer.active
                and not connection.frozen):
                toLayer = connection.toLayer
                if self._quickprop or self.splitEpsilon:
                    # doing it one vector at a time, to match layer bias training (a quickprop abstraction)
                    for i in range(len(connection.dweight)):
                        Numeric.put(connection.dweight[i],
                                    Numeric.arange(len(connection.dweight[i])),
                                    self.deltaWeight(self.epsilon,
                                                     connection.wed[i],
                                                     self.momentum,
                                                     connection.dweight[i],
                                                     connection.wedLast[i],
                                                     connection.weight[i],
                                                     self.numConnects(connection.toLayer.name)))
                else:
                    connection.dweight = self.epsilon * connection.wed + self.momentum * connection.dweight
                connection.weight += connection.dweight
                #print "connection.wed = ",connection.wed
                #print "connection.weight = ",connection.weight," connection.dweight = ",connection.dweight
                # reset values:
                connection.wedLast = Numeric.array(connection.wed) # make copy
                if self._quickprop:
                    connection.wed = connection.weight * self.decay 
                else:
                    connection.wed = connection.wed * 0.0 # keeps the same Numeric type, but makes it zero
                # get some stats
                dw_count += Numeric.multiply.reduce(connection.dweight.shape)
                dw_sum += Numeric.add.reduce(Numeric.add.reduce(abs(connection.dweight)))
        if self.verbosity >= 1:
            print("WEIGHTS CHANGED")
            if self.verbosity > 2:
                self.display()
        return (dw_count, dw_sum)

    def errorFunction(self, t, a):
        """
        Using a hyperbolic arctan on the error slightly exaggerates
        the actual error non-linearly. Return t - a to just use the difference.
        t - target vector
        a - activation vector
        """
        def difference(v):
            if not self.hyperbolicError:
                #if -0.1 < v < 0.1: return 0.0
                #else:
                return v
            else:
                if   v < -0.9999999: return -17.0
                elif v >  0.9999999: return  17.0
                else: return math.log( (1.0 + v) / (1.0 - v) )
                #else: return Numeric.arctanh(v) # half that above
        return list(map(difference, t - a))

    def ce_init(self):
        """
        Initializes error computation. Calculates error for output
        layers and initializes hidden layer error to zero.
        """
        retval = 0.0; correct = 0; totalCount = 0
        for layer in self.layers:
            if layer.active:
                if layer.type == 'Output':
                    layer.error = self.errorFunction(layer.target, layer.activation) 
                    totalCount += layer.size
                    retval += Numeric.add.reduce((layer.target - layer.activation) ** 2)
                    correct += Numeric.add.reduce(Numeric.fabs(layer.target - layer.activation) < self.tolerance)
                elif (layer.type == 'Hidden'):
                    for i in range(layer.size): # do it this way so you don't break reference links
                        layer.error[i] = 0.0
        return (retval, correct, totalCount)
    def compute_error(self, **args):
        """
        Computes error for all non-output layers backwards through all
        projections.
        """
        for key in args:
            layer = self.getLayer(key)
            if layer.kind == 'Output':
                self.copyTargets(layer, args[key])
        self.verifyTargets() # better have targets set
        error, correct, total = self.ce_init()
        pcorrect = {}
        # go backwards through each proj but don't redo output errors!
        if len(self.cacheConnections) != 0:
            changeConnections = self.cacheConnections
        else:
            changeConnections = self.connections
        for connect in reverse(changeConnections):
            if connect.active and connect.toLayer.active and connect.fromLayer.active:
                connect.toLayer.delta = (connect.toLayer.error *
                                         (self.ACTPRIME(connect.toLayer.activation)))
                connect.fromLayer.error = connect.fromLayer.error + \
                                          Numeric.matrixmultiply(connect.weight, connect.toLayer.delta)
        # now all errors are set on all layers!
        pcorrect = self.getLayerErrors()
        return (error, correct, total, pcorrect)
    def getError(self, *layerNames):
        totalSquaredError, totalSize, totalCorrect, pcorrect = 0.0, 0, 0, {}
        for layerName in layerNames:
            layer = self.layersByName[layerName]
            totalSquaredError += reduce(operator.add, [v ** 2 for v in (layer.target - layer.activation)])
            totalSize += len(layer.error)
            totalCorrect += reduce(operator.add, [abs(d) < self.tolerance for d in layer.target - layer.activation])
            if layer.patternReport: 
                lyst = [0,0,0,0]  # correct, total, pcorrect, ptotal
                lyst[0] = Numeric.add.reduce(Numeric.fabs(layer.target - layer.activation) < self.tolerance)
                lyst[1] = layer.size
                if self.compare(layer.target, layer.activation): # if the same
                    lyst[2] += 1 # correct
                lyst[3] += 1 # ptotal
                pcorrect[layer.name] = lyst
        return (totalSquaredError, totalCorrect, totalSize, pcorrect)
    def getLayerErrors(self):
        pcorrect = {}
        for layer in self.layers:
            if layer.patternReport: 
                lyst = [0,0,0,0]  # correct, total, pcorrect, ptotal
                lyst[0] = Numeric.add.reduce(Numeric.fabs(layer.target - layer.activation) < self.tolerance)
                lyst[1] = layer.size
                if self.compare(layer.target, layer.activation): # if the same
                    lyst[2] += 1 # correct
                lyst[3] += 1 # ptotal
                pcorrect[layer.name] = lyst
        return pcorrect
    def compute_wed(self):
        """
        Computes weight error derivative for all connections in
        self.connections starting with the last connection.
        """
        if len(self.cacheConnections) != 0:
            changeConnections = self.cacheConnections
        else:
            changeConnections = self.connections            
        for connect in reverse(changeConnections):
            if connect.active and connect.fromLayer.active and connect.toLayer.active:
                connect.wed = connect.wed + Numeric.outerproduct(connect.fromLayer.activation,
                                                                 connect.toLayer.delta)
        if len(self.cacheLayers) != 0:
            changeLayers = self.cacheLayers
        else:
            changeLayers = self.layers
        for layer in changeLayers:
            if layer.active:
                layer.wed = layer.wed + layer.delta

    def diff(self, value):
        """
        Returns value to within 0.001. Then returns 0.0.
        """
        if math.fabs(value) < 0.001:
            return 0.0
        else:
            return value

    # display and string methods
    def toString(self):
        """
        Returns the network layers as a string.
        """
        output = ""
        for layer in reverse(self.layers):
            output += layer.toString()
        return output
    def prompt(self):
        s = input("<q>uit, <g>o, or press Enter: ")
        if s is None or s == "q":
            sys.exit(1)
        elif s == 'g':
            self.interactive = 0
        else:
            return
    def displayConnections(self, title = "Connections"):
        fromColWidth    = 8
        decimals        = 2
        colWidth        = 7
        print(title + ":")
        print("-" * (len(title) + 1))
        for c in self.connections:
            print("Weights (%s, %s) from '%s' (%s, %s) to '%s' (%s, %s):" % (
                ["not frozen","frozen"][c.frozen],
                ["not active","active"][c.active],
                c.fromLayer.name, ["not frozen","frozen"][c.fromLayer.frozen], ["not active","active"][c.fromLayer.active], 
                c.toLayer.name, ["not frozen","frozen"][c.toLayer.frozen], ["not active","active"][c.toLayer.active] ))
            # top bar
            line = ("=" * fromColWidth) + "="
            line += ("=" * colWidth) + "="
            for i in range(c.toLayer.size):
                line += ("=" * colWidth) + "="
            print(line)
            # to layer name:
            line = " " * fromColWidth + "|" + pad(c.toLayer.name, (colWidth * (c.toLayer.size + 1)) + c.toLayer.size, align="center", )
            print(line)
            # sep bar:
            line = ("-" * fromColWidth) + "+"
            line += ("-" * colWidth) + "+"
            for i in range(c.toLayer.size):
                line += ("-" * colWidth) + "+"
            print(line)
            # col header:
            line = pad(c.fromLayer.name, fromColWidth, align="center")
            line += pad("bias", colWidth, align="center")
            for i in range(c.toLayer.size):
                line += pad(str(i), colWidth, align = "center")
            print(line)
            # sep bar:
            line = ("-" * fromColWidth) + "+"
            line += ("-" * colWidth) + "+"
            for i in range(c.toLayer.size):
                line += ("-" * colWidth) + "+"
            print(line)
            # biases, to layer:
            line = pad("bias", fromColWidth, align = "center")
            line += (" " * colWidth) + "|"
            for j in range(c.toLayer.size):
                line += pad(("%." + str(decimals) + "f") % c.toLayer.weight[j], colWidth, align = "right")
            print(line)
            # weights:
            for i in range(c.fromLayer.size):
                line = pad(str(i), fromColWidth, align = "center")
                line += pad(("%." + str(decimals) + "f") % c.fromLayer.weight[i], colWidth, align = "right")
                for j in range(c.toLayer.size):
                    line += pad(("%." + str(decimals) + "f") % c.weight[i][j], colWidth, align = "right")
                print(line)
            # bottom bar:
            line = ("=" * fromColWidth) + "="
            line += ("=" * colWidth) + "="
            for i in range(c.toLayer.size):
                line += ("=" * colWidth) + "="
            print(line)
                
    def display(self):
        """
        Displays the network to the screen.
        """
        print("Display network '" + self.name + "':")
        size = list(range(len(self.layers)))
        size.reverse()
        for i in size:
            if self.layers[i].active:
                self.layers[i].display()
                if self.patterned and self.layers[i].type != 'Hidden':
                    targetWord, diff = self.getWord( self.layers[i].target, returnDiff = 1)
                    if self.layers[i].kind == 'Output':
                        if targetWord == None:
                            print("Target Pattern   = %s" % "No match")
                        else:
                            if diff == 0.0:
                                print("Target Pattern   = '%s'; (exact)" % targetWord)
                            else:
                                print("Target Pattern   = '%s'; Match difference = %f)" % (targetWord, diff))
                    actWord, diff = self.getWord( self.layers[i].activation, returnDiff = 1 )
                    if (self.layers[i].kind == 'Input' or self.layers[i].kind == 'Output'):
                        if actWord == None:
                            print("Matching Pattern = %s" % "No match")
                        else:
                            if diff == 0.0:
                                print("Matching Pattern = '%s'; (exact)" % actWord)
                            else:
                                print("Matching Pattern = '%s'; Match difference = %f" % (actWord, diff))
                if self.verbosity >= 1:
                    weights = list(range(len(self.connections)))
                    weights.reverse()
                    for j in weights:
                        if self.connections[j].toLayer.name == self.layers[i].name:
                            self.connections[j].display()
                            
    # GA support
    def arrayify(self):
        """
        Returns an array of node bias values and connection weights
        for use in a GA.
        """
        gene = []
        for layer in self.layers:
            if layer.type != 'Input':
                for i in range(layer.size):
                    gene.append( layer.weight[i] )
        for connection in self.connections:
            for i in range(connection.fromLayer.size):
                for j in range(connection.toLayer.size):
                    gene.append( connection.weight[i][j] )
        return gene
    def unArrayify(self, gene):
        """
        Copies gene bias values and weights to network bias values and
        weights.
        """
        g = 0
        # if gene is too small an IndexError will be thrown
        for layer in self.layers:
            if layer.type != 'Input':
                for i in range(layer.size):
                    layer.weight[i] = float( gene[g])
                    g += 1
        for connection in self.connections:
            for i in range(connection.fromLayer.size):
                for j in range(connection.toLayer.size):
                    connection.weight[i][j] = gene[g]
                    g += 1
        # if gene is too long we may have a problem
        if len(gene) > g:
            raise IndexError('Argument to unArrayify is too long.', len(gene))

    # file IO 
    def logMsg(self, layerName, message):
        """
        Logs a message with layerName log.
        """
        # will throw an exception if setLog not called
        self.getLayer(layerName).logMsg(message)
    def logLayer(self, layerName, fileName):
        """
        Sets the layerName's log feature.
        """
        self.getLayer(layerName).setLog(fileName)
    def closeLog(self, layerName):
        """
        Close the layerName's log file.
        """
        self.getLayer(layerName).closeLog()

    def saveWeightsToFile(self, filename, mode='pickle', counter=None):
        """
        Deprecated. Use saveWeights instead.
        """
        self.saveWeights(filename, mode, counter)

    def saveWeights(self, filename, mode='pickle', counter=None):
        """
        Saves weights to file in pickle, plain, or tlearn mode.
        """
        # modes: pickle/conx, plain, tlearn
        if "?" in filename: # replace ? pattern in filename with epoch number
            import re
            char = "?"
            match = re.search(re.escape(char) + "+", filename)
            if match:
                num = self.epoch
                if counter != None:
                    num = counter
                elif self.totalEpoch != 0: # use a total epoch, if one:
                    num = self.totalEpoch
                fstring = "%%0%dd" % len(match.group())
                filename = filename[:match.start()] + \
                           fstring % self.epoch + \
                           filename[match.end():]
                self.lastAutoSaveWeightsFilename = filename
        if mode == 'pickle':
            mylist = self.arrayify()
            import pickle
            fp = open(filename, "w")
            pickle.dump(mylist, fp)
            fp.close()
        elif mode in ['plain', 'conx']:
            fp = open(filename, "w")
            fp.write("# Biases\n")
            for layer in self.layers:
                if layer.type != 'Input':
                    fp.write("# Layer: " + layer.name + "\n")
                    for i in range(layer.size):
                        fp.write("%f " % layer.weight[i] )
                    fp.write("\n")
            fp.write("# Weights\n")
            for connection in self.connections:
                fp.write("# from " + connection.fromLayer.name + " to " +
                         connection.toLayer.name + "\n")
                for i in range(connection.fromLayer.size):
                    for j in range(connection.toLayer.size):
                        fp.write("%f " % connection.weight[i][j] )
                    fp.write("\n")
            fp.close()
        elif mode == 'tlearn':
            fp = open(filename, "w")
            fp.write("NETWORK CONFIGURED BY TLEARN\n")
            fp.write("# weights after %d sweeps\n" % self.epoch)
            fp.write("# WEIGHTS\n")
            cnt = 1
            for lto in self.layers:
                if lto.type != 'Input':
                    for j in range(lto.size):
                        fp.write("# TO NODE %d\n" % cnt)
                        fp.write("%f\n" % lto.weight[j] )
                        for lfrom in self.layers:
                            try:
                                connection = self.getConnection(lfrom.name,lto.name)
                                for i in range(connection.fromLayer.size):
                                    fp.write("%f\n" % connection.weight[i][j])
                            except NetworkError: # should return an exception here
                                for i in range(lfrom.size):
                                    fp.write("%f\n" % 0.0)
                        cnt += 1
            fp.close()            
        else:
            raise ValueError('Unknown mode in saveWeights().', mode)

    def loadWeightsFromFile(self, filename, mode='pickle'):
        """
        Deprecated. Use loadWeights instead.
        """
        self.loadWeights(filename, mode)

    def loadWeights(self, filename, mode='pickle'):
        """
        Loads weights from a file in pickle, plain, or tlearn mode.
        """
        # modes: pickle, plain/conx, tlearn
        if mode == 'pickle':
            import pickle
            fp = open(filename, "r")
            mylist = pickle.load(fp)
            fp.close()
            self.unArrayify(mylist)
        elif mode in ['plain', 'conx']:
            arr = []
            fp = open(filename, "r")
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line == '' or line[0] == '#':
                    pass
                else:
                    data = list(map( float, line.split()))
                    arr.extend( data )
            self.unArrayify( arr )
            fp.close()
        elif mode == 'tlearn':
            fp = open(filename, "r")
            fp.readline() # NETWORK CONFIGURED BY
            fp.readline() # # weights after %d sweeps
            fp.readline() # # WEIGHTS
            cnt = 1
            for lto in self.layers:
                if lto.type != 'Input':
                    for j in range(lto.size):
                        fp.readline() # TO NODE %d
                        lto.weight[j] = float(fp.readline())
                        for lfrom in self.layers:
                            try:
                                connection = self.getConnection(lfrom.name, lto.name)
                                for i in range(connection.fromLayer.size):
                                    connection.weight[i][j] = float( fp.readline() )
                            except NetworkError:
                                for i in range(lfrom.size):
                                    # 0.0
                                    fp.readline()
                        cnt += 1
            fp.close()            
        elif mode == 'nbench':
            # reads weights and constructs network
            fp = open(filename, "r")
            line = fp.readline()
            while line[:8] != "Ninputs:":
                line = fp.readline()
            itxt, inputs, ttxt, total, otxt, outputs = line.split()
            inputs, total, outputs = int(inputs), int(total), int(outputs)
            # cascor's total value is the bias + inputs + hiddens
            hiddens = total - inputs - 1
            # create the layers:
            self.addLayer("input", inputs)
            for i in range(hiddens):
                self.addLayer("hidden%d" % i, 1)
            self.addLayer("output", outputs)
            # connect input to all hiddens, output:
            for i in range(hiddens):
                self.connect("input", "hidden%d" % i)
            self.connect("input", "output")
            # connect all hiddens to all later hiddens:
            for i in range(hiddens - 1):
                for j in range(i + 1, hiddens):
                    if j != i:
                        self.connect("hidden%d" % i, "hidden%d" % j )
            # connect all hiddens to outputs:
            for i in range(hiddens):
                self.connect("hidden%d" % i, "output")
            # now, let's set the weights:
            for outcount in range(outputs):
                while line[:9] != "# Output:":
                    line = fp.readline()
                line = fp.readline() # $
                line = fp.readline() # bias, input to output, hidden to output?
                data = ""
                while line and line[0] != "#":
                    data += " " + line.strip()
                    line = fp.readline()
                weights = list(map(float, data.split()))
                self["output"].weight[outcount] = weights[0] # bias
                next = 1
                for i in range(self["input"].size):
                    self["input", "output"].weight[i][outcount] = weights[next]
                    next += 1
                for h in range(hiddens):
                    for i in range(self["hidden%d" % h].size): # normally just 1
                        self["hidden%d" % h, "output"].weight[i][outcount] = weights[next]
                        next += 1
            # now, for each hidden "layer":
            while line and line[0] != "$":
                line = fp.readline()
            line = fp.readline()
            for hidcount in range(hiddens):
                weights = []
                while line and line[0] != "$" and line[0] != "#": # next line is a weight line
                    weights.extend( list(map(float, line.split()))) # bias, input to hidden, hidden to hidden?
                    line = fp.readline() 
                self[("hidden%d" % hidcount)].weight[0] = weights[0] # bias first
                next = 1
                for i in range(self["input"].size):
                    for j in range(self["hidden%d" % hidcount].size): # normally just 1
                        self["input", ("hidden%d" % hidcount)].weight[i][j] = weights[next]
                        next += 1
                for h in range(hidcount): # all those hids leading up to this one
                    for i in range(self["hidden%d" % h].size): # normally just 1
                        for j in range(self["hidden%d" % hidcount].size): # normally just 1
                            self[("hidden%d" % h), ("hidden%d" % hidcount)].weight[i][j] = weights[next]
                            next += 1
                line = fp.readline() # $
                line = fp.readline() # beginning of weights
        else:
            raise ValueError('Unknown mode in loadWeights()', mode)

    def saveNetwork(self, filename, makeWrapper = 1, mode = "pickle", counter = None):
        """
        Saves network to file using pickle.
        """
        self.saveNetworkToFile(filename, makeWrapper, mode, counter)

    def saveNetworkToFile(self, filename, makeWrapper = 1, mode = "pickle", counter = None):
        """
        Deprecated.
        """
        if "?" in filename: # replace ? pattern in filename with epoch number
            import re
            char = "?"
            match = re.search(re.escape(char) + "+", filename)
            if match:
                num = self.epoch
                if counter != None:
                    num = counter
                elif self.totalEpoch != 0: # use a total epoch, if one:
                    num = self.totalEpoch
                fstring = "%%0%dd" % len(match.group())
                filename = filename[:match.start()] + \
                           fstring % num + \
                           filename[match.end():]
                self.lastAutoSaveNetworkFilename = filename
        if mode == "pickle":
            # dump network via pickle:
            import pickle
            basename = filename.split('.')[0]
            filename += ".pickle"
            fp = open(filename, 'w')
            pickle.dump(self, fp)
            fp.close()
            # make wrapper python file:
            if makeWrapper:
                fp = open(basename + ".py", "w")
                fp.write("from pyrobot.brain.conx import *\n")
                fp.write("import pickle\n")
                fp.write("fp = open('%s', 'r')\n" % filename)
                fp.write("network = pickle.load(fp)")
                fp.close()
            # give some help:
            print("To load network:")
            print("   %% python -i %s " % (basename + ".py"))
            print("   >>> network.train() # for example")
            print("--- OR ---")
            print("   % python")
            print("   >>> from pyrobot.brain.conx import *")
            print("   >>> network = loadNetwork(%s)" % filename)
            print("   >>> network.train() # for example")
        elif mode in ["plain", "conx"]:
            fp = open(filename, "w")
            fp.write("network, %s\n" % (self.__class__.__name__))
            for layer in self.layers:
                fp.write("layer, %s, %s\n" % (layer.name, layer.size))
                # biases:
                for i in range(layer.size):
                    fp.write("%f " % layer.weight[i])
                fp.write("\n")
            for connection in self.connections:
                fp.write("connection, %s, %s\n" %(connection.fromLayer.name, connection.toLayer.name))
                # weights:
                for i in range(connection.fromLayer.size):
                    for j in range(connection.toLayer.size):
                        fp.write("%f " % connection.weight[i][j])
                    fp.write("\n")
            fp.close()

    def loadVectors(self, filename, cols = None, everyNrows = 1,
                    delim = ' ', checkEven = 1, patterned = 0):
        """
        Load a set of vectors from a file. Takes a filename, list of cols
        you want (or None for all), get every everyNrows (or 1 for no
        skipping), and a delimeter.
        """
        return self.loadVectorsFromFile(filename, cols, everyNrows,
                                        delim, checkEven, patterned)

    def loadVectorsFromFile(self, filename, cols = None, everyNrows = 1,
                            delim = ' ', checkEven = 1, patterned = 0):
        """
        Deprecated.
        """
        fp = open(filename, "r")
        line = fp.readline()
        lineno = 0
        lastLength = None
        data = []
        while line:
            if lineno % everyNrows == 0:
                if patterned:
                    linedata1 = [x for x in line.strip().split(delim)]
                else:
                    linedata1 = [float(x) for x in line.strip().split(delim)]
            else:
                lineno += 1
                line = fp.readline()
                continue
            if cols == None: # get em all
                newdata = linedata1
            else: # just get some cols
                newdata = []
                for i in cols:
                    newdata.append( linedata1[i] )
            if lastLength == None or (not checkEven) or (checkEven and len(newdata) == lastLength):
                data.append( newdata )
            else:
                raise ValueError("Data Format Error: line = %d, data = %s" % (lineno, newdata))
            lastLength = len(newdata)
            lineno += 1
            line = fp.readline()    
        fp.close()
        return data

    def loadInputPatterns(self, filename, cols = None, everyNrows = 1,
                          delim = ' ', checkEven = 1):
        """
        Loads inputs as patterns from file.
        """
        self.loadInputPatternsFromFile(filename, cols, everyNrows,
                                       delim, checkEven)

    def loadInputPatternsFromFile(self, filename, cols = None, everyNrows = 1,
                                  delim = ' ', checkEven = 1):
        """
        Deprecated.
        """
        self.inputs = self.loadVectors(filename, cols, everyNrows, delim, checkEven, patterned = 1)
        self.loadOrder = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            self.loadOrder[i] = i

    def loadInputs(self, filename, cols = None, everyNrows = 1,
                   delim = ' ', checkEven = 1):
        """
        Loads inputs from file. Patterning is lost.
        """
        self.loadInputsFromFile(filename, cols, everyNrows,
                                delim, checkEven)

    def loadInputsFromFile(self, filename, cols = None, everyNrows = 1,
                           delim = ' ', checkEven = 1):
        """
        Depricated.
        """
        self.inputs = self.loadVectors(filename, cols, everyNrows, delim, checkEven)
        self.loadOrder = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            self.loadOrder[i] = i

    def saveInputs(self, filename):
        """
        Saves inputs to file.
        """
        self.saveInputsToFile(filename)

    def saveInputsToFile(self, filename):
        """
        Deprecated.
        """
        fp = open(filename, 'w')
        for input in self.inputs:
            vec = self.replacePatterns(input)
            for item in vec:
                fp.write("%f " % item)
            fp.write("\n")

    def loadTargets(self, filename, cols = None, everyNrows = 1,
                            delim = ' ', checkEven = 1):
        """
        Loads targets from file.
        """
        self.loadTargetsFromFile(filename, cols, everyNrows,
                                 delim, checkEven)

    def loadTargetsFromFile(self, filename, cols = None, everyNrows = 1,
                            delim = ' ', checkEven = 1):
        """
        Loads targets from file.
        """
        self.targets = self.loadVectors(filename, cols, everyNrows,
                                        delim, checkEven)

    def loadTargetPatterns(self, filename, cols = None, everyNrows = 1,
                            delim = ' ', checkEven = 1):
        """
        Loads targets as patterns from file.
        """
        self.loadTargetPatternssFromFile(filename, cols, everyNrows,
                                         delim, checkEven)

    def loadTargetPatternsFromFile(self, filename, cols = None, everyNrows = 1,
                                   delim = ' ', checkEven = 1):
        """
        Deprecated.
        """
        self.targets = self.loadVectors(filename, cols, everyNrows,
                                        delim, checkEven, patterned=1)

    def saveTargets(self, filename):
        """
        Saves targets to file.
        """
        self.saveTargetsToFile(filename)

    def saveTargetsToFile(self, filename):
        """
        Deprecated.
        """
        fp = open(filename, 'w')
        for target in self.targets:
            vec = self.replacePatterns(target)
            for item in vec:
                fp.write("%f " % item)
            fp.write("\n")

    def saveData(self, filename):
        """
        Saves data (targets/inputs) to file.
        """
        self.saveDataToFile(filename)

    def saveDataToFile(self, filename):
        """
        Deprecated.
        """
        fp = open(filename, 'w')
        for i in range(len(self.inputs)):
            try:
                vec = self.replacePatterns(self.inputs[i])
                for item in vec:
                    fp.write("%f " % item)
            except:
                pass
            try:
                vec = self.replacePatterns(self.targets[i])
                for item in vec:
                    fp.write("%f " % item)
            except:
                pass
            fp.write("\n")

    def loadData(self, filename, ocnt = -1):
        """
        Loads data (targets/inputs) from file.
        """
        self.loadDataFromFile(filename, ocnt)

    def loadDataFromFile(self, filename, ocnt = -1):
        """
        Deprecated.
        """
        if ocnt == -1:
            ocnt = int(self.layers[len(self.layers) - 1].size)
        fp = open(filename, 'r')
        line = fp.readline()
        self.targets = []
        self.inputs = []
        while line:
            data = list(map(float, line.split()))
            cnt = len(data)
            icnt = cnt - ocnt
            self.inputs.append(self.patternVector(data[0:icnt]))
            self.targets.append(self.patternVector(data[icnt:]))
            line = fp.readline()
        self.loadOrder = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            self.loadOrder[i] = i

    # patterning
    def lookupPattern(self, name, layer):
        """ See if there is a name/layer pattern combo, else return the name pattern. """
        if (name, layer) in self.patterns:
            return self.patterns[(name, layer)]
        else:
            return self.patterns[name]
    def replacePatterns(self, vector, layer = None):
        """
        Replaces patterned inputs or targets with activation vectors.
        """
        if not self.patterned: return vector
        if type(vector) == str:
            return self.replacePatterns(self.lookupPattern(vector, layer), layer)
        elif type(vector) != list:
            return vector
        # should be a vector if we made it here
        vec = []
        for v in vector:
            if type(v) == str:
                retval = self.replacePatterns(self.lookupPattern(v, layer), layer)
                if type(retval) == list:
                    vec.extend( retval )
                else:
                    vec.append( retval )
            else:
                vec.append( v )
        return vec
    def patternVector(self, vector):
        """
        Replaces vector with patterns. Used for loading inputs or
        targets from a file and still preserving patterns.
        """
        if not self.patterned: return vector
        if type(vector) == int:
            if self.getWord(vector) != '':
                return self.getWord(vector)
            else:
                return vector
        elif type(vector) == float:
            if self.getWord(vector) != '':
                return self.getWord(vector)
            else:
                return vector
        elif type(vector) == str:
            return vector
        elif type(vector) == list:
            if self.getWord(vector) != '':
                return self.getWord(vector)
        # should be a list
        vec = []
        for v in vector:
            if self.getWord(v) != '':
                retval = self.getWord(v)
                vec.append( retval )
            else:
                retval = self.patternVector(v)
                vec.append( retval )                
        return vec
    def setPatterns(self, patterns):
        """
        Sets patterns to the dictionary argument.

        Example: net.setPatterns( {"tom": [0, 0, 0, 1], "mary": [1, 0, 0, 0]} )

        Sets net.patterned to 1 as a side-effect.
        
        """
        self.patterns = patterns
        self.patterned = 1
    def getPattern(self, word):
        """
        Returns the pattern with key word.

        Example: net.getPattern("tom") => [0, 0, 0, 1]
        
        """
        if word in self.patterns:
            return self.patterns[word]
        else:
            raise ValueError('Unknown pattern in getPattern().', word)
    def getWord(self, pattern, returnDiff = 0):
        """
        Returns the word associated with pattern.

        Example: net.getWord([0, 0, 0, 1]) => "tom"

        This method now returns the closest pattern based on distance.
        """
        minDist = 10000
        closest = None
        for w in self.patterns:
            # There may be some patterns that are scalars; we don't search
            # those in this function:
            if type(self.patterns[w]) in [int, float, int]: continue
            if len(self.patterns[w]) == len(pattern):
                dist = reduce(operator.add, [(a - b) ** 2 for (a,b) in zip(self.patterns[w], pattern )])
                if dist == 0.0:
                    if returnDiff:
                        return w, dist
                    else:
                        return w
                if dist < minDist:
                    minDist = dist
                    closest = w
        if returnDiff:
            return closest, minDist
        else:
            return closest
    # use addPattern and delPattern
    def setPattern(self, word, vector):
        """
        Sets a pattern with key word. Better to use addPattern() and
        delPattern().

        Example: net.setPattern("tom", [1, 0, 0, 0])

        """
        self.patterns[word] = vector
    def addPattern(self, word, vector):
        """
        Adds a pattern with key word.

        Example: net.addPattern("tom", [0, 0, 0, 1])
        
        """
        if word in self.patterns:
            raise NetworkError('Pattern key already in use. Call delPattern to free key.', word)
        else:
            self.patterns[word] = vector
    # will raise KeyError if word is not in dict
    def delPattern(self, word):
        """
        Delete a pattern with key word.

        Example: net.delPattern("tom")
        
        """
        del self.patterns[word]
    def compare(self, v1, v2):
        """
        Compares two values. Returns 1 if all values are withing
        self.tolerance of each other.
        """
        try:
            if len(v1) != len(v2): return 0
            for x, y in zip(v1, v2):
                if abs( x - y ) > self.tolerance:
                    return 0
            return 1
        except:
            # some patterns may not be vectors
            try:
                if abs( v1 - v2 ) > self.tolerance:
                    return 0
                else:
                    return 1
            except:
                return 0
    def shareWeights(self, network, listOfLayerNamePairs = None):
        """
        Share weights with another network. Connection
        is broken after a randomize or change of size. Layers must have the same
        names and sizes for shared connections in both networks.

        Example: net.shareWeights(otherNet, [["hidden", "output"]])

        This example will take the weights between the hidden and output layers
        of otherNet and share them with net. Also, the bias values of
        otherNet["output"] will be shared with net["output"].

        If no list is given, will share all weights.
        """
        if listOfLayerNamePairs == None:
            listOfLayerNamePairs = []
            for c in self.connections:
                listOfLayerNamePairs.append( [c.fromLayer.name, c.toLayer.name] )
        if self.verbosity > 1:
            print("sharing weights:", self.name, listOfLayerNamePairs)
        # first, check to see if this will work:
        count = 0
        for (fromLayerName, toLayerName) in listOfLayerNamePairs:
            for c1 in range(len(self.connections)):
                if self.connections[c1].fromLayer.name == fromLayerName and \
                       self.connections[c1].toLayer.name == toLayerName:
                    for c2 in range(len(network.connections)):
                        if network.connections[c2].fromLayer.name == fromLayerName and \
                               network.connections[c2].toLayer.name == toLayerName:
                            if (self.connections[c1].fromLayer.size != network.connections[c2].fromLayer.size) or \
                               (self.connections[c1].toLayer.size != network.connections[c2].toLayer.size):
                                raise AttributeError("shareSomeWeights: layer sizes did not match")
                            count += 1
        if count != len(listOfLayerNamePairs):
            raise AttributeError("shareSomeWeights: layer names did not match")
        # ok, now let's share!
        self.sharedWeights = 1
        network.sharedWeights = 1
        for (fromLayerName, toLayerName) in listOfLayerNamePairs:
            for c1 in range(len(self.connections)):
                if self.connections[c1].fromLayer.name == fromLayerName and \
                       self.connections[c1].toLayer.name == toLayerName:
                    for c2 in range(len(network.connections)):
                        if network.connections[c2].fromLayer.name == fromLayerName and \
                               network.connections[c2].toLayer.name == toLayerName:
                            self.connections[c1].weight = network.connections[c2].weight
        for (fromLayerName, toLayerName) in listOfLayerNamePairs:
            for l1 in range(len(self.layers)):
                if self.layers[l1].name == toLayerName:
                    for l2 in range(len(network.layers)):
                        if network.layers[l2].name == toLayerName:
                            self.layers[l1].weight = network.layers[l2].weight

    def testGeneralization(self, incr = .1, start = 0, stop = 1, inputLayer = "input",
                           inputs = [0, 1], outputLayer = "output", outputs = None, sum = 0):
        """
        sum - use for SigmaNetworks
        FIXME: currently just works with one output layer
        """
        print("Testing Generalization:", self.name)
        resolution = int((stop - start) / incr)
        retString = ""
        for x in range(resolution):
            row = ""
            if sum:
                size = 1
            else:
                size = self[outputLayer].size
                for i in range(size):
                    for y in range(resolution):
                        input = (x/float(resolution), y/float(resolution))
                        results = self.propagate(input = input)
                        if sum:
                            retval = reduce(operator.add, self[outputLayer].activation) / self[outputLayer].size
                        else:
                            retval = results[i]
                        c = round(retval, 1)
                        if c > 0.95:
                            c = "#"
                        elif c < 0.05:
                            c = "."
                        else:
                            c = str(c * 10)[0]
                        row += "%s" % c
                    row += "   "
                retString += row + "\n"
        return retString
    # --------------------------------------- Network Properties
    maxRandom = property(getMaxRandom, setMaxRandom)
    verbosity = property(getVerbosity, setVerbosity)
    quickprop = property(getQuickprop, setQuickprop)

class SigmaNetwork(Network):
    """
    Uses CRBP to train a population encoded summation on output layers. 
    """
    def __init__(self, name = 'Sigma Network', verbosity = 0):
        Network.__init__(self, name, verbosity)
        self.times = 0
        self.outputLayers = None
        self.sigmaCorrect = 0        
    def initialize(self):
        Network.initialize(self)
        self.sigmaCorrect = 0        
    def preSweep(self):
        self.sigmaCorrect = 0
    def preBackprop(self, **dict):
        # could use a probabilistic version of round:
        # def prob(v): return int(random.random() < v)
        def myround(v):
            retval = round(v)
            return retval
        def mutate(v, p):
            for i in range(int(round(len(v) * p))):
                g = int(len(v) * random.random())
                symbols = [0.0,
                           1.0]
                v[g] = symbols[(symbols.index(v[g]) + 1) % 2]
            return v
        for name in dict:
            layer = self[name]
            if layer.kind == "Output":
                vector = list(map(myround, self[name].activation))
                score = abs(sum(vector)/float(self[name].size) - dict[name][0])
                if score > self.tolerance:
                    vector = mutate(vector, score)
                else:
                    self.sigmaCorrect += 1
                dict[name] = vector
        return dict
    def doWhile(self, *args):
        if self.outputLayers == None:
            self.outputLayers = 0
            for layer in self.layers:
                if layer.kind == "Output":
                    self.outputLayers += len(self.inputs)
        return self.sigmaCorrect != self.outputLayers

#=================================================================================
# newConx
# - simpler interface for teaching neural network concepts
# - graphical support for BackpropNetwork and SRN classes
#=================================================================================

import os
#from pyrobot.pyrographics import GraphWin, Line, Point, color_rgb, Text

# support functions

def validateShape(values, shape, default):
    if shape is 0: shape = default
    if shape is None:
        shape = (1, len(values))
    elif type(shape) is int and 0 < shape <= len(values):
        cols = shape
        rows = int(math.ceil(float(len(values))/shape))
        shape = (rows, cols)
    elif type(shape) is tuple and len(shape) == 1:
        shape = (1, shape[0])
    assert type(shape) is tuple and len(shape) == 2, 'invalid shape: %s' % (shape,)
    (rows, cols) = shape
    assert rows * cols == len(values), \
           "can't display %d values with shape %s" % (len(values), shape)
    return shape

def validateScale(scale, default):
    if scale is 0: scale = default
    if scale is None: scale = 10
    assert type(scale) is int and scale > 0, 'invalid scale: %s' % scale
    return scale

def pretty(values, max=0, places=2):
    code = '%%.%df' % places
    if max > 0 and len(values) > max:
        return ' '.join([code % v for v in values[0:max]]) + ' ...'
    else:
        return ' '.join([code % v for v in values])

#------------------------------------------------------------------------------------------
# PGM class is used by the BackpropNetwork methods showInput,
# showPattern, and showPatterns for displaying input patterns

# class PGM(GraphWin):

#     def __init__(self, file=None, values=None, title='', shape=0, scale=0, highlight=None, invert=False):
#         # invert flag displays an inverted version of the image, but does not invert
#         # the actual intensity values
#         if file is None and values is not None:
#             assert type(values) is list and len(values) > 0, 'a list of values is required'
#             for v in values:
#                 assert 0 <= v <= 1, 'image values must be in range 0-1'
#             shape = validateShape(values, shape, None)
#             self.rows, self.cols = shape
#             self.normalized = values
#             self.raw = [int(100*v) for v in values]
#             self.maxval = 100
#         elif file is not None and values is None:
#             assert shape is 0, 'shape is determined by PGM file'
#             f = open(file)
#             pgmType = f.readline().strip()
#             if pgmType != 'P5':
#                 raise IOError, 'file is not a raw PGM file'
#             title = os.path.basename(file)
#             self.cols, self.rows = [int(v) for v in f.readline().split()]
#             self.maxval = int(f.readline().strip())
#             self.raw = [ord(v) for v in f.read()]
#             for v in self.raw:
#                 assert 0 <= v <= self.maxval, 'incorrect PGM file format'
#             self.normalized = [float(v)/self.maxval for v in self.raw]
#             f.close()
#         else:
#             raise AttributeError, 'must specify file=<filename> or values=<vector>'
#         if highlight is not None:
#             assert type(highlight) is int and self.rows == 1, \
#                    'cannot highlight images with more than one row'
#             assert 0 <= highlight < self.cols, 'highlight out of range'
#         scale = validateScale(scale, 10)
#         GraphWin.__init__(self, title, self.cols*scale, self.rows*scale)
#         self.title = title
#         self.rectangles = []
#         for i in xrange(len(self.normalized)):
#             x = (i % self.cols) * scale
#             y = (i / self.cols) * scale
#             grayLevel = int(100 * self.normalized[i])
#             if invert:
#                 grayLevel = 100 - grayLevel
#             fColor = oColor = 'gray%d' % grayLevel
#             if self.rows == 1 or self.cols == 1: oColor = 'black'
#             r = self.create_rectangle(x, y, x+scale, y+scale, outline=oColor, fill=fColor)
#             self.rectangles.append(r)
#         if highlight is not None:
#             x = highlight * scale + 1
#             self.create_rectangle(x, 1, x+scale-1, scale, outline='red')

#     def __str__(self):
#         s = '\ntitle:  %s\n' % self.title
#         s += 'size:   %d rows, %d cols\n' % (self.rows, self.cols)
#         s += 'maxval: %d\n' % self.maxval
#         border = '+%s+\n' % ('-' * (2 * self.cols + 1))
#         s += border
#         palette = ' .,:+*O8@@'
#         for r in xrange(self.rows):
#             s += '| '
#             for c in xrange(self.cols):
#                 i = r * self.cols + c
#                 if i >= len(self.normalized):
#                     s += '  '
#                 else:
#                     s += '%s ' % palette[int(self.normalized[i]*(len(palette)-1))]
#             s = s + '|\n'
#         s += border
#         return s

#     def setTitle(self, title):
#         self.winfo_toplevel().title(title)
#         self.title = title

#     def updateImage(self, newValues, invert=False):
#         # invert flag True displays an inverted version of the image,
#         # but does not invert the intensity values themselves
#         assert len(newValues) == len(self.rectangles), 'wrong number of values'
#         for i in xrange(len(newValues)):
#             assert 0 <= newValues[i] <= 1, 'image values must be in range 0-1'
#             grayLevel = int(100 * newValues[i])
#             if invert:
#                 grayLevel = 100 - grayLevel
#             fColor = oColor = 'gray%d' % grayLevel
#             if self.rows == 1 or self.cols == 1: oColor = 'black'
#             self.itemconfigure(self.rectangles[i], fill=fColor, outline=oColor)
#         self.update_idletasks()
#         while self.tk.dooneevent(2): pass
#         self.normalized = newValues
#         self.raw = [int(100*v) for v in self.normalized]
#         self.maxval = 100

#     def invert(self):
#         """"
#         Sets the intensity values of the image to their inverse.
#         """
#         self.raw = [self.maxval-v for v in self.raw]
#         self.normalized = [float(v)/self.maxval for v in self.raw]
#         for i in xrange(len(self.normalized)):
#             grayLevel = int(100 * self.normalized[i])
#             fColor = oColor = 'gray%d' % grayLevel
#             if self.rows == 1 or self.cols == 1: oColor = 'black'
#             self.itemconfigure(self.rectangles[i], fill=fColor, outline=oColor)
#         self.update_idletasks()
#         while self.tk.dooneevent(2): pass

#     def saveImage(self, pathname):
#         f = open(pathname, mode='w')
#         f.write('P5\n')
#         f.write('%d %d\n' % (self.cols, self.rows))
#         f.write('%d\n' % self.maxval)
#         padding = []
#         if len(self.raw) < self.rows * self.cols:
#             padding = [0] * (self.rows * self.cols - len(self.raw))
#         f.write('%s' % ''.join([chr(v) for v in self.raw + padding]))
#         f.close()

# #------------------------------------------------------------------------------------------
# # NetworkActivationDisplay class displays network activations and connections graphically

# class NetworkActivationDisplay(GraphWin):

#     def __init__(self, net, title, style, horizSpacing, vertSpacing, invert, info):
#         validStyles = ['line', 'lines', 'arrow', 'arrows', 'full', 'fullarrows']
#         assert style in validStyles, 'style must be one of: %s' % ', '.join(validStyles)
#         defaultScale = 30
#         sideMargin = 40
#         topMargin = 30
#         bottomMargin = 20
#         textSize = 14
#         self.bgColor = color_rgb(176, 196, 222)  # 'light steel blue'
#         fgText = 'blue'
#         bgText = self.bgColor
#         self.wrongColor = 'red'
#         self.wrong = False
#         self.invert = invert
#         self.net = net
#         self.info = info
#         self.rectangles = {}
#         for name in info:
#             # fill in any shape or scale info not provided by the user
#             if 'shape' not in info[name]:
#                 info[name]['shape'] = 1, net[name].size
#             if 'scale' not in info[name]:
#                 info[name]['scale'] = defaultScale
#             rows, cols = info[name]['shape']
#             assert rows*cols == net[name].size, 'mismatched shape for layer %s' % name
#             scale = info[name]['scale']
#             info[name]['width'] = cols * scale
#             info[name]['height'] = rows * scale
#             info[name]['size'] = net[name].size
#         # assign level numbers to each layer
#         levelNums = []
#         for layer in net.layers:
#             if layer.kind == 'Undefined':
#                 raise Exception('bad network architecture')
#             elif layer.kind in ('Input', 'Context'):
#                 info[layer.name]['level'] = 1
#             elif 'level' not in info[layer.name]:
#                 prevLayers = net.getIncomingLayers(layer.name)
#                 prevLevels = [info[l.name]['level'] for l in prevLayers]
#                 info[layer.name]['level'] = max(prevLevels) + 1
#             levelNums.append((layer.name, info[layer.name]['level']))
# #        print 'levelNums:', levelNums
#         maxLevel = max([num for (name, num) in levelNums])
#         # group layers together by level
#         # example: levels = [['input', 'context'] ['hidden'] ['output']]
#         levels = []
#         for i in range(1, maxLevel+1):
#             levels.append([name for (name, num) in levelNums if num == i])
# #        print 'levels:', levels
#         # figure out window size
#         widths, heights = [], []
#         for level in levels:
#             if level == []:
#                 raise Exception('bad network architecture')
#             levelWidth = sum([info[name]['width'] for name in level]) + (len(level)-1)*horizSpacing
#             widths.append(levelWidth)
#             levelHeight = max([info[name]['height'] for name in level]) + textSize
#             heights.append(levelHeight)
#         width = max(widths) + 2*sideMargin
#         height = sum(heights) + (len(levels)-1)*vertSpacing + topMargin + bottomMargin
#         # create window
#         GraphWin.__init__(self, title, width, height)
#         self.setBackground(self.bgColor)
#         labels = []
#         centerX = width / 2
#         baseline = height - bottomMargin
#         # figure out bounding box of each layer at each level
#         for level in levels:
#             levelWidth = sum([info[name]['width'] for name in level]) + (len(level)-1)*horizSpacing
#             maxHeight = max([info[name]['height'] for name in level])
#             leftX = centerX - levelWidth / 2
#             centerY = baseline - maxHeight / 2
#             x1 = leftX
#             for name in level:
#                 rows, cols = info[name]['shape']
#                 scale = info[name]['scale']
#                 w, h = info[name]['width'], info[name]['height']
#                 x2 = x1 + w
#                 y1 = centerY - h/2
#                 y2 = centerY + h/2
#                 info[name]['bbox'] = x1, y1, x2, y2
#                 # create activation rectangles
#                 rects = []
#                 x, y = x1, y1
#                 for r in xrange(rows):
#                     for c in xrange(cols):
#                         rect = self.create_rectangle(x, y, x+scale, y+scale, outline='black', fill='gray50')
#                         rects.append(rect)
#                         x += scale
#                     x, y = x1, y+scale
#                 self.rectangles[name] = rects
#                 # create label
#                 textX = (x1 + x2) / 2
#                 textY = y1 - 0.8 * textSize
#                 if info[name]['size'] > 1:
#                     label = Text(Point(textX, textY), '%s units' % name)
#                 else:
#                     label = Text(Point(textX, textY), '%s unit' % name)
#                 label.setSize(textSize)
#                 labels.append(label)
#                 x1 = x2 + horizSpacing
#             baseline = baseline - maxHeight - textSize - vertSpacing
#         # draw feedforward connections
#         lines = []
#         for c in net.connections:
#             from_x1, from_y1, from_x2, from_y2 = info[c.fromLayer.name]['bbox']
#             to_x1, to_y1, to_x2, to_y2 = info[c.toLayer.name]['bbox']
#             from_rows, from_cols = info[c.fromLayer.name]['shape']
#             to_rows, to_cols = info[c.toLayer.name]['shape']
#             from_scale = info[c.fromLayer.name]['scale']
#             to_scale = info[c.toLayer.name]['scale']
#             if style == 'line':
#                 from_xc = (from_x1 + from_x2) / 2
#                 to_xc = (to_x1 + to_x2) / 2
#                 l = Line(Point(from_xc, from_y1), Point(to_xc, to_y2))
#                 lines.append(l)
#             elif style == 'lines':
#                 l1 = Line(Point(from_x1, from_y1), Point(to_x1, to_y2))
#                 l2 = Line(Point(from_x2, from_y1), Point(to_x2, to_y2))
#                 lines.extend([l1, l2])
#             elif style == 'arrow':
#                 from_xc = (from_x1 + from_x2) / 2
#                 to_xc = (to_x1 + to_x2) / 2
#                 l = Line(Point(from_xc, from_y1), Point(to_xc, to_y2))
#                 l.setArrow('last')
#                 lines.append(l)
#             elif style == 'arrows':
#                 l1 = Line(Point(from_x1, from_y1), Point(to_x1, to_y2))
#                 l1.setArrow('last')
#                 l2 = Line(Point(from_x2, from_y1), Point(to_x2, to_y2))
#                 l2.setArrow('last')
#                 lines.extend([l1, l2])
#             elif style == 'full':
#                 from_x = from_x1 + from_scale/2
#                 for j in range(from_cols):
#                     to_x = to_x1 + to_scale/2
#                     for i in range(to_cols):
#                         l = Line(Point(from_x, from_y1), Point(to_x, to_y2))
#                         lines.append(l)
#                         to_x += to_scale
#                     from_x += from_scale
#             elif style == 'fullarrows':
#                 from_x = from_x1 + from_scale/2
#                 for j in range(from_cols):
#                     to_x = to_x1 + to_scale/2
#                     for i in range(to_cols):
#                         l = Line(Point(from_x, from_y1), Point(to_x, to_y2))
#                         l.setArrow('last')
#                         lines.append(l)
#                         to_x += to_scale
#                     from_x += from_scale
#         # draw recurrent context connections if any
#         if 'contextLayers' in dir(net):
#             recurrent_width = 2
#             for name in net.contextLayers:
#                 h_name = name
#                 c_name = net.contextLayers[h_name].name
#                 h_x1, h_y1, h_x2, h_y2 = info[h_name]['bbox']
#                 c_x1, c_y1, c_x2, c_y2 = info[c_name]['bbox']
#                 rec_y1 = (h_y1 + h_y2) / 2
#                 rec_y2 = (c_y1 + c_y2) / 2
#                 rec_x2 = max(h_x2, c_x2) + horizSpacing/2
#                 if rec_x2 >= width:
#                     rec_x2 = (max(h_x2, c_x2) + width) / 2
#                 l1 = Line(Point(h_x2, rec_y1), Point(rec_x2, rec_y1))
#                 l1.setWidth(recurrent_width)
#                 l2 = Line(Point(rec_x2, rec_y1), Point(rec_x2, rec_y2))
#                 l2.setWidth(recurrent_width)
#                 l3 = Line(Point(rec_x2, rec_y2), Point(c_x2, rec_y2))
#                 l3.setWidth(recurrent_width)
#                 l3.setArrow('last')
#                 lines.extend([l1, l2, l3])
# #                 # draw the arrowhead
# #                 arrowSize = 5
# #                 ax, ay = c_x2, rec_y2
# #                 l1 = Line(Point(ax, ay), Point(ax+arrowSize, ay-arrowSize))
# #                 l1.setWidth(recurrent_width)
# #                 l2 = Line(Point(ax, ay), Point(ax+arrowSize, ay+arrowSize))
# #                 l2.setWidth(recurrent_width)
# #                 lines.extend([l1, l2])
#         for l in lines:
#             l.draw(self)
#             l.canvas.tag_lower(l.id)
#         # draw labels
#         for label in labels:
#             label.setFill(fgText)
#             label.draw(self)
#             # draw the text background
#             bb_x1, bb_y1, bb_x2, bb_y2 = label.canvas.bbox(label.id)
#             self.create_rectangle(bb_x1, bb_y1, bb_x2, bb_y2, fill=bgText, outline=bgText)
#             label.canvas.tag_raise(label.id)
            
#     def update(self, layerName=None):
#         if self.closed: return
#         if self.wrong:
#             self.setBackground(self.bgColor)
#             self.wrong = False
#         for layer in self.net.layers:
#             if layerName is None or layerName == layer.name:
#                 rectangles = self.rectangles[layer.name]
#                 rows, cols = self.info[layer.name]['shape']
#                 for i in xrange(layer.size):
#                     grayLevel = int(100 * layer.activation[i])
#                     if self.invert:
#                         grayLevel = 100 - grayLevel
#                     fColor = oColor = 'gray%d' % grayLevel
#                     if rows == 1 or cols == 1: oColor = 'black'
#                     self.itemconfigure(rectangles[i], fill=fColor, outline=oColor)
#         self.update_idletasks()
#         #GraphWin.update(self)
#         while self.tk.dooneevent(2): pass

#     def showWrong(self):
#         if self.closed: return
#         self.wrong = True
#         self.setBackground(self.wrongColor)

# #------------------------------------------------------------------------------------------
# # NetworkWeightDisplay displays weight values of a network graphically

# class NetworkWeightDisplay(GraphWin):

#     def __init__(self, net, title, invert, kwInfo):
#         defaultScale = 20
#         sideMargin = 20
#         topMargin = 30
#         bottomMargin = 20
#         layoutColSpacing = 40
#         layoutRowSpacing = 20
#         vertSpacing = 40
#         textSize = 14
#         self.bgColor = color_rgb(176, 196, 222)  # 'light steel blue'
#         fgText = 'blue'
#         bgText = self.bgColor
#         self.invert = invert
#         self.net = net
#         # weight display is disabled during showPerformance
#         self.disabled = False
#         if net.actDisplay is None:
#             actInfo = None
#         else:
#             actInfo = net.actDisplay.info
#         # fill in default info for each layer and figure out window size
#         widths = []
#         heights = []
#         weightInfo = {}
#         self.info = weightInfo
#         for connection in net.connections:
#             toName = connection.toLayer.name
#             fromName = connection.fromLayer.name
#             # defaults
#             weightInfo[(fromName, toName)] = {'layout': (1, net[toName].size),
#                                               'scale': defaultScale,
#                                               'shape': (1, net[fromName].size)}
#             # use info from activation display if it exists
#             if actInfo is not None:
#                 weightInfo[(fromName, toName)]['layout'] = actInfo[toName]['shape']
#                 weightInfo[(fromName, toName)]['scale'] = actInfo[fromName]['scale']
#                 weightInfo[(fromName, toName)]['shape'] = actInfo[fromName]['shape']
#             # any info provided in keyword args overrides defaults
#             if 'layout' in kwInfo[toName]:
#                 weightInfo[(fromName, toName)]['layout'] = kwInfo[toName]['layout']
#             if 'scale' in kwInfo[fromName]:
#                 weightInfo[(fromName, toName)]['scale'] = kwInfo[fromName]['scale']
#             if 'shape' in kwInfo[fromName]:
#                 weightInfo[(fromName, toName)]['shape'] = kwInfo[fromName]['shape']
#             layoutRows, layoutCols = weightInfo[(fromName, toName)]['layout']
#             scale = weightInfo[(fromName, toName)]['scale']
#             rows, cols = weightInfo[(fromName, toName)]['shape']
#             unitWidth = cols*scale
#             unitHeight = rows*scale
#             totalWidth = unitWidth*layoutCols + (layoutCols-1)*layoutColSpacing
#             totalHeight = (unitHeight+textSize)*layoutRows + (layoutRows-1)*layoutRowSpacing
#             widths.append(totalWidth)
#             heights.append(totalHeight)
#             weightInfo[(fromName, toName)]['dims'] = (unitWidth, unitHeight, totalWidth, totalHeight)
#         width = max(widths) + 2*sideMargin
#         height = sum(heights) + (len(heights)-1)*vertSpacing + topMargin + bottomMargin
#         # create window
#         GraphWin.__init__(self, title, width, height)
#         self.setBackground(self.bgColor)
#         # lay out connections
#         centerX = width / 2
#         baseline = height - bottomMargin
#         for connection in net.connections:
#             toName = connection.toLayer.name
#             fromName = connection.fromLayer.name
#             layoutRows, layoutCols = weightInfo[(fromName, toName)]['layout']
#             unitWidth, unitHeight, totalWidth, totalHeight = weightInfo[(fromName, toName)]['dims']
#             xDelta = unitWidth + layoutColSpacing
#             yDelta = unitHeight + textSize + layoutRowSpacing
#             x1 = centerX - totalWidth / 2
#             x2 = centerX + totalWidth / 2
#             y1 = baseline - totalHeight + textSize
#             y2 = baseline
#             # rectangles for fromName->toName weights
#             weightInfo[(fromName, toName)]['rectangles'] = {}
#             for i in xrange(connection.toLayer.size):
#                 lrow, lcol = i / layoutCols, i % layoutCols
#                 x0, y0 = x1 + lcol*xDelta, y1 + lrow*yDelta
#                 rows, cols = weightInfo[(fromName, toName)]['shape']
#                 scale = weightInfo[(fromName, toName)]['scale']
#                 rects = []
#                 x, y = x0, y0
#                 for r in xrange(rows):
#                     for c in xrange(cols):
#                         rect = self.create_rectangle(x, y, x+scale, y+scale, outline='black', fill='gray50')
#                         rects.append(rect)
#                         x += scale
#                     x, y = x0, y+scale
#                 weightInfo[(fromName, toName)]['rectangles'][i] = rects
#                 # draw label
#                 textX = x0 + unitWidth / 2
#                 textY = y0 - 0.8 * textSize
#                 labelString = '%s->%s[%d]' % (fromName, toName, i)
#                 label = Text(Point(textX, textY), labelString)
#                 label.setSize(textSize)
#                 label.setFill(fgText)
#                 label.draw(self)
#             baseline = baseline - totalHeight - vertSpacing

#     def update(self):
#         if self.closed: return
#         if self.disabled: return
#         for connection in self.net.connections:
# #             print '%s (%d) -> %s (%d)' % (connection.fromLayer.name, connection.fromLayer.size,
# #                                           connection.toLayer.name, connection.toLayer.size)
#             toName = connection.toLayer.name
#             fromName = connection.fromLayer.name
#             rows, cols = self.info[(fromName, toName)]['shape']
#             matrix = connection.weight
#             # calculate current min and max values without allocating a new list
#             minWeight = matrix[0][0]
#             maxWeight = matrix[0][0]
#             for row in matrix:
#                 x = min(row)
#                 if x < minWeight:
#                     minWeight = x
#                 x = max(row)
#                 if x > maxWeight:
#                     maxWeight = x
#             maxMagnitude = max(1.0, abs(maxWeight), abs(minWeight))
#             rectangles = self.info[(fromName, toName)]['rectangles']
#             for i in xrange(connection.toLayer.size):
#                 for j in xrange(connection.fromLayer.size):
#                     w = matrix[j][i]
#                     scaled_w = (w + maxMagnitude)/(2 * maxMagnitude)
#                     grayLevel = int(100 * scaled_w)
#                     assert 0 <= grayLevel <= 100, 'bad grayLevel value: %s' % grayLevel
#                     if self.invert:
#                         grayLevel = 100 - grayLevel
#                     fColor = oColor = 'gray%d' % grayLevel
#                     if rows == 1 or cols == 1: oColor = 'black'
#                     self.itemconfigure(rectangles[i][j], fill=fColor, outline=oColor)
#         self.update_idletasks()
#         #GraphWin.update(self)
#         while self.tk.dooneevent(2): pass

#     def setTitle(self, title):
#         self.winfo_toplevel().title(title)
#         self.title = title

#------------------------------------------------------------------------------------------
# This code is obsolete, but should not be deleted yet because it
# includes support for displaying bias values, which still needs to be
# added to the NetworkWeightDisplay class in some form.

# class WeightDisplay(PGM):

#     def __init__(self, connection, i, shape, scale, showBias):
# #        global allWDs
#         # need to save connection object itself in order for initialize() to work
#         self.connection = connection
#         self.i = i
#         self.showBias = showBias
#         weights = [row[i] for row in self.connection.weight]
#         if showBias:
#             bias = connection.toLayer.weight[i]
#             weights = [bias] + weights
#             highlight = 0
#         else:
#             highlight = None
#         maxMagnitude = max(1.0, abs(max(weights)), abs(min(weights)))
#         scaledWeights = [(w + maxMagnitude)/(2 * maxMagnitude) for w in weights]
#         PGM.__init__(self, values=scaledWeights, shape=shape, scale=scale, highlight=highlight)
# #        allWDs.append(self)

#     def update(self):
#         weights = [row[self.i] for row in self.connection.weight]
#         if self.showBias:
#             bias = self.connection.toLayer.weight[self.i]
#             weights = [bias] + weights
#         maxMagnitude = max(1.0, abs(max(weights)), abs(min(weights)))
#         scaledWeights = [(w + maxMagnitude)/(2 * maxMagnitude) for w in weights]
#         self.updateImage(scaledWeights)

# class ActivationDisplay(PGM):

#     def __init__(self, layer, units, shape, scale):
# #        global allADs
#         # units is a list of unit index numbers
#         self.layer = layer
#         self.units = units
#         self.shape = shape
#         activations = [layer.activation[i] for i in units]
#         PGM.__init__(self, values=activations, shape=shape, scale=scale)
# #        allADs.append(self)

#     def update(self):
#         # assumes activations are always in range 0-1
#         activations = [self.layer.activation[i] for i in self.units]
#         self.updateImage(activations)

#------------------------------------------------------------------------------------------
# BackpropNetwork class wraps Network class with a more intuitive
# interface and support for displaying the network graphically.  This
# code should eventually be merged directly with the Network class.

class BackpropNetwork(Network):

    def __init__(self, name='Backprop Network', verbosity=0):
        Network.__init__(self, name=name, verbosity=verbosity)
        self.actDisplay = None
        self.weightDisplay = None
        #self.weightDisplays = []
        #self.activationDisplays = []
        self.testInputs = []
        self.testTargets = []
        self.resetLimit = 1
        self.classifications = None
        self.countWrong = False

#     def __del__(self):
#         Network.__del__(self)
#         print 'closing windows'
#         for wd in self.weightDisplays:
#             wd.close()
#         for ad in self.activationDisplays:
#             ad.close()

    # give arrayify method a better name
    def getWeightList(self):
        return self.arrayify()

    # give unArrayify method a better name
    def installWeights(self, weightList):
        self.unArrayify(weightList)
        self.updateGraphics()

    def getIncomingLayers(self, layerName):
        layers = []
        for c in self.connections:
            if c.toLayer.name == layerName:
                layers.append(c.fromLayer)
        return layers

    def getDefault(self, layerName, attribute):
        if self.actDisplay is None:
            return None
        elif layerName not in self.actDisplay.info:
            return None
        elif attribute not in self.actDisplay.info[layerName]:
            return None
        else:
            return self.actDisplay.info[layerName][attribute]

    def train(self, sweeps=None, cont=0):
        if sweeps is None:
            self.resetEpoch = 10000
        else:
            assert type(sweeps) is int and sweeps > 0, 'invalid number of sweeps'
            self.resetEpoch = sweeps
        Network.train(self, cont=cont)

    def showData(self):
        print("%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs)))

    def swapData(self, verbose=True):
        # swap training and testing datasets
        if verbose:
            print("Swapping training and testing sets...")
        self.inputs, self.testInputs = self.testInputs, self.inputs
        self.targets, self.testTargets = self.testTargets, self.targets
        if verbose:
            self.showData()

    # splitData takes a percentage from 0 to 100 as input, called trainingPortion,
    # and partitions the data into a training set and a testing set.
    def splitData(self, trainingPortion=None):
        if type(trainingPortion) not in [int, float] or not 0 <= trainingPortion <= 100:
            print('percentage of dataset to train on is required (0-100)')
            return
        patterns = list(zip(self.inputs + self.testInputs, self.targets + self.testTargets))
        assert len(patterns) > 0, "no dataset"
        print("Randomly shuffling data patterns...")
        random.shuffle(patterns)
        numTraining = int(math.ceil(trainingPortion / 100.0 * len(patterns)))
        self.inputs = [i for (i, t) in patterns[:numTraining]]
        self.targets = [t for (i, t) in patterns[:numTraining]]
        self.testInputs = [i for (i, t) in patterns[numTraining:]]
        self.testTargets = [t for (i, t) in patterns[numTraining:]]
        print("%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs)))

    def showPerformance(self):
        if len(self.inputs) ==  0:
            print('no patterns to test')
            return
        if self.weightDisplay is not None:
            self.weightDisplay.disabled = True
        #if 'classify' in dir(self):
        if self.classifications is not None:
            self.countWrong = True
            self.numRight = 0
            self.numWrong = 0
        else:
            self.countWrong = False
        learn, order, interact = self.learning, self.orderedInputs, self.interactive
        self.setLearning(0)
        self.setInteractive(1)
        # this forces sweep to go through the dataset patterns in order
        self.setOrderedInputs(1)
        self.quitFromSweep = False
        self.sweep()
        if self.countWrong and self.interactive:
            print('Got %d right, %d wrong' % (self.numRight, self.numWrong))
        # restore previous values
        self.setLearning(learn)
        self.setOrderedInputs(order)
        self.setInteractive(interact)
        self.countWrong = False
        if self.weightDisplay is not None:
            self.weightDisplay.disabled = False

    def setClassifications(self, classifications):
        self.classifications = classifications

    def classify(self, output):
        if self.classifications is None:
            return None
        for (target, classification) in self.classifications:
            if len(target) == len(output):
                numWrong = 0
                for i in range(len(target)):
                    if abs(target[i] - output[i]) > self.tolerance:
                        numWrong += 1
                if numWrong == 0:
                    return classification
        return '???'

    def evaluate(self):
        if self.classifications is None:
            print('no classifications specified')
            return
        if len(self.inputs) == 0:
            print('no patterns to evaluate')
            return
        correct = 0
        wrong = 0
        for i in range(len(self.inputs)):
            pattern = self.inputs[i]
            target = self.targets[i]
            output = self.propagate(input=pattern)
            networkAnswer = self.classify(output)
            correctAnswer = self.classify(target)
            if networkAnswer == correctAnswer:
                correct = correct + 1
            else:
                wrong = wrong + 1
                print('network classified pattern #%d (%s) as %s' % \
                      (i, correctAnswer, networkAnswer))
        total = len(self.inputs)
        correctPercent = float(correct) / total * 100
        wrongPercent = float(wrong) / total * 100
        print('%d patterns: %d correct (%.1f%%), %d wrong (%.1f%%)' % \
              (total, correct, correctPercent, wrong, wrongPercent))

    def showGeneralization(self):
        self.swapData(verbose=False)
        self.showPerformance()
        self.swapData(verbose=False)

    def updateGraphics(self):
        if self.actDisplay is not None and not self.actDisplay.closed:
            self.actDisplay.update()
        if self.weightDisplay is not None and not self.weightDisplay.closed:
            self.weightDisplay.update()

    def showNetwork(self, title=None, style='arrows', horizSpacing=40, vertSpacing=40, invert=False, **kw_args):
        if title is None:
            title = self.name
        info = self.process_keyword_args(kw_args)
        if self.actDisplay is not None and not self.actDisplay.closed:
            self.actDisplay.close()
        self.actDisplay = NetworkActivationDisplay(self, title, style, horizSpacing, vertSpacing, invert, info)
        self.updateGraphics()

    def showWeights(self, title='Weights', invert=False, **kw_args):
        info = self.process_keyword_args(kw_args)
        if self.weightDisplay is not None and not self.weightDisplay.closed:
            self.weightDisplay.close()
        self.weightDisplay = NetworkWeightDisplay(self, title, invert, info)
        self.updateGraphics()

    def process_keyword_args(self, kw_args):
        commonScale = None
        info = {}
        for layer in self.layers:
            info[layer.name] = {}
        for keyword in kw_args:
            if keyword == 'scale':
                commonScale = kw_args[keyword]
            elif keyword[-6:] == '_scale':
                name = keyword[:keyword.index('_scale')]
                assert name in info, 'bad layer name: %s' % name
                scale = kw_args[keyword]
                assert type(scale) is int and scale > 0, 'bad %s scale: %s' % (name, scale)
                info[name]['scale'] = scale
            elif keyword[-6:] == '_shape':
                name = keyword[:keyword.index('_shape')]
                assert name in info, 'bad layer name: %s' % name
                shape = kw_args[keyword]
                assert type(shape) in (tuple, list) and len(shape) == 2, \
                    '%s layer shape not of the form (rows, cols)' % name
                rows, cols = shape
                info[name]['shape'] = shape
            elif keyword[-5:] == '_cols':
                # layout option is used only by NetworkWeightDisplay
                name = keyword[:keyword.index('_cols')]
                assert name in info, 'bad layer name: %s' % name
                layout = kw_args[keyword]
                assert type(layout) is int and layout > 0, 'bad %s column size: %s' % (name, layout)
                cols = min(layout, self[name].size)
                rows = int(math.ceil(self[name].size / float(cols)))
                info[name]['layout'] = (rows, cols)
            elif keyword[-6:] == '_level':
                name = keyword[:keyword.index('_level')]
                assert name in info, 'bad layer name: %s' % name
                level = kw_args[keyword]
                assert type(level) is int and level > 1, 'bad %s level number: %s' % (name, level)
                info[name]['level'] = level
            else:
                raise Exception('Unrecognized keyword: %s' % keyword)
        if commonScale is not None:
            for name in info:
                if 'scale' not in info[name]:
                    info[name]['scale'] = commonScale
        return info

    # displays a specific input pattern graphically in a popup window
    def showInput(self, inputNumber, shape=0, scale=0, invert=False):
        assert len(self.inputs) > 0, 'no input patterns are currently defined'
        assert type(inputNumber) is int and 0 <= inputNumber < len(self.inputs), \
            "input number must be in range 0-%d" % (len(self.inputs) - 1)
        pattern = self.inputs[inputNumber]
        shape = validateShape(pattern, shape, self.getDefault('input', 'shape'))
        scale = validateScale(scale, self.getDefault('input', 'scale'))
        title = 'input #%d' % inputNumber
        pgm = PGM(values=pattern, title=title, shape=shape, scale=scale, invert=invert)

    def showInputs(self, inputNums, shape=0, scale=0, invert=False):
        assert len(self.inputs) > 0, 'no input patterns are currently defined'
        for i in inputNums:
            assert type(i) is int, 'a list of input numbers is required'
            assert 0 <= i < len(self.inputs), \
                'input numbers must be in range 0-%d' % (len(self.inputs) - 1)
        self.showPatterns([self.inputs[i] for i in inputNums], shape, scale, invert)

    def showPattern(self, pattern, title='', shape=0, scale=0, invert=False):
        if type(pattern) is int:
            print('calling showInput(%d)' % pattern)
            self.showInput(pattern, shape, scale, invert)
            return
        elif type(pattern) is str:
            assert self.patterned and pattern in self.patterns, \
                'bad pattern name: %s' % pattern
            pattern = self.patterns[pattern]
        assert type(pattern) is list and len(pattern) > 0, 'bad pattern: %s' % pattern
        shape = validateShape(pattern, shape, self.getDefault('input', 'shape'))
        scale = validateScale(scale, self.getDefault('input', 'scale'))
        pgm = PGM(values=pattern, title=title, shape=shape, scale=scale, invert=invert)

    def showPatterns(self, patterns, shape=0, scale=0, invert=False):
        assert type(patterns) is list and len(patterns) > 0, 'a list of patterns is required'
        if type(patterns[0]) is str:
            # assume patterns is a list of pattern names
            assert self.patterned, 'no pattern names defined'
            for name in patterns:
                assert name in self.patterns, 'unknown pattern name: %s' % name
            patterns = [self.patterns[name] for name in patterns]
        elif type(patterns[0]) is not list:
            # assume patterns is really a single pattern
            self.showPattern(patterns, shape=shape, scale=scale, invert=invert)
            return
        shape = validateShape(patterns[0], shape, self.getDefault('input', 'shape'))
        scale = validateScale(scale, self.getDefault('input', 'scale'))
        pgm = PGM(values=patterns[0], shape=shape, scale=scale, invert=invert)
        for pattern in patterns[1:]:
            answer = input('<enter> to continue, <q> to quit... ')
            if answer in ['Q', 'q']:
                pgm.close()
                return
            validateShape(pattern, shape, self.getDefault('input', 'shape'))
            if pgm.closed: return
            pgm.updateImage(pattern, invert)
        input('<enter> to close... ')
        pgm.close()

    def initialize(self):
        Network.initialize(self)
        self.updateGraphics()

    def apply(self, pattern):
        # save mode
        interactive = self.interactive
        self.setInteractive(1)
        output = self.propagate(input=pattern)
        print('output is [%s]' % pretty(output))
        # restore mode
        self.setInteractive(interactive)

    def propagate(self, *arg, **kw):
        """ Propagates activation through the network."""
        output = Network.propagate(self, *arg, **kw)
        if self.interactive:
            self.updateGraphics()
        # IMPORTANT: convert results from numpy.floats to conventional floats 
        if type(output) == dict:
            for layerName in output:
                output[layerName] = [float(x) for x in output[layerName]]
            return output
        else:
            return [float(x) for x in output]

    def sweep(self):
        result = Network.sweep(self)
        self.updateGraphics()
        return result

    def loadWeightsFromFile(self, filename, mode='pickle'):
        """
        Deprecated. Use loadWeights instead.
        """
        Network.loadWeights(self, filename, mode)
        self.updateGraphics()

    def loadWeights(self, filename, mode='pickle'):
        """
        Loads weights from a file in pickle, plain, or tlearn mode.
        """
        Network.loadWeights(self, filename, mode)
        self.updateGraphics()

    # saveHiddenReps creates a file containing the hidden layer representations
    # generated by the network for all of the input patterns in the dataset,
    # and, if the classify method is present, a parallel file of labels
    # corresponding to the network's classification of each input.
    def saveHiddenReps(self, filename):
        if len(self.inputs) == 0:
            print('no input patterns available')
            return
        learn, order, interact = self.learning, self.orderedInputs, self.interactive
        self.setLearning(0)
        self.setInteractive(0)
        # this forces sweep to go through the dataset patterns in order
        self.setOrderedInputs(1)
        # record the internal hidden layer patterns in a log file
        logfile = filename + '.hiddens'
        hidden = self.getLayer('hidden')
        hidden.setLog(logfile)
        self.sweep()
        hidden.closeLog()
        print('%d hidden layer patterns saved in %s' % (len(self.inputs), logfile))
        #if 'classify' in dir(self):
        if self.classifications is not None:
            # create a parallel file of classification labels
            labelfile = filename + '.labels'
            f = open(labelfile, 'w')
            for pattern in self.inputs:
                output = self.propagate(input=pattern)
                label = self.classify(output)
                f.write('%s\n' % label)
            f.close()
            print('%d classifications saved in %s' % (len(self.inputs), labelfile))
        # restore previous values
        self.setLearning(learn)
        self.setOrderedInputs(order)
        self.setInteractive(interact)
        self.updateGraphics()

    def printWeights(self, fromLayerName=None, toLayerName=None, whole=2, frac=4, color=False):
        if fromLayerName is None and toLayerName is None:
            connections = [c for c in self.connections]
            connections.reverse()
            for connection in connections:
                fromName = connection.fromLayer.name
                toName = connection.toLayer.name
                self.printWeights(fromName, toName, whole, frac, color)
        elif fromLayerName is None or toLayerName is None:
            print('Usage: printWeights(fromLayerName, toLayerName)')
        else:
            layerNames = [layer.name for layer in self.layers]
            assert fromLayerName in layerNames, 'no such layer: %s' % fromLayerName
            assert toLayerName in layerNames, 'no such layer: %s' % toLayerName
            connection = self.getConnection(fromLayerName, toLayerName)
            numFrom = connection.fromLayer.size
            numTo = connection.toLayer.size
            fromName = connection.fromLayer.name
            toName = connection.toLayer.name
            fromLabelWidth = len('%s[%d]' % (fromName, numFrom - 1))
            toLabelWidth = whole + frac + 2
            print('Weights from %s to %s:' % (fromName, toName))
            print(' ' * fromLabelWidth, end=" ")
            for i in range(numTo):
                label = '%s[%d]' % (toName[0:min(3, len(toName))], i)
                print(self.colorize('gray', '%*s' % (toLabelWidth, label), color=color), end=" ")
            print(self.colorize('red', '\n%-*s' % (fromLabelWidth, 'bias'), color=color), end=" ")
            for i in range(numTo):
                bias = connection.toLayer.weight[i]
                print(self.colorize('red', self.formatReal(bias, whole, frac), color=color), end=" ")
            print()
            for j in range(numFrom):
                label = '%s[%d]' % (fromName, j)
                print(self.colorize('gray', '%-*s' % (fromLabelWidth, label), color=color), end=" ")
                for i in range(numTo):
                    w = connection.weight[j][i]
                    print(self.formatReal(w, whole, frac), end=" ")
                print()
            print()

    def formatReal(self, value, maxWholeSize, maxFracSize):
        maxSize = maxWholeSize + maxFracSize + 2
        wholeSize = len(str(int(abs(value))))
        if wholeSize > maxWholeSize:
            maxFracSize = max(0, maxFracSize - (wholeSize - maxWholeSize))
        if wholeSize > maxSize - 2:
            if value > 0:
                s = '>+' + ('9' * (maxSize - 2))
            else:
                s = '<-' + ('9' * (maxSize - 2))
        else:
            s = '%+*.*f' % (maxSize, maxFracSize, value)
        return s

    def colorize(self, colorName, string, color=True):
        colors = {'red': 31, 'green': 32, 'brown': 33, 'blue': 34, 'magenta': 35,
                  'darkblue': 36, 'gray': 37, 'underline': 38, 'white': 40, 'invRed': 41,
                  'invGreen': 42, 'invBrown': 43, 'invBlue': 44, 'invMagenta': 45}
        if color is False or colorName not in colors:
            return string
        else:
            return '\033[01;%dm%s\033[0m' % (colors[colorName], string)

    # overrides conx version (this version produces simpler output) - 11/29/06 
    def reportEpoch(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print('Epoch #%6d | TSS Error: %.4f | Correct: %.4f' % \
                   (epoch, tssErr, totalCorrect * 1.0 / totalCount))
        for layerName in pcorrect:
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()

    # overrides conx version (this version produces simpler output) - 11/29/06 
    def reportFinal(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print('Final #%6d | TSS Error: %.4f | Correct: %.4f' % \
                   (epoch-1, tssErr, totalCorrect * 1.0 / totalCount))
        for layerName in pcorrect:
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()

    # overrides Network method
    def display(self):
        """Displays the network to the screen."""
        size = list(range(len(self.layers)))
        size.reverse()
        for i in size:
            layer = self.layers[i]
            if layer.active:
                print('%s layer (size %d)' % (layer.name, layer.size))
                tlabel, olabel = '', ''
                if (layer.type == 'Output'):
                    if self.countWrong:
                        tlabel = ' (%s)' % self.classify(layer.target.tolist())
                        olabel = ' (%s)' % self.classify(layer.activation.tolist())
                        if olabel == tlabel:
                            self.numRight += 1
                        else:
                            olabel += '  *** WRONG ***'
                            self.numWrong += 1
                            if self.actDisplay is not None:
                                self.actDisplay.showWrong()
                    print('Target    : %s%s' % (pretty(layer.target, max=15), tlabel))
                print('Activation: %s%s' % (pretty(layer.activation, max=15), olabel))
                if self.patterned and layer.type != 'Hidden':
                    targetWord, diff = self.getWord( layer.target, returnDiff = 1)
                    if layer.kind == 'Output':
                        if targetWord == None:
                            print("Target Pattern   = %s" % "No match")
                        else:
                            if diff == 0.0:
                                print("Target Pattern   = '%s'" % targetWord)
                            else:
                                print("Target Pattern   = '%s'; difference = %f)" % (targetWord, diff))
                    actWord, diff = self.getWord( layer.activation, returnDiff = 1 )
                    if (layer.kind == 'Input' or layer.kind == 'Output'):
                        if actWord == None:
                            print("Matching Pattern = %s" % "No match")
                        else:
                            if diff == 0.0:
                                print("Matching Pattern = '%s'" % actWord)
                            else:
                                print("Matching Pattern = '%s'; difference = %f" % (actWord, diff))
                print("------------------------------------")
                #print "-----------------------------------"
   
# The following methods have been superseded by the NetworkActivationDisplay
# and NetworkWeightDisplay classes, but are retained here for future reference.

#     def showWeights(self, layerName=None, i=None, shape=0, scale=0, showBias=False):
#         if layerName is None:
#             for layer in self.layers:
#                 if layer.name != 'input':
#                     self.showWeights(layer.name, i, shape, scale, showBias)
#         else:
#             assert layerName in self.layersByName, 'no such layer: %s' % layerName
#             assert layerName in ['hidden', 'output'], \
#                    'showWeights currently only works for hidden or output units'
#             toLayer = self[layerName]
#             if layerName == 'hidden':
#                 fromLayer = self['input']
#             elif layerName == 'output':
#                 if 'hidden' not in self.layersByName:
#                     fromLayer = self['input']
#                 else:
#                     fromLayer = self['hidden']
#             shape = validateShape(fromLayer, shape, self.getDefault(fromLayer.name, 'shape'))
#             scale = validateScale(scale, self.getDefault(fromLayer.name, 'scale'))
#             if showBias:
#                 (rows, cols) = shape
#                 assert rows == 1, 'shape %s cannot display bias value' % (shape,)
#                 shape = (rows, cols+1)
#             connection = self.getConnection(fromLayer.name, toLayer.name)
#             if i is None:
#                 units = range(toLayer.size)
#             else:
#                 assert 0 <= i < toLayer.size, 'invalid %s unit number: %d' % (toLayer.name, i)
#                 units = [i]
#             for i in units:
#                 wd = WeightDisplay(connection, i, shape, scale, showBias)
#                 wd.setTitle('%s[%d] weights' % (toLayer.name, i))
#                 self.weightDisplays.append(wd)

#     def showActivations(self, layerName=None, units='ALL', shape=0, scale=0):
#         if layerName is None:
#             for layer in self.layers:
#                 self.showActivations(layer.name, units, shape, scale)
#         else:
#             assert layerName in [layer.name for layer in self.layers], 'no such layer: %s' % layerName
#             layer = self[layerName]
#             if units is 'ALL':
#                 units = range(layer.size)
#             else:
#                 assert type(units) is list and len(units) > 0, 'a list of unit numbers is required'
#                 for i in units:
#                     assert type(i) is int and 0 <= i < layer.size, \
#                            'invalid %s unit number: %s' % (layer.name, i)
#             shape = validateShape(units, shape, self.getDefault(layerName, 'shape'))
#             scale = validateScale(scale, self.getDefault(layerName, 'scale'))
#             ad = ActivationDisplay(layer, units, shape, scale)
#             if units == range(units[0], units[-1]+1):
#                 # units are contiguous
#                 unitNums = '%d-%d' % (units[0], units[-1])
#             else:
#                 unitNums = string.join([str(i) for i in units], ', ')
#             ad.setTitle('%s units %s' % (layer.name, unitNums))
#             self.activationDisplays.append(ad)

#------------------------------------------------------------------------------------------
# SRN class

class SRN(BackpropNetwork):
    """
    A subclass of BackpropNetwork. SRN allows for simple recursive networks by
    copying hidden activations back to a context layer. This subclass
    adds support for sequencing, prediction, and context layers.
    """
    # constructor
    def __init__(self, name = "Simple Recurrent Network", verbosity = 0):
        """
        Constructor for SRN sub-class. Support for sequences and prediction added.
        """
        BackpropNetwork.__init__(self, name=name, verbosity=verbosity)
        self.sequenceType = None # You will need to set this!
        # It should be one or the other:
        # self.sequenceType = "ordered-continuous"    
        self.orderedInputs = 1           
        self.initContext = 0             
        # self.sequenceType = "random-segmented":
        # self.orderedInputs = 0
        # self.initContext = 1
        # self.sequenceType = "random-continuous":
        # self.orderedInputs = 0
        # self.initContext = 0
        # self.sequenceType = "ordered-segmented":
        # self.orderedInputs = 1
        # self.initContext = 1
        # Other options:
        self.learnDuringSequence = 1
        self.contextCopying = 1
        # Internal stuff:
        self.prediction = []
        self.contextLayers = {} # records layer reference and associated hidden layer
    def setSequenceType(self, value):
        """
        You must set this!
        """
        if value == "ordered-continuous":
            self.orderedInputs = 1           
            self.initContext = 0             
        elif value == "random-segmented":
            self.orderedInputs = 0
            self.initContext = 1
        elif value == "random-continuous":
            self.orderedInputs = 0
            self.initContext = 0
        elif value == "ordered-segmented":
            self.orderedInputs = 1
            self.initContext = 1
        else:
            raise AttributeError("invalid sequence type: '%s'" % value)
        self.sequenceType = value
    # set and get methods for attributes
    def predict(self, inName, outName):
        """
        Sets prediction between an input and output layer.
        """
        self.prediction.append((inName, outName))
    def setInitContext(self, value):
        """
        Clear context layer between sequences.
        """
        self.initContext = value
    def setLearnDuringSequence(self, value):
        """
        Set self.learnDuringSequence.
        """
        self.learnDuringSequence = value

    # methods for constructing and modifying SRN network
    def addLayers(self, *arg, **kw):
        """
        Creates an N layer network with 'input', 'hidden1', 'hidden2',...
        and 'output' layers. Keyword type indicates "parallel" or
        "serial". If only one hidden layer, it is called "hidden".
        """
        netType = "serial"
        if "type" in kw:
            netType = kw["type"]
        self.addLayer('input', arg[0])
        hiddens = []
        contexts = []
        if len(arg) > 3: # FIXME: add context for each
            hcount = 0
            for hidc in arg[1:-1]:
                name = 'hidden%d' % hcount
                cname = 'context%d' % hcount
                self.addLayer(name, hidc)
                hiddens.append(name)
                self.addContextLayer(cname, hidc, name)
                hiddens.append((cname, name))
                hcount += 1
        elif len(arg) == 3:
                name = 'hidden'
                self.addContextLayer('context', arg[1], name)
                self.addLayer(name, arg[1])
                hiddens.append(name)
                contexts.append(('context', name))
        elif len(arg) == 2:
            pass
        else:
            raise AttributeError("not enough layers! need >= 2")
        self.addLayer('output', arg[-1])
        # Connect contexts
        for (fromName, toName) in contexts:
            self.connect(fromName, toName)
        lastName = "input"
        for name in hiddens:
            if netType == "parallel":
                self.connect('input', name)
                self.connect(name, 'output')
            else: # serial
                self.connect(lastName, name)
                lastName = name
        if netType == "serial" or lastName == "input":
            self.connect(lastName, "output")
    def addThreeLayers(self, inc, hidc, outc):
        """
        Creates a three level network with a context layer.
        """
        self.addLayer('input', inc)
        self.addContextLayer('context', hidc, 'hidden')
        self.addLayer('hidden', hidc)
        self.addLayer('output', outc)
        self.connect('input', 'hidden')
        self.connect('context', 'hidden')
        self.connect('hidden', 'output')
    def addSRNLayers(self, inc, hidc, outc):
        """
        Wraps SRN.addThreeLayers() for compatibility.
        """
        self.addThreeLayers(inc, hidc, outc)
    def addContextLayer(self, name, size, hiddenLayerName='hidden', verbosity=0):
        layer = Layer(name, size)
        self.addContext(layer, hiddenLayerName, verbosity)
    def addContext(self, layer, hiddenLayerName = 'hidden', verbosity = 0):
        """
        Adds a context layer. Necessary to keep self.contextLayers dictionary up to date. 
        """
        # better not add context layer first if using sweep() without mapInput
        SRN.add(self, layer, verbosity)
        if hiddenLayerName in self.contextLayers:
            raise KeyError('There is already a context layer associated with this hidden layer.', \
                             hiddenLayerName)
        else:
            self.contextLayers[hiddenLayerName] = layer
            layer.kind = 'Context'

    # new methods for sweep, step, propagate
    def copyHiddenToContext(self):
        """
        Uses key to identify the hidden layer associated with each
        layer in the self.contextLayers dictionary. 
        """
        for item in list(self.contextLayers.items()):
            if self.verbosity > 2: print('Hidden layer: ', self.getLayer(item[0]).activation)
            if self.verbosity > 2: print('Context layer before copy: ', item[1].activation)
            item[1].copyActivations(self.getLayer(item[0]).activation)
            if self.verbosity > 2: print('Context layer after copy: ', item[1].activation)
    def setContext(self, value = .5):
        """
        Clears the context layer by setting context layer to (default) value 0.5. 
        """
        for context in list(self.contextLayers.values()):
            context.resetFlags() # hidden activations have already been copied in
            context.setActivations(value)
    def propagate(self, **args):
        """
        SRN.propagate: Sets error flags and propagates.
        """
        return BackpropNetwork.propagate(self, **args)
    def step(self, **args):
        """
        SRN.step()
        Extends network step method by automatically copying hidden
        layer activations to the context layer.
        """
        if self.sequenceType == None:
            raise AttributeError("""sequenceType not set! Use SRN.setSequenceType() """)
        # take care of any params other than layer names:
        # two ways to clear context:
        # 1. force it to right now with arg initContext = 1:
        if 'initContext' in args:
            if args['initContext']:
                self.setContext()
            del args['initContext']
        # 2. have initContext be true
        elif self.initContext:
            self.setContext()
        # if initContext is off, then we assume user knows that,
        # so we reset the flags on all context layers:
        if self.initContext == 0:
            for context in list(self.contextLayers.values()):
                context.activationSet = 1
        # replace all patterns
        for key in args:
            args[key] = self.replacePatterns( args[key], key )
        # Get all of the input/output layer names:
        inputBankNames = [layer.name for layer in self.layers if layer.kind == 'Input']
        outputBankNames = [layer.name for layer in self.layers if layer.kind == 'Output']
        inputBankSizes = [layer.size for layer in self.layers if layer.kind == 'Input']
        inputBankTotalSize = sum(inputBankSizes)
        inputArgSizes = [len(args[name]) for name in inputBankNames if name in args]
        inputArgTotalSize = sum(inputArgSizes)
        sequenceLength = inputArgTotalSize // inputBankTotalSize
        learning = self.learning
        totalRetvals = (0.0, 0, 0) # error, correct, total
        totalPCorrect = {}
        for step in range(sequenceLength):
            if self.verbosity >= 1 or self.interactive:
                print("-----------------------------------Step #", step + 1)
            dict = {}
            dict.update(args) # in case context, or others
            # now, overwrite input and output, if necessary
            for name in inputBankNames:
                if name in args:
                    patternLength = self[name].size
                    offset = step * patternLength
                    if (offset + patternLength) >= len(args[name]):
                        # if this seq is too big, use last part:
                        dict[name] = args[name][-patternLength:]
                    else:
                        # else, go to the right spot in seq:
                        dict[name] = args[name][offset:offset+patternLength]
            for name in outputBankNames:
                if name in args:
                    patternLength = self[name].size
                    offset = step * patternLength
                    if (offset + patternLength) >= len(args[name]):
                        # if this seq is too big, use last part:
                        dict[name] = args[name][-patternLength:]
                    else:
                        # else, go to the right spot in seq:
                        dict[name] = args[name][offset:offset+patternLength]
            # get info for predicition -------------------------
            for p in self.prediction:
                (inName, outName) = p
                inLayer = self.getLayer(inName)
                if not inLayer.type == 'Input':
                    raise LayerError('Prediction input layer not type \'Input\'.', inLayer.type)
                outLayer = self.getLayer(outName)
                if not outLayer.type == 'Output':
                    raise LayerError('Prediction output layer not type \'Output\'.', outLayer.type)
                if step == sequenceLength - 1: # last one in sequence; what do we do?
                    start = 0 # wrap to next input vector
                    if not self._sweeping: # not in sweep, in step, no target
                        raise LayerError("Attempting to predict last item in sequence, but using step(). Use sweep() instead.")
                    else: # in a sweep, so get the next pattern if one:
                        if self.currentSweepCount == None: # last item in epoch, predict back to first pattern
                            # Train it to predict first pattern, first sequence item
                            pattern = self.getData(self.loadOrder[0])
                            for key in pattern:
                                pattern[key] = self.replacePatterns( pattern[key], key )
                            if inName in inputBankNames:
                                if inName in pattern:
                                    dict[outName] = pattern[inName][start:start+patternLength]
                            #dict[outName] = pattern["input"][start:start+patternLength]
                        else:
                            pattern = self.getData(self.loadOrder[self.currentSweepCount+1]) 
                            for key in pattern:
                                pattern[key] = self.replacePatterns( pattern[key], key )
                            if inName in inputBankNames:
                                if inName in pattern:
                                    dict[outName] = pattern[inName][start:start+patternLength]
                            #dict[outName] = pattern["input"][start:start+patternLength]
                else: # in middle of sequence
                    start = (step + 1) * inLayer.size
                    dict[outName] = args[inName][start:start+patternLength]
            # end predicition code -----------------------------
            if step < sequenceLength - 1: # not the last one
                if not self.learnDuringSequence:
                    self.learning = 0
            retvals = self.networkStep(**dict)
            self.learning = learning # in case we turned it off
            totalRetvals = list(map(lambda x,y: x+y, totalRetvals[:3], retvals[:3]))
            sumMerge(totalPCorrect, retvals[3])
            totalRetvals.append( totalPCorrect)
        return totalRetvals
    def prePropagate(self, **args):
        if not self.contextCopying:
            for layer in self.layers:
                if layer.kind == "Context":
                    layer.activationSet = 1
        return None
    def postBackprop(self, **args):
        if self.contextCopying:
            self.copyHiddenToContext() # must go after error computation
        return None
    def networkStep(self, **args):
        """
        This exists so that other extensions can interface at the point
        where the call is made to BackpropNetwork.step(). See governor.py for
        an example.
        """
        return BackpropNetwork.step(self, **args)
    def sweepCrossValidation(self):
        return BackpropNetwork.sweepCrossValidation(self)

    # newConx methods

    def showPerformance(self):
        """
        SRN.showPerformance()
        Clears the context layer(s) and then repeatedly cycles through
        training patterns until the user decides to quit.
        """
        if len(self.inputs) == 0:
            print('no patterns to test')
            return
        self.setContext()
        while True:
            BackpropNetwork.showPerformance(self)
            if self.quitFromSweep:
                return

#---------------------------------------------------------------------------------
# Test code

def main():
    global n
    # Con-x: Sample Networks
    # (c) 2001, D.S. Blank
    # Bryn Mawr College
    # http://emergent.brynmawr.edu/

    def ask(question):
        ans = eval(input(question + ' [y/n/q]'))
        #ans = sys.stdin.readline()[0].lower()
        if ans == 'q':
            sys.exit()
        return ans == 'y'

    n = Network()
    n.setSeed(114366.64)
    n.addThreeLayers(2, 2, 1)
    n.setInputs([[0.0, 0.0],
                 [0.0, 1.0],
                 [1.0, 0.0],
                 [1.0, 1.0]
                 ])
    n.setTargets([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]
                  ])
    n.setReportRate(10)

    if ask("Do you want to test the SigmaNetwork?"):
        net = SigmaNetwork()
        net.setSeed(457646.23)
        net.addLayers(2, 5, 11)
        net.setInputs( [[0, 0], [0, 1], [1, 0], [1, 1]] )
        net.setTargets( [[0], [1], [1], [0]] )
        net.train()
        #net.interactive = 1
        #net.learning = 0
        net.sweep()
        
    if ask("Do you want to test the pattern replacement utility?"):
        net = Network()
        net.addThreeLayers(3, 2, 3)
        net.setTolerance(.4)
        print("Setting patterns to one 0,0,0; two 1,1,1...")
        net.setPatterns( {"one" : [0, 0, 0], "two" :  [1, 1, 1]} )
        print(net.getPattern("one"))
        print(net.getPattern("two"))
        print("Replacing patterns... (should return [0, 0, 0, 1, 1, 1])")
        print(net.replacePatterns(["one", "two"]))
        net.setInputs([ "one", "two" ])
        net.copyActivations(net["input"], net.inputs[0])
        net.resetFlags()
        print("one is: ", end=" ")
        print(net["input"].getActivations())
        net.copyActivations(net["input"], net.inputs[1])
        net.resetFlags()
        print("two is: ", end=" ")
        print(net["input"].getActivations())
        net.addPattern("1", 1)
        net.addPattern("0", 0)
        print("Setting patterns to 0 and 1...")
        net.setInputs([ [ "0", "1", "0" ], ["1", "1", "1"]])
        print("Testing replacePatterns and patternVector...")
        print(net.replacePatterns(net.inputs[0]))
        print(net.patternVector(net.replacePatterns(net.inputs[0])))
        net.copyActivations(net["input"], net.inputs[0])
        net.resetFlags()
        print("0 1 0 is: ", end=" ")
        print(net["input"].getActivations())
        net.copyActivations(net["input"], net.inputs[1])
        print("1 1 1 is: ", end=" ")
        print(net["input"].getActivations())
        print("Reverse look up of .2, .3, .2 is ", net.getWord([.2, .3, .2]))
        print("Reverse look up of .8, .7, .5 is ", net.getWord([.8, .7, .5]))
        print("Reverse look up of .8, .9, 1 is ", net.getWord([.8, .9, 1]))
        if ask("Do you want to test saving and loading of patterned inputs and targets?"):
            print(net.patterns)
            print(net.patterned)
            net.setInputs(['one','two'])
            net.setTargets([['1'],['0']])
            print("Filename to save inputs: ", end=" ")
            filename = sys.stdin.readline().strip()
            print("Saving Inputs: ", net.inputs)
            net.saveInputs(filename)
            net.loadInputs(filename)
            print("Loading Inputs: ", net.inputs)
            print("Filename to save targets: ", end=" ")
            filename = sys.stdin.readline().strip()
            print("Saving Targets: ", net.targets)
            net.saveTargets(filename)
            net.loadTargets(filename)
            print("Loading Targets: ", net.targets)
        if ask("Do you want to test saving and loading patterned data?"):
            print("Setting inputs and targets...")
            net.setInputs(['one','two'])
            net.setTargets([['1'],['0']])
            print("Filename to save data: ", end=" ")
            filename = sys.stdin.readline().strip()
            print("Saving data: ")
            print(net.inputs)
            print(net.targets)
            net.saveData(filename)
            print("Loading data: ")
            net.loadData(filename, 1)
            print(net.inputs)
            print(net.targets)
            
    if ask("Do you want to test saving and loading inputs and targets with XOR?"):
        print("Filename to save inputs: ", end=" ")
        filename = sys.stdin.readline().strip()
        print("Saving Inputs: ", n.inputs)
        n.saveInputs(filename)
        n.loadInputs(filename)
        print("Loading Inputs: ", n.inputs)
        print("Filename to save targets: ", end=" ")
        filename = sys.stdin.readline().strip()
        print("Saving Targets: ", n.targets)
        n.saveTargets(filename)
        n.loadTargets(filename)
        print("Loading Targets: ", n.targets)

    if ask("Do you want to test saving and loading XOR data?"):
        print("Filename to save data: ", end=" ")
        filename = sys.stdin.readline().strip()
        print("Saving data: ")
        print(n.inputs)
        print(n.targets)
        n.saveData(filename)
        print("Loading data: ")
        n.loadData(filename, 1)
        print(n.inputs)
        print(n.targets)
        
    if ask("Do you want to see some test values?"):
        print('Input Activations:', n.getLayer('input').getActivations())
        print("Setting target to .5")
        n.getLayer("output").copyTargets([.5])
        print('Output Targets:', n.getLayer('output').getTargets())
        n.compute_error()
        print('Output TSS Error:', n.TSSError("output"))
        print('Output Correct:', n.getCorrect('output'))

    if ask("Do you want to run an XOR BACKPROP network in QUICKPROP mode?"):
        print("XOR Quickprop mode: ..............................")
        n = Network()
        n.setSeed(1234)
        n.addLayers(2, 2, 1)
        n.quickprop = 1
        n.mu = 1.75
        n.epsilon = 4.0
        n.maxRandom = 1.0
        n.setInputs( [[0, 0], [0, 1], [1, 0], [1, 1]] )
        n.setTargets( [[0], [1], [1], [0]] )
        n.reset()
        n.setReportRate(5)
        n.resetEpoch = 100
        n.resetLimit = 5
        n.train()
        print("getError('output') = (tss, correct, total):", n.getError("output"))
        print("getError('output', 'output') :", n.getError("output", "output"))
        # test crossvalidation ---------------------------------
        print("Testing crossvalidation.. saving network sweep in 'sample.cv'...")
        import posix, posixpath
        if posixpath.exists("sample.cv"):
            posix.unlink("sample.cv")
        n.learning = 0
        n.crossValidationSampleRate = 1
        #saves in "sample.cv"
        n.sweep()
        n.learning = 1
        n.crossValidationSampleRate = 0
        print("Loading crossvalidation from 'sample.cv'...")
        n.loadCrossValidation("sample.cv")
        print("Corpus:", n.crossValidationCorpus)
        # reset to something normal
        n = Network()
        n.addLayers(2, 2, 1)
        n.setInputs( [[0, 0], [0, 1], [1, 0], [1, 1]] )
        n.setTargets( [[0], [1], [1], [0]] )

    if ask("Do you want to run an XOR BACKPROP network in BATCH mode with cross validation?"):
        print("XOR Backprop batch mode: ..............................")
        n.crossValidationCorpus = ({"input" : [0.1, 0.1], "output" : [0.0]},
                                   {"input" : [0.2, 0.2], "output" : [0.0]},
                                   {"input" : [0.3, 0.3], "output" : [0.0]},
                                   {"input" : [0.4, 0.4], "output" : [0.0]},
                                   {"input" : [0.5, 0.5], "output" : [0.0]},
                                   {"input" : [0.6, 0.6], "output" : [0.0]},
                                   {"input" : [0.7, 0.7], "output" : [0.0]},
                                   {"input" : [0.8, 0.8], "output" : [0.0]},
                                   {"input" : [0.9, 0.9], "output" : [0.0]},
                                   )
        n.setBatch(1)
        n.reset()
        n.setEpsilon(0.5)
        n.setMomentum(.975)
        n.setReportRate(100)
        n.train()

    if ask("Do you want to run an XOR BACKPROP network in NON-BATCH mode with NON-HYPERBOLIC ERROR?"):
        print("XOR Backprop non-batch mode non-hyperbloc error...........")
        n.setBatch(0)
        n.hyperbolicError = 0
        n.reset()
        n.setEpsilon(0.5)
        n.setMomentum(.975)
        n.setReportRate(10)
        n.train()
        if ask("Do you want to run an XOR BACKPROP network in NON-BATCH mode with cross validation?"):
            print("XOR Backprop non-batch mode: ..........................")
            n.setBatch(0)
            n.reset()
            n.setEpsilon(0.5)
            n.setMomentum(.975)
            n.setReportRate(10)
            n.train()
        if ask("Do you want to test prop_from() method?"):
            n.prop_from([n.getLayer('input')])
            print("Output activations: ", n.getLayer('output').getActivations())
            print("Hidden activations: ", n.getLayer('hidden').getActivations())
            print("Now propagating directly from hidden layer...")
            n.prop_from([n.getLayer('hidden')])
            print("Output activations (should not have changed): ",  n.getLayer('output').getActivations())
            print("Hidden activations (should not have changed): ", n.getLayer('hidden').getActivations())
            print("Now setting hidden activations...")
            n.getLayer('hidden').setActivations([0.0, 0.0])
            print("Output activations: ", n.getLayer('output').getActivations())
            print("Hidden activations: ", n.getLayer('hidden').getActivations())
            print("Now propagating directly from hidden layer...")
            n.prop_from([n.getLayer('hidden')])
            print("Output activations: ",  n.getLayer('output').getActivations())
            print("Hidden activations: ", n.getLayer('hidden').getActivations())

            
    if ask("Do you want to test an AND network?"):
        print("Creating and running and network...")
        n = Network()
        n.setSeed(114366.64)
        n.add(Layer('input',2)) 
        n.add(Layer('output',1)) 
        n.connect('input','output') 
        n.setInputs([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]) 
        n.setTargets([[0.0],[0.0],[0.0],[1.0]]) 
        n.setEpsilon(0.5) 
        n.setTolerance(0.2) 
        n.setReportRate(5) 
        n.train() 
        if ask("Do you want to pickle the previous network?"):
            import pickle
            print("Pickling network...")
            print("Filename to save data (.pickle): ", end=" ")
            filename = sys.stdin.readline().strip()
            print("Setting log layer...")
            n.logLayer('input', 'input.log')
            # previously did not work if layer had a file pointer
            n.saveNetwork(filename)
            print("Loading file...")
            fp = open(filename + ".pickle")
            n = pickle.load(fp)
            fp.close()
            print("Sweeping...")
            n.setInteractive(1)
            n.sweep()

    if ask("Do you want to train an SRN to predict the seqences 1,2,3 and 1,3,2?"):
        print("SRN ...................................................")
        print("It is not possible to perfectly predict the sequences")
        print("1,2,3 and 1,3,2 because after a 1 either a 2 or 3 may")
        print("follow.")
        n = SRN()
        n.setSeed(114366.64)
        n.addSRNLayers(3,2,3)
        n.predict('input','output')
        seq1 = [1,0,0, 0,1,0, 0,0,1]
        seq2 = [1,0,0, 0,0,1, 0,1,0]
        n.setInputs([seq1, seq2])
        n.setSequenceType("random-continuous")
        n.setReportRate(75)
        n.setEpsilon(0.1)
        n.setMomentum(0)
        n.setBatch(1)
        n.setTolerance(0.25)
        n.setStopPercent(0.7)
        n.setResetEpoch(2000)
        n.setResetLimit(0)
        #n.setInteractive(1)
        #n.verbosity = 3
        temp = time.time()
        n.train()
        timer = time.time() - temp
        print("Time...", timer)
    if ask("Do you want to train an SRN to predict the seqences 1,2,3,2,1?"):
        print("SRN ...................................................")
        n = SRN()
        n.setSeed(114366.64)
        n.addSRNLayers(3,5,3)
        n.predict('input','output')
        seq1 = [1,0,0, 0,1,0, 0,0,1, 0,1,0, 1,0,0]
        n.setInputs([seq1])
        n.setSequenceType("ordered-continuous")
        n.setReportRate(75)
        n.setEpsilon(0.1)
        n.setMomentum(0)
        n.setBatch(1)
        n.setTolerance(0.25)
        n.setStopPercent(1.00)
        n.setResetEpoch(20000)
        n.setResetLimit(0)
        #n.verbosity = 3
        temp = time.time()
        n.train()
        timer = time.time() - temp
        print("Time...", timer)
        n.setLearning(0)
        n.setInteractive(1)
        n.sweep()
    if ask("Do you want to auto-associate on 3 bit binary patterns?"):
        print("Auto-associate ..........................................")
        n = Network()
        n.setSeed(114366.64)
        n.addThreeLayers(3,2,3)
        n.setInputs([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        n.associate('input','output')
        n.setReportRate(25)
        n.setEpsilon(0.1)
        n.setMomentum(0.9)
        n.setBatch(1)
        n.setTolerance(0.25)
        n.setStopPercent(0.9)
        n.setResetEpoch(1000)
        n.setResetLimit(2)
        n.train()

    if ask("Do you want to test a raam network?"):
        print("Raam ....................................................")
        # create network:
        raam = SRN()
        raam.setSeed(114366.64)
        raam.setPatterns({"john"  : [0, 0, 0, 1],
                          "likes" : [0, 0, 1, 0],
                          "mary"  : [0, 1, 0, 0],
                          "is" : [1, 0, 0, 0],
                          })
        size = len(raam.getPattern("john"))
        raam.addSRNLayers(size, size * 2, size)
        raam.add( Layer("outcontext", size * 2) )
        raam.connect("hidden", "outcontext")
        raam.associate('input', 'output')
        raam.associate('context', 'outcontext')
        raam.setInputs([ [ "john", "likes", "mary" ],
                         [ "mary", "likes", "john" ],
                         [ "john", "is", "john" ],
                         [ "mary", "is", "mary" ],
                         ])
        # network learning parameters:
        raam.setSequenceType("ordered-continuous")
        raam.setReportRate(10)
        raam.setEpsilon(0.1)
        raam.setMomentum(0.0)
        raam.setBatch(0)
        # ending criteria:
        raam.setTolerance(0.4)
        raam.setStopPercent(1.0)
        raam.setResetEpoch(5000)
        raam.setResetLimit(0)
        # this can be pickled, lambda's can't:
        # but don't do this... not enough training
        #def testDone(c, t): return raam["output"].correct != 1.0
        #raam.doWhile = testDone
        # train:
        temp = time.time()
        raam.train()
        timer = time.time() - temp
        print("Time...", timer)
        raam.setLearning(0)
        raam.setInteractive(1)
        if ask("Do you want to see (and save) the previous network?"):
            print("Filename to save network (.pickle): ", end=" ")
            filename = sys.stdin.readline().strip()
            raam.saveNetwork(filename)
            raam.sweep()
        if ask("Do you want to save weights of previous network?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            raam.saveWeights(filename)
            if ask("Do you want to try loading the weights you just saved (and sweep())?"):
                print("Loading standard style weights...")
                raam.loadWeights(filename)
                raam.sweep()
        if ask("Do you want to save weights of previous network in plain format?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            raam.saveWeights(filename, 'plain')
            if ask("Do you want to try loading the weights you just saved (and sweep())?"):
                print("Loading plain style weights...")
                raam.loadWeights(filename, 'plain')
                raam.sweep()
        if ask("Do you want to save weights of previous network in tlearn format?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            raam.saveWeights(filename, 'tlearn')
            if ask("Do you want to try loading the tlearn style weights you just saved (and sweep())?"):
                print("Loading tlearn style weights...")
                raam.loadWeights(filename, 'tlearn')
                raam.sweep()
                
    if ask("Do you want to train a network to both predict and auto-associate?"):
        print("SRN and auto-associate ...................................")
        n = SRN()
        n.setSeed(114366.64)
        n.addSRNLayers(3,3,3)
        n.add(Layer('assocInput',3))
        n.connect('hidden', 'assocInput')
        n.associate('input', 'assocInput')
        n.predict('input', 'output')
        n.setInputs([[1,0,0, 0,1,0, 0,0,1, 0,0,1, 0,1,0, 1,0,0]])
        n.setSequenceType("ordered-continuous")
        n.setReportRate(25)
        n.setEpsilon(0.1)
        n.setMomentum(0.3)
        n.setBatch(1)
        n.setTolerance(0.1)
        n.setStopPercent(0.7)
        n.setResetEpoch(2000)
        n.setResetLimit(0)
        n.setOrderedInputs(1)
        temp = time.time()
        n.train()
        timer = time.time() - temp
        print("Time...", timer)
        n.setLearning(0)
        n.setInteractive(1)
        if ask("Do you want to see (and save) the previous network?"):
            print("Filename to save network (.pickle): ", end=" ")
            filename = sys.stdin.readline().strip()
            n.saveNetwork(filename)
            n.sweep()
        if ask("Do you want to save weights of previous network?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            n.saveWeights(filename)
            if ask("Do you want to try loading the weights you just saved (and sweep())?"):
                print("Loading standard style weights...")
                n.loadWeights(filename)
                n.sweep()
        if ask("Do you want to save weights of previous network in plain format?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            n.saveWeights(filename, 'plain')
            if ask("Do you want to try loading the weights you just saved (and sweep())?"):
                print("Loading plain style weights...")
                n.loadWeights(filename, 'plain')
                n.sweep()
        if ask("Do you want to save weights of previous network in tlearn format?"):
            print("Filename to save weights (.wts): ", end=" ")
            filename = sys.stdin.readline().strip() + ".wts"
            n.saveWeights(filename, 'tlearn')
            if ask("Do you want to try loading the tlearn style weights you just saved (and sweep())?"):
                print("Loading tlearn style weights...")
                n.loadWeights(filename, 'tlearn')
                n.sweep()
                
    if ask("Do you want to change the size of a hidden layer in 3-3-3 network?"):
        print("Creating 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        n.display()
        print("New hidden layer size: ", end=" ")
        size = int(sys.stdin.readline().strip())
        if not type(size) == int:
            size = 0
        print("Changing size of hidden layer...")
        try:
            # exception thrown from changeSize in Connection class
            n.changeLayerSize('hidden', size)
        except LayerError as err:
            print(err)
        else:
            n.display()

    if ask("Do you want to test LayerError exceptions in Layer class?"):
        print("Trying to create a layer with size zero...")
        try:
            l = Layer('broken', 0)
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to change a layer of size three to size zero with changeSize()...")
        l = Layer('broken', 3)
        try:
            l.changeSize(0)
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to call getWinner for a layer attribute that doesn't exist...")
        l = Layer('broken', 3)
        try:
            l.getWinner('someAttribute')
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        # note that backprop() resets target flag
        print("Trying to call getWinner('target') where target has not been set...")
        l = Layer('broken', 3)
        try:
            l.getWinner('target')
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to call setActivations() for a layer whose activations have already been set...")
        l = Layer('broken', 3)
        l.setActivations(.5)
        try:
            l.setActivations(.2)
        except LayerError as err:
            print(err)
        else:
            print("No exception caught."     )
        print("Trying to call copyActivations() for a layer whose activations have already been set...")
        l = Layer('broken', 3)
        l.setActivations(.5)
        try:
            l.copyActivations([.2,.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to call copyActivations() with an array of incorrect size...")
        l = Layer('broken', 3)
        try:
            l.copyActivations([.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        l = Layer('broken', 3)
        try:
            l.copyActivations([.2,.2,.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to call setTargets() for a layer whose activations have already been set...")
        l = Layer('broken', 3)
        l.setTargets(.5)
        try:
            l.setTargets(.2)
        except LayerError as err:
            print(err)
        else:
            print("No exception caught."     )
        print("Trying to call copyTargets() for a layer whose activations have already been set...")
        l = Layer('broken', 3)
        l.setTargets(.5)
        try:
            l.copyTargets([.2,.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to call copyTargets() with an array of incorrect size...")
        l = Layer('broken', 3)
        try:
            l.copyTargets([.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        l = Layer('broken', 3)
        try:
            l.copyTargets([.2,.2,.2,.2])
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
                                 
    if ask("Do you want to test arrayify and unArrayify?"):
        print("Creating a 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        print("Arrayifying network...")
        array = n.arrayify()
        displayArray('Test Array', array, 6)
        print("unArrayifying network...")
        n.unArrayify(array)
        print("Now test unArrayify() for a short gene...")
        del array[-1:]
        try:
            n.unArrayify(array)
        except IndexError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Now test unArrayigy() for long gene...")
        array.extend([0.1,0.2])
        try:
            n.unArrayify(array)
        except IndexError as err:
            print(err)
        else:
            print("No exception caught.")

    if ask("Do you want to test load exception?"):
        print("Creating a 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        print("Loading input...")
        try:
            n.copyActivations(n["input"], n.inputs[0])
        except IndexError as err:
            print(err)
        else:
            print("No exception caught.")
        
    if ask("Do you want to test association and prediction exceptions?"):
        print("Creating a 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        n.setInputs([[1,1,1]])
        n.associate('hidden','output')
        print("Attempting to associate hidden and output layers...")
        try:
            n.step()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        n.resetFlags()
        n.associate('input','hidden')
        print("Attempting to associate input and hidden layers...")
        try:
            n.step()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Creating SRN network...")
        n = SRN()
        n.addSRNLayers(3,3,3)
        n.setInputs([[1,1,1]])
        n.predict('hidden','output')
        n.setSequenceType("ordered-continuous")
        print("Attempting to predict hidden and output layers...")
        try:
            n.step()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        n.resetFlags()
        n.predict('input','hidden')
        print("Attempting to predict input and hidden layers...")
        try:
            n.step()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")

    if ask("Do you want to test verifyInputs and verifyTargets?"):
        print("Creating a 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        try:
            n.verifyInputs()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")
        try:
            n.verifyTargets()
        except LayerError as err:
            print(err)
        else:
            print("No exception caught.")

    if ask("Do you want to test NetworkError exceptions?"):
        print("Creating a 3-3-3 network...")
        n = Network()
        n.addThreeLayers(3,3,3)
        print("Calling sweep() without inputs...")
        try:
            # NetworkError here indicates that no inputs have been set
            n.sweep()
        except NetworkError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Calling propagate on empty network...")
        n = Network()
        try:
            n.propagate()
        except NetworkError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Calling propagate with no connections...")
        n.add(Layer('input', 3))
        n.add(Layer('output',3))
        try:
            n.propagate()
        except NetworkError as err:
            print(err)
        else:
            print("No exception caught.")
        print("Trying to use a nonexistant connection...")
        try:
            n.getConnection('input','output')
        except NetworkError as err:
            print(err)
        else:
            print("No exception caught.")
        try:
            n.getWeights('input','output')
        except NetworkError as err:
            print(err)
        else:
            print("No exception caught.")

    if ask("Do you want to test SRN exceptions?"):
        print("Creating SRN network...")
        n = SRN()
        n.addSRNLayers(3,3,3)
        n.setInputs([[1,1,1]])
        n.setSequenceType("ordered-continuous")

    if ask("Do you want to test verifyArchitecture()?"):
        print("Creating normal 3-3-3 architecture...")
        n = Network()
        n.addThreeLayers(3,3,3)
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("ERROR: Caught exception. ", err)
        else:
            print("Good! Didn't catch exception.")
        print("Giving two layers the same name...")
        n.getLayer('hidden').name = 'input'
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")
        print("Creating 3-3-3 architecture with context layer...")
        n = SRN()
        n.addSRNLayers(3,3,3)
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("ERROR: Caught exception. ", err)
        else:
            print("Good! Didn't catch exception.")
        print("Connecting context to hidden layer again...")
        n.connect('context','hidden')
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")
        print("Creating an architecture with a cycle...")
        try:
            n = Network()
            n.add(Layer('1',1))
            n.add(Layer('2',1))
            n.add(Layer('3',1))
            n.add(Layer('4',1))
            n.add(Layer('5',1))
            n.connect('1','3')
            n.connect('2','3')
            n.connect('3','4')
            n.connect('4','3') #cycle
            n.connect('4','5')
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")
        print("Creating an architecture with an unconnected layer...")
        n = Network()
        n.add(Layer('1',1))
        n.add(Layer('2',1))
        n.add(Layer('3',1))
        n.connect('1','3')
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")
        print("Creating an architecture with two unconnected subarchitectures...")
        n = Network()
        n.add(Layer('1',1))
        n.add(Layer('2',1))
        n.add(Layer('3',1))
        n.add(Layer('4',1))
        n.connect('1','3')
        n.connect('2','4')
        try:
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")
        print("Creating an architecture with a connection between a layer and itself.")
        n = Network()
        try:
            n.add(Layer('1',1))
            n.add(Layer('2',1))
            n.connect('1','2')
            n.connect('2','2')
            n.verifyArchitecture()
        except Exception as err:
            print("Good! Caught exception. ", err)
        else:
            print("ERROR: Didn't catch exception.")

    if ask("Test the mapping input/target system?"):
            n = Network()
            n.add(Layer('input',2))
            n.add(Layer('hidden',2))
            n.add(Layer('output',1))
            n.connect('input','hidden')
            n.connect('hidden','output')
            n.verifyArchitecture()
            n.initialize()
            n.setEpsilon(0.5)
            n.setMomentum(.975)
            n.mapInput('input',0)
            n.mapTarget('output',0)
            n.setInputs([[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])
            n.setTargets([[0.0],
                          [1.0],
                          [1.0],
                          [0.0]])
            n.train()
    if ask("Test the new step() method?"):
        print("Creating 2-2-1 network...")
        n = Network()
        n.addThreeLayers(2,2,1)
        #n.setInteractive(1)
        print("Using step with arguments...")
        n.step(input = [1.0,0.0], output = [1.0])
        print("Using step without arguments...")
        n.getLayer('input').copyActivations([1.0,1.0])
        n.getLayer('output').copyTargets([0.0])
        n.step()
        print("Creating SRN Network...")
        n = SRN()
        n.addSRNLayers(3,3,3)
        n.setSequenceType("ordered-continuous")
        #n.setInteractive(1)
        print("Using step with arguments...")
        n.step(input = [1.0,0.0,0.0], output = [1.0, 0.0, 0.0], initContext = 1)
        n.step(input = [0.0,1.0,1.0], output = [0.0, 1.0, 1.0], initContext = 0)
        print("Using step without arguments...")
        n.getLayer('input').copyActivations([1.0,0.0,0.0])
        n.getLayer('output').copyTargets([1.0, 0.0, 0.0])
        n.step() # if you don't want context cleared, initContext = 0

    if ask("Additional tests?"):
        n = Network()
        n.addThreeLayers(2,2,1)
        try:
            n.setInputs([0.0,1.0])
        except Exception as err:
            print(err)
        try:
            n.setInputs([[[0.0]]])
        except Exception as err:
            print(err)
        try:
            n.setTargets([['two']])
        except Exception as err:
            print(err)
        n.patterned = 1
        try:
            n.setTargets([['two']])
        except Exception as err:
            print(err)
        else:
            print("Ok, No exception caught. (changed setTargets)")
        try:
            n = Network()
            n.add(Layer('1',2))
            n.add(Layer('2',2))
            n.add(Layer('3',1))
            n.connect('2','3')
            n.connect('1','2')
            n.verifyArchitecture()
            n.setInputs([[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])
            n.setTargets([[0.0],
                          [1.0],
                          [1.0],
                          [0.0]])
            n.setReportRate(100)
            n.setBatch(0)
            n.initialize()
            n.setEpsilon(0.5)
            n.setMomentum(.975)
            n.train()
        except Exception as err:
            print("Good!", err)
        try:
            n = Network()
            n.add(Layer('input',2))
            n.add(Layer('output',1))
            n.add(Layer('hidden',2))
            n.connect('input','hidden')
            n.connect('hidden','output')
            n.verifyArchitecture()
            n.initialize()
            n.setEpsilon(0.5)
            n.setMomentum(.975)
            #n.setInteractive(1)
            n.mapInput('input',0)
            n.mapTarget('output',0)
            n.setInputs([[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])
            n.setTargets([[0.0],
                          [1.0],
                          [1.0],
                          [0.0]])
            n.train()
        except Exception as err:
            print("Good!", err)

if __name__ == '__main__':
    main()
