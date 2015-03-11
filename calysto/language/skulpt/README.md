**Skulpt Python Kernel** is the merging of [Skulpt](http://www.skulpt.org/) (a Python implementation that runs in the browser) with [Project Jupyter](http://jupyter.org/) (aka IPython). In addtion, it has an interface to [ProcessingJS](http://processingjs.org/). Skulpt scripts are entered into a cell, where they run independently from other cells. 

Because Skulpt Kernel uses [MetaKernel](https://github.com/Calysto/metakernel/blob/master/README.rst), it has a fully-supported set of "magics"---meta-commands for additional functionality. A list of magics can be seen at [MetaKernel Magics](https://github.com/Calysto/metakernel/blob/master/metakernel/magics/README.md).

Skulpt Kernel in use:

* [Video](https://www.youtube.com/watch?v=iSGXOU5C3sQ)
* [Example notebook from video](http://jupyter.cs.brynmawr.edu/hub/dblank/public/Examples/Skulpt%20Python%20Examples.ipynb)

You can install Skulpt Kernel with:

```
pip install --update skulpt-kernel
```

or in the system kernels with:

```
sudo pip install --update skulpt-kernel
```

Use it in the notebook with:

```
ipython notebook
```

and select `Skulpt Python` as the kernel for a new notebook.

Requires:

* ipython-3.0
* Python2 or Python3
* metakernel (installed with pip)
* calysto (installed with pip)

Skulpt Kernel supports:

* MetaKernel Magics
* processing
* turtle
