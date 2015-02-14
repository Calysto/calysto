Calysto Processing is the merging of ProcessingJS with Jupyter (aka IPython).

Calysto Processing in use:

* [CS110: Introduction to Computing](http://jupyter.cs.brynmawr.edu/hub/dblank/public/CS110%20Intro%20to%20Computing/2015/Syllabus.ipynb)

You can install Calysto Processing with:

```
pip install --update calysto-processing
```

or in the system kernels with:

```
sudo pip install --update calysto-processing
```

Use it in the notebook with:

```
ipython notebook --kernel calysto_processing
```

Requires:

* ipython-3.0
* Python2 or Python3
* metakernel (installed automatically)

Calysto Processing supports:

* MetaKernel Magics
* All of ProcessingJS, plus pause/restart and stepper
