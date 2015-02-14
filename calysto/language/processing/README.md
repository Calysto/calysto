**Calysto Processing** is the merging of [ProcessingJS](http://processingjs.org/) with [Project Jupyter](http://jupyter.org/) (aka IPython). Processing Sketches are entered into Jupyter notebook cells, and even run in rendered notebooks. Sketches can be paused, and stepped one draw() at a time. 

Because Calysto Processing uses [MetaKernel](https://github.com/Calysto/metakernel/blob/master/README.rst), it has a fully-supported set of "magics"---meta-commands for additional functionality. A list of magics can be seen at [MetaKernel Magics](https://github.com/Calysto/metakernel/blob/master/metakernel/magics/README.md).

Calysto Processing in use:

* [CS110: Introduction to Computing](http://jupyter.cs.brynmawr.edu/hub/dblank/public/CS110%20Intro%20to%20Computing/2015/Syllabus.ipynb)
* [Video](https://www.youtube.com/watch?v=V4TzARh-ClY)

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
