try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup
import sys

svem_flag = '--single-version-externally-managed'
if svem_flag in sys.argv:
    # Die, setuptools, die.
    sys.argv.remove(svem_flag)

setup(name='calysto',
      version='1.0.6',
      description='Libraries and Languages for Python and IPython',
      long_description="Libraries and Languages for IPython and Python",
      author='Douglas Blank',
      author_email='doug.blank@gmail.com',
      url="https://github.com/Calysto/calysto",
      install_requires=['IPython', 'metakernel', 'svgwrite', 'cairosvg',
                        "ipywidgets"],
      packages=['calysto',
                'calysto.util',
                'calysto.widget',
                'calysto.chart',
                'calysto.simulations',
                'calysto.magics',
                'calysto.ai'],
      data_files = [("calysto/images", ["calysto/images/logo-64x64.png",
                                        "calysto/images/logo-32x32.png"])],
      classifiers = [
          'Framework :: IPython',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2',
          'Programming Language :: Scheme',
      ]
)
