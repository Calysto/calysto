from distutils.command.install import install
from distutils.core import setup
from distutils import log
import os
import json
import sys

PY3 = sys.version_info[0] >= 3

kernel_json = {
    "argv": [sys.executable, 
	     "-m", "calysto.language.scheme.kernel", 
	     "-f", "{connection_file}"],
    "display_name": "Calysto Scheme %i" % (3 if PY3 else 2),
    "language": "scheme",
    "name": "calysto_scheme"
}

class install_with_kernelspec(install):
    def run(self):
        install.run(self)
        from IPython.kernel.kernelspec import install_kernel_spec
        from IPython.utils.tempdir import TemporaryDirectory
        from metakernel.utils.kernel import install_kernel_resources
        with TemporaryDirectory() as td:
            os.chmod(td, 0o755) # Starts off as 700, not user readable
            with open(os.path.join(td, 'kernel.json'), 'w') as f:
                json.dump(kernel_json, f, sort_keys=True)
            install_kernel_resources(td, resource="calysto")
            log.info('Installing kernel spec')
            try:
                install_kernel_spec(td, 'calysto_scheme', replace=True)
            except:
                install_kernel_spec(td, 'calysto_scheme', user=self.user, replace=True)


svem_flag = '--single-version-externally-managed'
if svem_flag in sys.argv:
    # Die, setuptools, die.
    sys.argv.remove(svem_flag)

setup(name='calysto_scheme',
      version='0.7.0',
      description='A Scheme kernel for Jupyter/IPython',
      long_description="A Scheme kernel for Jupyter/IPython, based on MetaKernel",
      url="https://github.com/Calysto/calysto/tree/master/calysto/language/scheme",
      author='Douglas Blank',
      author_email='doug.blank@gmail.com',
      #py_modules=['calysto_scheme'],
      install_requires=["metakernel", "calysto"],
      cmdclass={'install': install_with_kernelspec},
      classifiers = [
          'Framework :: IPython',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2',
      ]
)
