#-----------------------------------------------------------------------------
#  Copyright (C) 2014, Doug Blank <doug.blank@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

# based on code in IPython

import os
import json
import uuid
import hmac
import hashlib
import sys

PY3 = (sys.version_info[0] >= 3)

def read_secret_key():
    try:
        filename = os.path.expanduser("~/.ipython/profile_calico/security/notebook_secret")
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return "NOSECRET-FIXME"

encoder = json.encoder.JSONEncoder(sort_keys=True, indent=1)
key = read_secret_key()
if PY3:
    SECRET = key
    unicode_type = str
else:
    SECRET = unicode(key).encode("ascii")
    unicode_type = unicode

def convert(py_file):
    py_full_path = os.path.abspath(py_file)
    base_path, base_name = os.path.split(py_full_path)
    base, ext = os.path.splitext(base_name)
    code_list = open(py_full_path).readlines()
    nb_full_path = os.path.join(base_path, base + ".ipynb")
    ## ---------------------
    notebook = make_notebook(code_list)
    sign(notebook)
    save(notebook, nb_full_path)

def sign(notebook):
    notebook["metadata"]["signature"] = sign_notebook(notebook)

def save(notebook, filename):
    fp = open(filename, "w")
    fp.write(encoder.encode(notebook))
    fp.close()

def cast_bytes(s, encoding=None):
    if not isinstance(s, bytes):
        return encode(s, encoding)
    return s

def yield_everything(obj):
    if isinstance(obj, dict):
        for key in sorted(obj):
            value = obj[key]
            yield cast_bytes(key)
            for b in yield_everything(value):
                yield b
    elif isinstance(obj, (list, tuple)):
        for element in obj:
            for b in yield_everything(element):
                yield b
    elif isinstance(obj, unicode_type):
        yield obj.encode('utf8')
    else:
        yield unicode_type(obj).encode('utf8')

def sign_notebook(notebook, digestmod=hashlib.sha256):
    h = hmac.HMAC(SECRET, digestmod=digestmod)
    for b in yield_everything(notebook):
        h.update(b)
    return "sha256:" + h.hexdigest()

def make_notebook(code_list=None):
    notebook = {}
    notebook["metadata"] = {
        "name": "",
        "signature": "", ## to be signed later
    }
    notebook["nbformat"] = 3
    notebook["nbformat_minor"] = 0
    notebook["worksheets"] = [make_worksheet(code_list)]
    return notebook

def make_worksheet(code_list=None, cell_type="code", language="python"):
    worksheet = {}
    if code_list:
        worksheet["cells"] = [make_cell(code_list, cell_type="code", language="python")]
    else:
        worksheet["cells"] = []
    worksheet["metadata"] = {}
    return worksheet

def add_cell(notebook, cell):
    notebook["worksheets"][0]["cells"].append(cell)

def make_cell(code_list=None, cell_type="code", language="python"):
    cell = {
        "cell_type": cell_type, # markdown, code, 
        "collapsed": False,
        "metadata": {},
    }
    if cell_type == "code":
        cell["input"] = code_list
        cell["language"] = language
        cell["outputs"] = []
    elif cell_type == "markdown":
        cell["source"] = code_list
    return cell

if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        convert(arg)
