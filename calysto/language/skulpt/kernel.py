from metakernel import MetaKernel
from IPython.display import HTML, Javascript
import sys
import re

class SkulptPythonKernel(MetaKernel):
    implementation = 'Skulpt Python'
    implementation_version = '1.0'
    language = 'python'
    language_version = '0.1'
    language_info = {
        'mimetype': 'text/x-python',
        'name': 'python',
        'codemirror_mode': {
            "version": 2,
            "name": "text/x-python"
        },
        # 'pygments_lexer': 'language',
        # 'version'       : "x.y.z",
        'file_extension': '.py',
    }
    banner = "Skulpt Python kernel - evaluates Python programs in the browser"
    canvas_id = 0
    keywords = []

    def get_usage(self):
        return "This is a Skulpt Python kernel"

    def do_execute_direct(self, code):
        """%%processing - run contents of cell as a Skulpt Python script"""
        if code.strip() == "":
            return
        self.canvas_id += 1

        env = {"code": repr(code)[1:] if sys.version.startswith('2') else repr(code),
               "id": self.canvas_id}
        code = """
<div id='canvas_div_%(id)s'>
      <b>Canvas #%(id)s:</b><br/>
      <pre id='output_%(id)s' ></pre>
      <canvas id='canvas_%(id)s'></canvas><br/>
</div>
<script>
/*
var component = document.getElementById("output_%(id)s");
if (component != undefined)
    component.remove();
component = document.getElementById("canvas_div_%(id)s");
if (component != undefined)
    component.remove();

$("<div id='canvas_div_%(id)s'> \
      <b>Canvas #%(id)s:</b><br/>  \
      <pre id='output_%(id)s' ></pre> \
      <canvas id='canvas_%(id)s'></canvas><br/> \
   </div>").appendTo('body');    
*/
require([window.location.protocol + "//calysto.github.io/javascripts/skulpt/skulpt.min.js",
         window.location.protocol + "//calysto.github.io/javascripts/processing/processing.js"], function () {
  require([window.location.protocol + "//calysto.github.io/javascripts/skulpt/skulpt-stdlib.js"], function () {
    function outf_%(id)s(text) {
        var mypre = document.getElementById("output_%(id)s"); 
        mypre.innerHTML = mypre.innerHTML + text; 
    }
    function builtinRead(x) {
        if (Sk.builtinFiles === undefined || 
            Sk.builtinFiles["files"][x] === undefined)
            throw "File not found: '" + x + "'";
        return Sk.builtinFiles["files"][x];
    }
    function runit(id) { 
       var prog = %(code)s; 
       var mypre = document.getElementById("output_" + id); 
       mypre.innerHTML = ''; 
       Sk.canvas = "canvas_" + id;
       Sk.pre = "output_" + id;
       Sk.configure({output:outf_%(id)s, read:builtinRead, python3:false}); 
       var myPromise = Sk.misceval.asyncToPromise(function() {
           return Sk.importMainWithBody("<stdin>", false, prog, true);
       });
       myPromise.then(
        function(mod) {
            console.log('success');
        },
        function(err) {
            alert(err.toString());
        });
    }
    runit(%(id)s);
  });
});

</script>
""" % env
        html = HTML(code)
        self.Display(html)

    def get_completions(self, info):
        token = info["full_obj"]
        self.last_info = info
        return [command for command in self.keywords if command.startswith(token)]

    def get_kernel_help_on(self, info, level=0, none_on_fail=False):
        expr = info["full_obj"]
        self.last_info = info
        url = None
        if expr in self.keywords:
            return "Help on " + expr
        elif none_on_fail:
            return None
        else:
            return "Sorry, no available help for '%s'" % expr

if __name__ == '__main__': 
    from IPython.kernel.zmq.kernelapp import IPKernelApp 
    IPKernelApp.launch_instance(kernel_class=SkulptPythonKernel) 
