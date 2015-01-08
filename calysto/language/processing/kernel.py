from metakernel import MetaKernel
from IPython.display import HTML
import sys
import re

class ProcessingKernel(MetaKernel):
    implementation = 'Processing'
    implementation_version = '1.0'
    language = 'java'
    language_version = '0.1'
    language_info = {
        'mimetype': 'text/x-processing',
        'name': 'java',
        # ------ If different from 'language':
        # 'codemirror_mode': {
        #    "version": 2,
        #    "name": "ipython"
        # }
        # 'pygments_lexer': 'language',
        # 'version'       : "x.y.z",
        'file_extension': '.pjs',
    }
    banner = "Processing kernel - evaluates Processing programs"
    canvas_id = 0
    keywords = ["@pjs", "Array", "ArrayList", "HALF_PI",
                    "HashMap", "Object", "PFont", "PGraphics", "PI", "PImage",
                    "PShape", "PVector", "PrintWriter", "QUARTER_PI", "String",
                    "TWO_PI", "XMLElement", "abs", "acos", "alpha", "ambient",
                    "ambientLight", "append", "applyMatrix", "arc", "arrayCopy",
                    "asin", "atan", "atan2", "background", "beginCamera",
                    "beginRaw", "beginRecord", "beginShape", "bezier",
                    "bezierDetail", "bezierPoint", "bezierTangent",
                    "bezierVertex", "binary", "blend", "blendColor", "blue",
                    "boolean", "boolean", "box", "break", "brightness", "byte",
                    "byte", "camera", "case", "ceil", "char", "char", "class",
                    "color", "color", "colorMode", "concat", "constrain",
                    "continue", "copy", "cos", "createFont", "createGraphics",
                    "createImage", "createInput", "createOutput", "createReader",
                    "createWriter", "cursor", "curve", "curveDetail",
                    "curvePoint", "curveTangent", "curveTightness", "curveVertex",
                    "day", "default", "degrees", "directionalLight", "dist",
                    "double", "draw", "ellipse", "ellipseMode", "else",
                    "emissive", "endCamera", "endRaw", "endRecord", "endShape",
                    "exit", "exp", "expand", "extends", "false", "fill", "filter",
                    "final", "float", "float", "floor", "focused", "font", "for",
                    "frameCount", "frameRate", "frameRate", "frustum", "get",
                    "globalKeyEvents", "green", "height", "hex", "hint", "hour",
                    "hue", "if", "image", "imageMode", "implements", "import",
                    "int", "int", "join", "key", "keyCode", "keyPressed",
                    "keyPressed", "keyReleased", "keyTyped", "lerp", "lerpColor",
                    "lightFalloff", "lightSpecular", "lights", "line", "link",
                    "loadBytes", "loadFont", "loadImage", "loadPixels",
                    "loadShape", "loadStrings", "log", "long", "loop", "mag",
                    "map", "match", "matchAll", "max", "millis", "min", "minute",
                    "modelX", "modelY", "modelZ", "month", "mouseButton",
                    "mouseClicked", "mouseDragged", "mouseMoved", "mouseOut",
                    "mouseOver", "mousePressed", "mousePressed", "mouseReleased",
                    "mouseX", "mouseY", "new", "nf", "nfc", "nfp", "nfs",
                    "noCursor", "noFill", "noLights", "noLoop", "noSmooth",
                    "noStroke", "noTint", "noise", "noiseDetail", "noiseSeed",
                    "norm", "normal", "null", "online", "open", "ortho", "param",
                    "pauseOnBlur", "perspective", "pixels[]", "pmouseX",
                    "pmouseY", "point", "pointLight", "popMatrix", "popStyle",
                    "pow", "preload", "print", "printCamera", "printMatrix",
                    "printProjection", "println", "private", "public",
                    "pushMatrix", "quad", "radians", "random", "randomSeed",
                    "rect", "rectMode", "red", "requestImage", "resetMatrix",
                    "return", "reverse", "rotate", "rotateX", "rotateY",
                    "rotateZ", "round", "saturation", "save", "saveBytes",
                    "saveFrame", "saveStream", "saveStrings", "scale", "screen",
                    "screenX", "screenY", "screenZ", "second", "selectFolder",
                    "selectInput", "selectOutput", "set", "setup", "shape",
                    "shapeMode", "shininess", "shorten", "sin", "size", "smooth",
                    "sort", "specular", "sphere", "sphereDetail", "splice",
                    "split", "splitTokens", "spotLight", "sq", "sqrt", "static",
                    "status", "str", "stroke", "strokeCap", "strokeJoin",
                    "strokeWeight", "subset", "super", "switch", "tan", "text",
                    "textAlign", "textAscent", "textDescent", "textFont",
                    "textLeading", "textMode", "textSize", "textWidth", "texture",
                    "textureMode", "this", "tint", "translate", "triangle",
                    "trim", "true", "unbinary", "unhex", "updatePixels", "vertex",
                    "void", "while", "width", "year"]

    def get_usage(self):
        return "This is the Processing kernel based on Processingjs.org."

    def do_execute_direct(self, code):
        self.canvas_id += 1
        """%%processing - run contents of cell as a Processing script"""

        env = {"code": repr(code)[1:] if sys.version.startswith('2') else repr(code),
               "id": self.canvas_id}
        code = """
<canvas id="canvas_%(id)s"></canvas><br/>
<div id="state_%(id)s">Running...</div><br/>
<button id="run_button_%(id)s" onclick="startSketch('%(id)s');" disabled>Run</button>
<button id="pause_button_%(id)s" onclick="stopSketch('%(id)s');">Pause</button>
<button id="setup_button_%(id)s" onclick="setupSketch('%(id)s');">setup()</button>
<button id="draw_button_%(id)s" onclick="drawSketch('%(id)s');">draw()</button>

<script>
function startSketch(id) {
    switchSketchState(id, true);
    document.getElementById("state_" + id).innerHTML = "Running...";
    document.getElementById("run_button_" + id).disabled = true;
    document.getElementById("pause_button_" + id).disabled = false;
    document.getElementById("setup_button_" + id).disabled = true;
    document.getElementById("draw_button_" + id).disabled = true;
}
      
function stopSketch(id) {
    switchSketchState(id, false);
    document.getElementById("state_" + id).innerHTML = "Stopped.";
    document.getElementById("run_button_" + id).disabled = false;
    document.getElementById("pause_button_" + id).disabled = true;
    document.getElementById("setup_button_" + id).disabled = false;
    document.getElementById("draw_button_" + id).disabled = false;
}

function drawSketch(id) {
    var processingInstance = Processing.getInstanceById("canvas_" + id);
    document.getElementById("state_" + id).innerHTML = "Drawing...";
    processingInstance.draw();  
    document.getElementById("state_" + id).innerHTML = "Drawing... Stopped.";
    document.getElementById("run_button_" + id).disabled = false;
    document.getElementById("pause_button_" + id).disabled = true;
    document.getElementById("setup_button_" + id).disabled = false;
    document.getElementById("draw_button_" + id).disabled = false;
}

function setupSketch(id) {
    var processingInstance = Processing.getInstanceById("canvas_" + id);
    document.getElementById("state_" + id).innerHTML = "Setting up...";
    processingInstance.setup();  
    document.getElementById("state_" + id).innerHTML = "Setting up... Stopped.";
    document.getElementById("run_button_" + id).disabled = false;
    document.getElementById("pause_button_" + id).disabled = true;
    document.getElementById("setup_button_" + id).disabled = false;
    document.getElementById("draw_button_" + id).disabled = false;
}

function switchSketchState(id, on) {
    var processingInstance = Processing.getInstanceById("canvas_" + id);
    if (on) {
        processingInstance.loop();  // call Processing loop() function
    } else {
        processingInstance.noLoop(); // stop animation, call noLoop()
    }
}

require(["http://cs.brynmawr.edu/gxk2013/examples/tools/alphaChannels/processing.js"], function () {
    var processingCode = %(code)s;
    var cc;
    try {
        cc = Processing.compile(processingCode);
    } catch (e) {
        console.log(e);
        
        cc = Processing.compile("println('Parse error: " + e.toString() + "');");
    }
    if (cc != undefined) {
        try {
            var processingInstance = new Processing("canvas_%(id)s", cc);
        } catch (e) {
            console.log(e);
            cc = Processing.compile("println('Runtime error: " + e.toString() + "');");
            var processingInstance = new Processing("canvas_%(id)s", cc);
        }
    }
});
</script>
""" % env
        html = HTML(code)
        self.Display(html)

    def get_completions(self, info):
        token = info["code"]
        return [command for command in self.keywords if command.startswith(token)]

    def get_kernel_help_on(self, info, level=0, none_on_fail=False):
        expr = info["code"]
        if expr in self.keywords:
            url = "http://processingjs.org/reference/%s_/" % expr
            try:
                import html2text
                import urllib
            except:
                return url
            html = urllib.urlopen(url).read()
            visible_text = html2text.html2text(html)
            pattern = re.compile("(.*?)### ", re.DOTALL)
            visible_text = re.sub(pattern, "### ", visible_text, 1)
            return visible_text
        elif none_on_fail:
            return None
        else:
            return "Sorry, no available help for '%s'" % expr


if __name__ == '__main__': 
    from IPython.kernel.zmq.kernelapp import IPKernelApp 
    IPKernelApp.launch_instance(kernel_class=ProcessingKernel) 
