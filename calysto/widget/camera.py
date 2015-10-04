try:
    from ipywidgets.widgets import DOMWidget
except:
    from IPython.html.widgets import DOMWidget

from IPython.utils.traitlets import Unicode, Bytes, Instance
from IPython.display import Javascript

try:
    import jupyter_kernel
    display = jupyter_kernel.get_jupyter().Display
except:
    from IPython.display import display

import base64
from PIL import Image
from numpy import array, ndarray
import time

try:
    import StringIO
except ImportError:
    from io import StringIO

class Camera(DOMWidget):
    _view_name = Unicode('CameraView', sync=True)
    image_uri = Unicode('', sync=True)
    image_array = Instance(ndarray)
    ##image = Instance(Image.Image)

    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)
        self.data = None
        display(Javascript(self.get_javascript()))
        time.sleep(.1)

    def _image_uri_changed(self, name, new):
        head, self.data = new.split(',', 1)

    def get_image(self):
        return Image.open(StringIO.StringIO(base64.b64decode(self.data)))

    def get_array(self):
        im = self.get_image()
        self.image_array = array(im)

    def get_javascript(self):
        return """   require(["widgets/js/widget"], function(WidgetManager){
		    var CameraView = IPython.DOMWidgetView.extend({
		        render: function(){
		            // based on https://developer.mozilla.org/en-US/docs/WebRTC/taking_webcam_photos
		            var video        = $('<video>')[0];
		            var canvas       = $('<canvas>')[0];
		            var startbutton  = $('<button id = picture_button>Take Picture</button>')[0];
		            var width = 320;
		            var height = 0;
		            var that = this;
		
		            setTimeout(function() {that.$el.append(video).append(startbutton).append(canvas);}, 200);
		            //$(canvas).hide();
		            //window.vvv=video;
		            var streaming = false;
		            navigator.getMedia = ( navigator.getUserMedia ||
		                                 navigator.webkitGetUserMedia ||
		                                 navigator.mozGetUserMedia ||
		                                 navigator.msGetUserMedia);
		
		            navigator.getMedia({video: true, audio: false},
		                function(stream) {
		                  if (navigator.mozGetUserMedia) {
		                    video.mozSrcObject = stream;
		                  } else {
		                    var vendorURL = window.URL || window.webkitURL;
		                    video.src = vendorURL.createObjectURL(stream);
		                  }
		                  video.play();
		                },
		                function(err) {
		                  console.log("An error occured! " + err);
		                }
		            );
		
		            video.addEventListener('canplay', function(ev){
		                if (!streaming) {
		                  height = video.videoHeight / (video.videoWidth/width);
		                  video.setAttribute('width', width);
		                  video.setAttribute('height', height);
		                  canvas.setAttribute('width', width);
		                  canvas.setAttribute('height', height);
		                  streaming = true;
		                }
		            }, false);
		            function takepicture() {
		                canvas.width = width;
		                canvas.height = height;
		                canvas.getContext('2d').drawImage(video, 0, 0, width, height);
		                that.model.set('image_uri',canvas.toDataURL('image/png'));
		                that.touch();
		            }
		            startbutton.addEventListener('click', function(ev){
		                takepicture();
		                ev.preventDefault();
		            }, false);
		        },
		    });
		    
		    // Register the DatePickerView with the widget manager.
		    IPython.WidgetManager.register_widget_view('CameraView', CameraView);
		
		});"""

    def click(self):
        display(Javascript("document.getElementById('picture_button').click();"))
