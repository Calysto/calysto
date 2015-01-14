from io import BytesIO

from PIL import Image

from calysto.chart import GoogleChart

__all__ = ['GoogleChart']

def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""
   from IPython.core import display
   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()


# register display func with PNG formatter:
try:
   png_formatter = get_ipython().display_formatter.formatters['image/png']
   dpi = png_formatter.for_type(Image.Image, display_pil_image)
except:
   pass # not in an IPython client
