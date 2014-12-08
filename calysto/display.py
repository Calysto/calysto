
__all__ = ["display"]

try:
    # use a metakernel Display:
    display = kernel.Display
except:
    # use the standard IPython display:
    from IPython.display import display
