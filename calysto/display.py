
__all__ = ["display"]

try:
    import jupyter_kernel
    display = jupyter_kernel.get_jupyter().Display
except:
    from IPython.display import display
