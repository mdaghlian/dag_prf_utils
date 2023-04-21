
try: 
    import cortex 
except ImportError:
    raise ImportError('Error importing pycortex... Not a problem unless you want to use pycortex stuff')    

try:
    from nibabel.freesurfer.io import write_morph_data
except ImportError:
    raise ImportError('Error importing nibabel... Not a problem unless you want to use FSMaker')