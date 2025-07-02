import scipy.io
import numpy as np

def load_mat_file(filepath):
    """
    Load a MATLAB file and convert it into a Python dictionary,
    ensuring that MATLAB structs are loaded as Python dictionaries.
    """
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)

    # Remove the default MATLAB fields
    mat_data.pop('__header__', None)
    mat_data.pop('__version__', None)
    mat_data.pop('__globals__', None)

    # Convert MATLAB structs to Python dicts
    def convert_struct_to_dict(obj):
        if isinstance(obj, scipy.io.matlab.mio5_params.mat_struct):
            return {field: convert_struct_to_dict(getattr(obj, field)) for field in obj._fieldnames}
        elif isinstance(obj, np.ndarray) and obj.size == 1:
            return convert_struct_to_dict(obj.item())
        elif isinstance(obj, np.ndarray) and obj.dtype == 'object':
            return [convert_struct_to_dict(elem) for elem in obj]  # 返回列表而不是NumPy数组
            # return np.array([convert_struct_to_dict(elem) for elem in obj])
        else:
            return obj

    # Apply the conversion to the loaded data
    for key in list(mat_data.keys()):
        mat_data[key] = convert_struct_to_dict(mat_data[key])

    return mat_data

import numpy as np

def print_dict_shapes(data):
    """
    Print the shapes of numpy array values in a dictionary, or type information for other types of values.
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # If the value is a NumPy array, print its shape
            print(f"Key: {key}, Shape: {value.shape}")
        elif isinstance(value, dict):
            # If the value is another dictionary, recursively print its content
            print(f"Key: {key} (nested dictionary):")
            print_dict_shapes(value)
        else:
            # For other types, print the type of the value
            print(f"Key: {key}, Type: {type(value)}")

