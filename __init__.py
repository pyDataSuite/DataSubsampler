from typing import Union, Iterable, Tuple
from h5py import Dataset
import numpy as np

from .methods import SubsampleMethods

modules = {}
def __register_module__( module_name, module_func ):
    global modules
    modules[ module_name ] = module_func

from . import minmax

def subsample(
        method: SubsampleMethods, 
        x_data: Union[Iterable, Dataset, np.array], 
        y_data: Union[Iterable, Dataset, np.array],
        target_len: int
    ) -> Tuple[Union[np.array,Dataset], Union[np.array,Dataset]]:
    """
    Produces x and y lists of sub-sampled data using the specified
    method.

    Args:
        method (SubsampleMethods): The type of sub-sampling to run
        x_data (Iterable, np.array, Dataset): The x data list
        y_data (Iterable, np.array, Dataset): The y data list
    
    Returns:
        Tuple containing x_subsampled, y_subsampled
    """

    # Ensure argument types are what we expect
    assert isinstance(method, SubsampleMethods)
    assert isinstance(x_data, Iterable)
    assert isinstance(y_data, Iterable)
    assert isinstance(target_len, int)

    # Turn generic lists into numpy arrays for further processing
    # if not isinstance(x_data, np.array) and not isinstance(x_data, Dataset):
    #     x_data = np.array( x_data )
    # if not isinstance(y_data, np.array) and not isinstance(y_data, Dataset):
    #     y_data = np.array( y_data )

    # Execute the sub-sampler
    # import pdb; pdb.set_trace()
    return modules[ method.value ]( x_data, y_data, target_len )