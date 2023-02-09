import numpy as np

from . import __register_module__
from .methods import SubsampleMethods

def min_max( x_data: np.array, y_data: np.array, target_len: int ):
    """
    Performs the min_max subsampling method.
    """

    if x_data.shape != y_data.shape:
        raise IndexError( "X data and Y data are not the same shape" )

    # Find the size of the chunks that are needed
    chunk_size = round( len( x_data ) / target_len )
    
    # Get the chunks
    x_chunks = np.array_split( x_data, len( x_data ) / chunk_size )
    y_chunks = np.array_split( y_data, len( y_data ) / chunk_size )

    # Get the average x value this chunk
    x_subsampled = np.zeros( len( x_chunks ) * 2 )
    y_subsampled = np.zeros( len( y_chunks ) * 2 )
    i = 0
    for x_chunk, y_chunk in zip( x_chunks, y_chunks ):
        x_subsampled[ i ] = x_subsampled[ i+1 ] = np.mean( x_chunk )
        y_subsampled[ i ] = np.min( y_chunk )
        y_subsampled[ i+1 ] = np.max( y_chunk )
        i += 2
    
    return x_subsampled, y_subsampled


__register_module__( SubsampleMethods.MinMax.value, min_max )