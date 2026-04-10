import os
from numpy.typing import NDArray
from numpy import float64

def gen_narma_nmse_filepath(narma_degree : int | str) -> str:
    """
    Generates a filepath for writing results of NARMA evaluated with NMSE.
    Assumes a directory of the variable 'dictname' exists in the working directory.

    Parameters
    ----------
    narma_degree : Int value of the degree of NARMA

    Returns
    -------
    A string value of the filepath
    """

    dictname = "reservoir_narma_nmse"
    filename = "narma" + str(narma_degree) + "_nmse.csv"
    return os.path.join(os.getcwd(),dictname, filename)

def write_to_row(value : str, filepath : str) -> None:
    """
    Writes a string value + ';' to a file

    Parameters
    ----------
    value : String value to be written in the file
    filepath : String value of the filepath
    """

    with open(filepath, 'a') as file:
        file.write(value + ";")

def finish_row(filepath : str) -> None:
    """
    Finishes a row in a file.

    Parameters
    ----------
    filepath : String value of the filepath
    """
    with open(filepath, 'a') as file:
        file.write('0\n')

def NMSE(true_values : NDArray[float64] | list[float64], predicted_values : NDArray[float64] | list[float64]) -> float64:
    """
    Calculates the normalized mean squared error of two sets of values

    Parameters
    ----------
    true_values : A 2D Numpy array of the true values
    predicted_values : A 2D Numpy array of the predicted values
    
    Returns
    -------
    A float value of NMSE
    """

    return sum([(x-y)**2 for x,y in zip(true_values, predicted_values)])/sum([x**2 for x in true_values])