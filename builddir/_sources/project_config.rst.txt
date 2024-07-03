.. _configuration_file:

Configuration File
==================

This is an example configuration file (`config.ini`) for the project. It contains various settings and options used throughout the script.

.. literalinclude:: ../config.ini
   :language: ini

Configuration Options
---------------------

Settings Section
~~~~~~~~~~~~~~~~
   
.. rubric:: Settings Section

- **rootDirPath** (`str`)
  
  Root directory path where the project resides.

- **dataset** (`str`)
  
  Name of the dataset to be used.

- **dimension** (`str`)
  
  Dimension parameter for analysis.

- **clip** (`str`)
  
  Clip value used in preprocessing.

- **model** (`str`)
  
  Name of the model to be used.

- **clasSeparator** (`str`)
  
  Classifier separator method.

- **aggregationMethod** (`str`)
  
  Aggregation method used.

- **task** (`str`)
  
  Task type for the model.

- **modalities** (`str`)
  
  Modalities included in the analysis.

- **threshold** (`float`)
  
  Threshold value for feature selection.

- **featureSelection** (`bool`)
  
  Whether feature selection is enabled (`True` or `False`).

