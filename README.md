# ArControl-convert2-nwb
Convert ArControl behavioral data to "Neurodata Without Borders" nwb file.

## Install
Setup the python enviroment.
> pip install -U git+https://github.com/chenxinfeng4/ArControl-convert2-nwb

## Convert file
The easiest way.
> python -m arcontrol2nwb "Arcontrol/data/A/B/2022-1113-224711.txt"

Else go to the script. See the `play ground`.
```python
import arcontrol2nwb

arcfile = './2022-1113-224711.txt'
arcontrol2nwb.convert(arcfile)
```

## View the result
See the `play ground`.
```python
# In jupyter notebook
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget

io = NWBHDF5IO('./2022-1113-224711.nwb', mode='r')
nwb = io.read()

nwb2widget(nwb)
```
