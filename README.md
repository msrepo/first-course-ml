# A First Course in Machine Learning: notes and Python code

My own study notes and Python ports of the MATLAB examples from *A First Course in Machine Learning* by Simon Rogers and Mark Girolami (Chapman & Hall/CRC). The official companion code is MATLAB only, so the scripts here are Python.

Notes are published with GitHub Pages from `docs/`.

The book itself is not included in this repository. Please get a copy from the publisher.

## Datasets

The files in `data/` are the companion datasets released by the book's authors, redistributed unchanged for convenience:

- Original source: <https://www.dcs.gla.ac.uk/~srogers/firstcourseml/firstcoursemldata.tar.gz>
- Book companion site: <https://www.dcs.gla.ac.uk/~srogers/firstcourseml/>

All credit for the data belongs to Simon Rogers and Mark Girolami. If you use it, cite the book and refer to the original source above.

The files are MATLAB `.mat` format. In Python:

```python
from scipy.io import loadmat
d = loadmat("data/olympics.mat")
```
