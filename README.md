# What is optzer?

*optzer* is a python package for parameter optimization that can be used for any external program and for any target value obtained with the external program.

# Who made this?
* [Ryo KOBAYASHI](http://ryokbys.web.nitech.ac.jp/index.html)
* Assistant Professor at Department of Physical Science and Engineering, Nagoya Institute of Technology.

# Requirements and dependencies

The *optzer* requires the following:

- *python3*
- *docopt*
- *numpy*
- *scipy*

# Installation

It can be installed via *pip* as,
```bash
pip install optzer
```

You can install as a develop mode installation as,
```shell
git clone https://github.com/ryo.kbys/optzer.git ./optzer
cd optzer
python setup.py sdist
pip install -e .
```

If you can find `optzer` command in your system via `which optzer`, the installation should be successful.

# Usage

Please see [optzer documentation](http://ryokbys.web.nitech.ac.jp/contents/optzer_doc/) and examples included in this repository.

# Questions and requests

Please leave questions and requests on issues in the github repository page.

# License

This software is released under the MIT License, see the LICENSE.

# Acknowlegements

This software was developed at [Nagoya Institute of Technology](https://www.nitech.ac.jp/), and supported by JSPS KAKENHI Grand Numbers 21K04650, JP20H05290 and 22H04613 (Grant-in-Aid for Scientific Research on Innovative Areas *[Interface Ionics](https://interface-ionics.jp/)*).
