# What's optzer

*optzer* is a python package for parameter optimization that can be used for any external program and for any target value obtained with the external program.

# Who made this?
* [Ryo KOBAYASHI](http://ryokbys.web.nitech.ac.jp/index.html)
* Assistant Professor at Department of Physical Science and Engineering, Nagoya Institute of Technology.

# Requirements and dependencies

The *optzer* requires the following packages:

- *docopt*
- *numpy*
- *scipy*

# Installation

It can be installed via *pip* as,
```bash
$ pip install optzer
```

You can install as a develop mode installation as,
```shell
$ git clone https://github.com/ryo.kbys/optzer.git ./optzer
$ cd optzer
$ python setup.py sdist
$ pip install -e .
```

If you can find `optzer` command in your system via `which optzer`, the installation should be successful.


# LICENSE
This software is released under the MIT License, see the LICENSE.

