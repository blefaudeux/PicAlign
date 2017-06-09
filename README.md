PicAlign
========

PicAlign is a tool that can be used to align a set of images taken from a similar but not-quite aligned point of view (think handeld). It is particularly useful for creating "time-lapse" videos. PicAlign is another take on [FaceAlign](https://github.com/roblourens/facealign). It was for instance used to produce [this video](https://goo.gl/photos/QkE5GK7PBJrwJJUA7)

How it works
------------

PicAlign is a Python script that uses [opencv python bindings](http://opencv.willowgarage.com/wiki/), which requires >= Python 2.6. It detects interest points using Fast, and computes SIFT features to match successive pictures. It then wraps pictures around so that small misalignments and perspective shifts are corrected.

Usage
-----

Once python and opencv are installed, open config.py and set HCDIR to the folder containing your opencv installation's Haar cascade files.

Run alignPictures.py. It takes a required input directory parameter, and an optional output directory parameter. The output directory will be created if it does not already exist. By default, images will be output to the current directory. Output file names will be numbered starting with 0001.jpg.

    $ python src/alignPictures.py ../in-images

    $ python src/alignPictures.py ../in-images ../out-images

Eventual plans
--------------

* Do an overall optimization of the viewports, not just consecutive.

Common Errors/Solutions
-----------------------
    ImportError: No module named cv

**Solution**: You have not installed OpenCV or the OpenCV Python bindings

