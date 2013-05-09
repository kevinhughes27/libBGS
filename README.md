libBGS
======

Background Subtraction Library

version 2.0

Original libBGS by Donovan Parks (http://dparks.wikidot.com/)
Updated version by Kevin Hughes

This a library of several common background subtraction algorithms. It was created from modifying the original libBGS by Donovan Parks, the major changes include updating to OpenCV 2.xx and making better use of the stl when appropriate. Several algorithms have also been added.

This is a release but also still a WIP. The algorithms work but I am working on adding serialization.

a few notes:
* you can build the lib using 'make' or open the .pro file with Qt Creator
* there is an example of how to use the lib in the top level directory
* Some of the methods support grayscale but they all support color
  * The methods the only support color are GMM methods, WrenGA and PratiMediod
  * This is easily fixed I just haven't had the time
* Serialization is a WIP
