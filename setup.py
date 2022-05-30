from setuptools import setup

setup(
   name='laat',
   version='1.0',
   description='Labelme Annotation Aiding Tool based on HSV color space & Morphological Operation',
   author='Seongwoo Kim',
   author_email='tjddn0402@naver.com',
   packages=['foo'],  # would be the same as name
   install_requires=['labelme', 'opencv-python', 'scikit-image', 'selenium'], #external packages acting as dependencies
)