from setuptools import setup

setup(
   name='cooccurrence',
   version='1.0',
   description='A module to give cooccurrence word embeddings',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['cooccurrence'],  #same as name
   install_requires=[
       "numpy==1.17.2",
       "scikit-learn==0.20.3"
   ], #external packages as dependencies
)