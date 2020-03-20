from setuptools import setup

setup(
   name='cooccurrence',
   version='1.0',
   description='A module to give cooccurrence word embeddings',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['cooccurrence'],  #same as name
   install_requires=[
       "scikit-learn==0.22.2.post1",
       "numpy==1.18.1"
   ], #external packages as dependencies
)