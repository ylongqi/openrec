from setuptools import find_packages, setup


setup(
    name='openrec',
    version='0.1.2',
    packages=find_packages(),
    description="An open-source and modular library for neural network-inspired recommendation algorithms",
    url="http://openrec.ai",
    license='Apache 2.0',
    author='Longqi Yang',
    author_email='ylongqi@cs.cornell.edu',
    install_requires=[
        'tqdm>=4.15.0',
        'numpy>=1.13.0',
        'termcolor>=1.1.0'
          ],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: Apache Software License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
