from setuptools import find_packages, setup


setup(
    name='openrec',
    version='0.1.0-beta',
    packages=find_packages(),
    license='Apache 2.0',
    author='Longqi Yang',
    author_email='ylongqi@cs.cornell.edu',
    install_requires=[
        'tensorflow>=1.3.0',
        'tqdm>=4.15.0',
        'numpy>=1.13.0',
        'termcolor>=1.1.0'
          ],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: Apache 2.0',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
