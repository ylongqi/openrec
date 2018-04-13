from setuptools import find_packages, setup


setup(
    name='openrec',
    version='0.1.5',
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
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
