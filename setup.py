from distutils.core import setup

setup(
    name='OpenRec',
    version='0.1dev',
    packages=['openrec'],
    license='BSD',
    long_description=open('README.md').read(),
    scripts=[],
    install_requires=[
        'tensorflow'
      ],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX :: Linux',
],
)
