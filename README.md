<div align="center">
  <a  href="http://www.openrec.ai/" target="_blank"><img src="https://github.com/ylongqi/openrec-web/blob/gh-pages/openrec.png?raw=true" width="60%"></a><br><br>
</div>

[![Docs](https://readthedocs.org/projects/openrec/badge/?version=latest)](http://openrec.readthedocs.io/en/latest/)

[**OpenRec**](http://www.openrec.ai/) is an open-source and modular library for neural network-inspired recommendation algorithms. Each recommender is modeled as a computational graph that consists of a structured ensemble of reusable modules connected through a set of well-defined interfaces. OpenRec is built to ease the process of extending and adapting state-of-the-art neural recommenders to heterogeneous recommendation scenarios, where different users', items', and contextual data sources need to be incorporated.

**For the structure and the design philosophy of OpenRec, please refer to the following paper published in WSDM'18:** 

[Longqi Yang](http://www.cs.cornell.edu/~ylongqi/), Eugene Bagdasaryan, Joshua Gruenstein, Cheng-Kang Hsieh, and [Deborah Estrin](http://destrin.smalldata.io/). 2018. [OpenRec: A Modular Framework for Extensible and Adaptable Recommendation Algorithms.](http://www.cs.cornell.edu/~ylongqi/paper/YangBGHE18.pdf) In Proceedings of WSDM’18, February 5–9, 2018, Marina Del Rey, CA, USA.  <img src="https://github.com/christinatsan/openrec-demo/blob/gh-pages/ccimage.png?raw=true" width="50">

**2018-08-31** Introducing new modular interfaces for OpenRec. Major changes:

- A new paradigm for defining, extending, and building recommenders.
  - Remove boilerplate class structure of recommenders.
  - Introduce a macro-based recommender construction paradigm.
  - Disentangle module construction and connection.
  - Support module construction directly using Tensorflow and Keras APIs.
- A more efficient and customizable pipeline for recommender training and evaluation.
  - A new Dataset class for complex data input.
  - A customizable ModelTrainer handling complex training/evaluation scenarios.
  - Caching mechanism to speed up evaluation of complex recommenders.
- Provide model training and evaluation examples for new interfaces.

More recommenders, examples, documents and tutorials are under development. Please checkout following events where we will present OpenRec new features:

*Strata Data Conference 2018:* https://conferences.oreilly.com/strata/strata-ny/public/schedule/detail/68280

*Recsys 2018:* https://recsys.acm.org/recsys18/tutorials/#content-tab-1-1-tab

**To use original openrec, simply import `openrec.legacy`**.

## Installation

Before installing OpenRec, please install [TensorFlow backend](https://www.tensorflow.org/install/) (GPU version is recommended). 

- **Install OpenRec from PyPI (recommended):**

```sh
pip install openrec
```

- **Install OpenRec from source code:**

First, clone OpenRec using `git`:

```sh
git clone https://github.com/ylongqi/openrec
```

Then, `cd` to the OpenRec folder and run the install command:

```sh
cd openrec
python setup.py install
```

## Dataset download

Use `download_dataset.sh` script.

## Get started

* [OpenRec website](http://www.openrec.ai/)
* [OpenRec legacy documents](http://openrec.readthedocs.io/en/latest/)
* [OpenRec legacy tutorials](https://github.com/ylongqi/openrec/tree/master/legacy_tutorials)
* [OpenRec legacy examples](https://github.com/ylongqi/openrec/tree/master/legacy_examples)

## How to cite

```
@inproceedings{yang2018openrec,
  title={OpenRec: A Modular Framework for Extensible and Adaptable Recommendation Algorithms},
  author={Yang, Longqi and Bagdasaryan, Eugene and Gruenstein, Joshua and Hsieh, Cheng-Kang and Estrin, Deborah},
  booktitle={Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining},
  year={2018},
  organization={ACM}
}
```

## License

[Apache License 2.0](LICENSE)

## Funders
<div>
  <img src="https://github.com/ylongqi/openrec-web/blob/gh-pages/imgs/funderlogonew.png?raw=true" width="20%"><br><br>
</div>



