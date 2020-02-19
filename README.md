<div align="center">
  <a  href="http://www.openrec.ai/" target="_blank"><img src="https://github.com/ylongqi/openrec-web/blob/gh-pages/openrec.png?raw=true" width="60%"></a><br><br>
</div>

[**OpenRec**](http://www.openrec.ai/) is an open-source and modular library for neural network-inspired recommendation algorithms. Each recommender is modeled as a computational graph that consists of a structured ensemble of reusable modules connected through a set of well-defined interfaces. OpenRec is built to ease the process of extending and adapting state-of-the-art neural recommenders to heterogeneous recommendation scenarios, where different users', items', and contextual data sources need to be incorporated.

**For the structure and the design philosophy of OpenRec, please refer to the following paper published in WSDM'18:** 

[Longqi Yang](https://ylongqi.com/), Eugene Bagdasaryan, Joshua Gruenstein, Cheng-Kang Hsieh, and [Deborah Estrin](http://destrin.smalldata.io/). 2018. [OpenRec: A Modular Framework for Extensible and Adaptable Recommendation Algorithms.](https://ylongqi.com/paper/YangBGHE18.pdf) In Proceedings of WSDM’18, February 5–9, 2018, Marina Del Rey, CA, USA.  <img src="https://github.com/christinatsan/openrec-demo/blob/gh-pages/ccimage.png?raw=true" width="50">

**2020-02-17** OpenRec now uses Tensorflow 2.0 by default. Supports for Tensorflow 1.x are deprecated (all prior APIs have been moved to `openrec.tf1`). Currently supported recommendation algorithms include:
* BPR (`openrec.tf2.recommenders.BPR`): Bayesian Personalized Ranking (Rendle et al., 2009)
* WRMF (`openrec.tf2.recommenders.WRMF`): Weighted Regularized Matrix Factorization (Hu et al., 2008)
* UCML (`openrec.tf2.recommenders.UCML`): Collaborative Metric Learning with uniformly sampled triplets (Hsieh et al., 2017)
* GMF (`openrec.tf2.recommenders.GMF`): Generalized Matrix Factorization, a.k.a., Neural Collaborative Filtering (He et al., 2017)
* DLRM (`openrec.tf2.recommenders.DLRM`): Deep Learning Recommendation Model, developed by Facebook (Naumov et al., 2019)

**2019-07-12** OpenRec is being migrated to [Tensorflow 2.0](https://www.tensorflow.org/beta). Major changes to expect:

- All OpenRec modules will be compatible with [tf.keras.layers.Layer](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Layer), so that they can be used seamlessly with any Tensorflow 2.0 code base.
- All OpenRec models will be compatible with [tf.keras.Model](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model).
- All input data pipelines will be compatible with [tf.data.Dataset](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset) but are made much more friendly for recommendation models.
- Minimizing boilerplate while keeping the modularity and adaptability of OpenRec.

To get things started, we introduce OpenRec (Tensorflow 2.0) implementations of [deep learning recommendation model (DLRM)](https://github.com/facebookresearch/dlrm). Check out `tf2_examples/dlrm_criteo.py`.

To experiment with these new features, do `pip3 install .` inside the repo and then `import openrec.tf2`. You need to have Tensorflow 2.0 installed (Follow the instructions [here](https://www.tensorflow.org/beta)).

More examples, tutorials and documents will be available soon. Check out `tf2_examples/`.

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

All datasets can be downloaded from Google drive [here](https://drive.google.com/drive/folders/1taJ91txiMAWBMUtezc_N5gaYuTEpvW_e?usp=sharing).

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



