<p align="center"> <a href="https://lcl-correct.it"><img height="125" src="./assets/heptabot_logo.svg" alt="heptabot logo"></a> </p>

# heptabot – a deep learning text corrector
Powered by [Text-To-Text Transfer Transformer](https://github.com/google-research/text-to-text-transfer-transformer) model, `heptabot` is designed and built to be a practical example of a powerful user-friendly open-source error correction engine based on cutting-edge technology.

## Description
`heptabot` is trained on 4 similar but distinct tasks: *correction* (default), which is just general text paragraph-wise text correction, *jfleg*, which is sentence-wise correction based on [JFLEG](https://github.com/keisks/jfleg) competition, and *conll* and *bea*, based on [CoNLL-2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html) and [BEA 2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/) competitions respectively, which are also sentence-wise correction tasks, but more focused on grammar errors. While the core model of `heptabot` is T5, which solves all of the describes tasks, it also provides post-correction error classification for the *correction* task and uses parsing results to enhance the performance on *conll* and *bea* tasks. Note that while `heptabot` should in theory be able to correct English texts of any genre, it was trained specifically on student essays and thus works best on them.

## Performance
Here's how `heptabot` scores against state-of-the-art systems on some of the most common Grammar Error Correction tasks:
|CoNLL-2014|JFLEG|BEA 2019|
|--|--|--|
|<table> <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F<sub>0.5</sub></th></tr><tr><td><a href="https://www.aclweb.org/anthology/2020.bea-1.16/">Omelianchuk et al., 2020</a></td><td><b>78.2</b></td><td>41.5</td><td>66.5</td></tr><tr><td><a href="https://www.aclweb.org/anthology/2020.tacl-1.41/">Lichtarge et al., 2020</a></td><td>74.7</td><td>46.9</td><td><b>66.8</b></td></tr><tr style="border-top: thick solid"><td>`heptabot`, <i>Web</i></td><td>62.84</td><td>46.36</td><td>58.67</td></tr><tr><td>`heptabot`, <i>max</i></td><td>65.95</td><td><b>53.92</b></td><td>63.13</td></tr> </table>| <table> <tr><th>Model</th><th>GLEU</th></tr><tr><td><a href="https://www.aclweb.org/anthology/N19-1333/">Lichtarge et al., 2019</a></td><td>63.4</td></tr><tr><td><a href="https://www.aclweb.org/anthology/2020.tacl-1.41/">Lichtarge et al., 2020</a></td><td><b>64.9</b></td></tr><tr style="border-top: thick solid"><td>`heptabot`, <i>Web</i></td><td>60.15</td></tr><tr><td>`heptabot`, <i>max</i></td><td>63.74</td></tr> </table>|<table> <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F<sub>0.5</sub></th></tr><tr><td><a href="https://www.aclweb.org/anthology/2020.bea-1.16/">Omelianchuk et al., 2020</a></td><td>79.4</td><td>57.2</td><td>73.7</td></tr><tr><td><a href="https://www.aclweb.org/anthology/2020.tacl-1.41/">Lichtarge et al., 2020</a></td><td>75.4</td><td>64.7</td><td>73.0</td></tr><tr><td><a href="https://competitions.codalab.org/my/competition/submission/778969/detailed_results/">zxlxdf, 2021</a></td><td><b>85.86</b></td><td>53.48</td><td><b>76.59</b></td></tr><tr style="border-top: thick solid"><td>`heptabot`, <i>Web</i></td><td>60.81</td><td>63.39</td><td>61.31</td></tr><tr><td>`heptabot`, <i>max</i></td><td>64.83</td><td><b>70.85</b></td><td>65.95</td></tr> </table>|

The performance of current `heptabot` *Web* version measures as follows:
| Measure                            | Value     |
| ---------------------------------- | --------- |
| GPU load                           | 10.74 GiB |
| RAM load                           | 5.6 GiB   |
| Average time/text (300 words, GPU) | 20.8 secs |
| Average time/word (roughly)        | 0.07 secs |

## Access
The `Web` model of `heptabot` is available at [https://lcl-correct.it/](https://lcl-correct.it/).

## Install
`heptabot` is designed such that everyone could deploy it. We recommend to follow the [Install](https://github.com/lcl-hse/heptabot/blob/master/notebooks/Install.ipynb) notebook. Check if your system meets the requirements (has GPU installed and has enough resources to meet the requirements described in the **Description** section), then install `jupyterlab` (or `jupyter`) and download the notebook using the following commands:
```sh
pip install jupyterlab
curl -L "https://github.com/lcl-hse/heptabot/raw/master/notebooks/Install.ipynb" -o Install.ipynb
```
After that, launch Jupyter (`jupyter lab --ip=127.0.0.1 --port=8080 --allow-root`) and follow the instructions in the notebook you've just downloaded.

## Reproduce
Feel free to reproduce our research: to do so, follow the notebooks from the [retrain](https://github.com/lcl-hse/heptabot/blob/master/retrain/) folder. Please note that you have to get access to some of the datasets we used before obtaining them, so this portion of code is omitted.

## Contact us
In case you have any questions or suggestions regarding `heptabot` or our research, feel free to contact us at [itorubarov@hse.ru](mailto:itorubarov@hse.ru).