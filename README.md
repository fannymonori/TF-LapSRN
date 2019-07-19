# TF-LapSRN

Implementation is not finished!

Tensorflow implementation of LapSRN algorithm described in [1].

To run the training:
1. Download training dataset (DIV2K [2] [3])\
`bash download_trainds.sh`
2. Run the training for 4X scaling factor\
`python main.py --train --scale 4` \
or\
Set training images directory\
`python main.py --train --scale 4 --traindir /path/to/dir`

To run the test:\
`python3 main.py --test --scale 4`\
`python3 main.py --test --scale 4 --testimg /path/to/image`

To export file to .pb format:
1. Run the export script\
`python3 main.py --export --scale 4`

\
References

[1] Lai, W., Huang, J., Ahuja, N. and Yang, M. (2019).
Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks.
Available at: https://arxiv.org/abs/1710.01992 \
[2] Agustsson, E., Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.
Available at: http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf \
https://data.vision.ee.ethz.ch/cvl/DIV2K/