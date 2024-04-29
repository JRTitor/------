Данный репозиоторий содержит дипломную работу по теме : Автоматическое формирование ответов в чате техподдержки с применением методов выравнивания языковых моделей
* Краткое описание присутвует в  [``` Презентация ```](<Презентация/Автоматическое формирование ответов в чате техподдержки с применением методов выравнивания языковых моделей.pdf>) или [```здесь```](https://docs.google.com/presentation/d/1sGo4M6T5CtPpvQD4fobayAcBBJ8OiwcNwOy13XwM3z8/edit#slide=id.g2ce8e6e14e7_2_75)
* Рабочее демо с переводом на русский  [```new_langchain_inference/interface.py```](new_langchain_inference/interface.py). Для его запуска потребуется создать файл в [```new_langchain_inference```](new_langchain_inference) с именем я ```env.json``` вида: ```{"langchain_api" : "ваш langchain токен"}```
* Негативные сообщения, на которые модель будет выдавать ответы, находятся [```memessages/negative_mess.csv```](memessages/negative_mess.csv)
* Системные промпты находятся [```prompts/prompts.csv```](prompts/prompts.csv)
* Для формирования отвевтов и записи испоьзовался [```new_langchain_inference/ollama_csv_writer.py```](new_langchain_inference/ollama_csv_writer.py)
* Размеченные ответы находятся в [```marked```](marked)
* Сводная таблица с оценками [```combined_stats```](new_langchain_inference/combined_stats.csv)

conda environment

```
# Name                    Version                   Build  Channel
_anaconda_depends         2024.02             py311_mkl_1
abseil-cpp                20211102.0           hd77b12b_0
accelerate                0.28.0                   pypi_0    pypi
aiobotocore               2.7.0           py311haa95532_0
aiohttp                   3.9.3           py311h2bbff1b_0
aioitertools              0.7.1              pyhd3eb1b0_0
aiosignal                 1.2.0              pyhd3eb1b0_0
alabaster                 0.7.12             pyhd3eb1b0_0
altair                    5.0.1           py311haa95532_0
anaconda-anon-usage       0.4.3           py311hfc23b7f_100
anaconda-catalogs         0.2.0           py311haa95532_0
anaconda-client           1.12.3          py311haa95532_0
anaconda-cloud-auth       0.1.4           py311haa95532_0
anaconda-navigator        2.5.2           py311haa95532_0
anaconda-project          0.11.1          py311haa95532_0
anthropic                 0.23.1                   pypi_0    pypi
anyio                     4.2.0           py311haa95532_0
aom                       3.6.0                hd77b12b_0
appdirs                   1.4.4              pyhd3eb1b0_0
archspec                  0.2.1              pyhd3eb1b0_0
argon2-cffi               21.3.0             pyhd3eb1b0_0
argon2-cffi-bindings      21.2.0          py311h2bbff1b_0
arrow                     1.2.3           py311haa95532_1
arrow-cpp                 14.0.2               ha81ea56_1
astroid                   2.14.2          py311haa95532_0
astropy                   5.3.4           py311hd7041d2_0
asttokens                 2.0.5              pyhd3eb1b0_0
async-lru                 2.0.4           py311haa95532_0
atomicwrites              1.4.0                      py_0
attrs                     23.1.0          py311haa95532_0
automat                   20.2.0                     py_0
autopep8                  1.6.0              pyhd3eb1b0_1
aws-c-auth                0.6.19               h2bbff1b_0
aws-c-cal                 0.5.20               h2bbff1b_0
aws-c-common              0.8.5                h2bbff1b_0
aws-c-compression         0.2.16               h2bbff1b_0
aws-c-event-stream        0.2.15               hd77b12b_0
aws-c-http                0.6.25               h2bbff1b_0
aws-c-io                  0.13.10              h2bbff1b_0
aws-c-mqtt                0.7.13               h2bbff1b_0
aws-c-s3                  0.1.51               h2bbff1b_0
aws-c-sdkutils            0.1.6                h2bbff1b_0
aws-checksums             0.1.13               h2bbff1b_0
aws-crt-cpp               0.18.16              hd77b12b_0
aws-sdk-cpp               1.10.55              hd77b12b_0
babel                     2.11.0          py311haa95532_0
backports                 1.1                pyhd3eb1b0_0
backports.functools_lru_cache 1.6.4              pyhd3eb1b0_0
backports.tempfile        1.0                pyhd3eb1b0_1
backports.weakref         1.0.post1                  py_1
bcrypt                    3.2.0           py311h2bbff1b_1
beautifulsoup4            4.12.2          py311haa95532_0
binaryornot               0.4.4              pyhd3eb1b0_1
bitsandbytes              0.43.0                   pypi_0    pypi
black                     23.11.0         py311haa95532_0
blas                      1.0                         mkl
bleach                    4.1.0              pyhd3eb1b0_0
blinker                   1.6.2           py311haa95532_0
blosc                     1.21.3               h6c2663c_0
bokeh                     3.3.4           py311h746a85d_0
boltons                   23.0.0          py311haa95532_0
boost-cpp                 1.82.0               h59b6b97_2
botocore                  1.31.64         py311haa95532_0
bottleneck                1.3.7           py311hd7041d2_0
brotli                    1.0.9                h2bbff1b_7
brotli-bin                1.0.9                h2bbff1b_7
brotli-python             1.0.9           py311hd77b12b_7
bzip2                     1.0.8                he774522_0
c-ares                    1.19.1               h2bbff1b_0
c-blosc2                  2.12.0               h2f4ed9d_0
ca-certificates           2024.2.2             h56e8100_0    conda-forge
cachetools                4.2.2              pyhd3eb1b0_0
cairo                     1.16.0               haedb8bc_5
catboost                  1.2.3           py311h1ea47a8_2    conda-forge
certifi                   2024.2.2           pyhd8ed1ab_0    conda-forge
cffi                      1.16.0          py311h2bbff1b_0
cfitsio                   3.470                h2bbff1b_7
chardet                   3.0.4                    pypi_0    pypi
charls                    2.2.0                h6c2663c_0
charset-normalizer        2.0.4              pyhd3eb1b0_0
click                     8.1.7           py311haa95532_0
cloudpickle               2.2.1           py311haa95532_0
clyent                    1.2.2           py311haa95532_1
cohere                    5.2.2                    pypi_0    pypi
colorama                  0.4.6           py311haa95532_0
colorcet                  3.0.1           py311haa95532_0
comm                      0.1.2           py311haa95532_0
conda                     24.1.2          py311h1ea47a8_0    conda-forge
conda-build               24.1.2          py311haa95532_0
conda-content-trust       0.2.0           py311haa95532_0
conda-index               0.4.0              pyhd3eb1b0_0
conda-libmamba-solver     24.1.0             pyhd3eb1b0_0
conda-pack                0.6.0              pyhd3eb1b0_0
conda-package-handling    2.2.0           py311haa95532_0
conda-package-streaming   0.9.0           py311haa95532_0
conda-repo-cli            1.0.75          py311haa95532_0
conda-token               0.4.0              pyhd3eb1b0_0
conda-verify              3.4.2                      py_1
console_shortcut          0.1.1                         4
constantly                23.10.4         py311haa95532_0
contourpy                 1.2.0           py311h59b6b97_0
cookiecutter              2.5.0           py311haa95532_0
cryptography              42.0.2          py311h89fc84f_0
cssselect                 1.2.0           py311haa95532_0
cuda-cccl                 12.4.99                       0    nvidia
cuda-cudart               12.1.105                      0    nvidia
cuda-cudart-dev           12.1.105                      0    nvidia
cuda-cupti                12.1.105                      0    nvidia
cuda-libraries            12.1.0                        0    nvidia
cuda-libraries-dev        12.1.0                        0    nvidia
cuda-nvrtc                12.1.105                      0    nvidia
cuda-nvrtc-dev            12.1.105                      0    nvidia
cuda-nvtx                 12.1.105                      0    nvidia
cuda-opencl               12.4.99                       0    nvidia
cuda-opencl-dev           12.4.99                       0    nvidia
cuda-profiler-api         12.4.99                       0    nvidia
cuda-runtime              12.1.0                        0    nvidia
curl                      8.5.0                he2ea4bf_0
cycler                    0.11.0             pyhd3eb1b0_0
cytoolz                   0.12.2          py311h2bbff1b_0
dask                      2023.11.0       py311haa95532_0
dask-core                 2023.11.0       py311haa95532_0
dataclasses-json          0.6.4                    pypi_0    pypi
datasets                  2.18.0             pyhd8ed1ab_0    conda-forge
datashader                0.16.0          py311haa95532_0
dav1d                     1.2.1                h2bbff1b_0
debugpy                   1.6.7           py311hd77b12b_0
decorator                 5.1.1              pyhd3eb1b0_0
defusedxml                0.7.1              pyhd3eb1b0_0
diff-match-patch          20200713           pyhd3eb1b0_0
dill                      0.3.7           py311haa95532_0
distributed               2023.11.0       py311haa95532_0
distro                    1.8.0           py311haa95532_0
docstring-to-markdown     0.11            py311haa95532_0
docutils                  0.18.1          py311haa95532_3
einops                    0.7.0                    pypi_0    pypi
entrypoints               0.4             py311haa95532_0
et_xmlfile                1.1.0           py311haa95532_0
evaluate                  0.4.1                    pypi_0    pypi
executing                 0.8.3              pyhd3eb1b0_0
expat                     2.6.1                h63175ca_0    conda-forge
fastapi                   0.110.1                  pypi_0    pypi
fastavro                  1.9.4                    pypi_0    pypi
filelock                  3.13.1          py311haa95532_0
flake8                    6.0.0           py311haa95532_0
flask                     2.2.5           py311haa95532_0
fmt                       9.1.0                h6d14046_0
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      3.000                h77eed37_0    conda-forge
font-ttf-source-code-pro  2.038                h77eed37_0    conda-forge
font-ttf-ubuntu           0.83                 h77eed37_1    conda-forge
fontconfig                2.14.1               h9c4af85_2
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
fonttools                 4.25.0             pyhd3eb1b0_0
freetype                  2.12.1               ha860e81_0
fribidi                   1.0.10               h8d14728_0    conda-forge
frozenlist                1.4.0           py311h2bbff1b_0
fsspec                    2023.10.0       py311haa95532_0
future                    0.18.3          py311haa95532_0
gensim                    4.3.0           py311heda8569_0
getopt-win32              0.1                  hcfcfb64_1    conda-forge
gflags                    2.2.2                hd77b12b_1
giflib                    5.2.1                h8cc25b3_3
gitdb                     4.0.7              pyhd3eb1b0_0
gitpython                 3.1.43                   pypi_0    pypi
glib                      2.78.4               hd77b12b_0
glib-tools                2.78.4               hd77b12b_0
glog                      0.5.0                hd77b12b_1
gmpy2                     2.1.2           py311h7f96b67_0
googletrans               4.0.0rc1                 pypi_0    pypi
graphite2                 1.3.14               hd77b12b_1
graphviz                  4.0.0                had6c3a3_0    conda-forge
greenlet                  3.0.1           py311hd77b12b_0
grpc-cpp                  1.48.2               hfe90ff0_1
gts                       0.7.6                h6b5321d_4    conda-forge
h11                       0.9.0                    pypi_0    pypi
h2                        3.2.0                    pypi_0    pypi
h5py                      3.9.0           py311h4e0e482_0
harfbuzz                  4.3.0                hb646838_2
hdf5                      1.12.1               h51c971a_3
heapdict                  1.0.1              pyhd3eb1b0_0
holoviews                 1.18.3          py311haa95532_0
hpack                     3.0.0                    pypi_0    pypi
hstspreload               2024.4.1                 pypi_0    pypi
httpcore                  0.9.1                    pypi_0    pypi
httpx                     0.13.3                   pypi_0    pypi
httpx-sse                 0.4.0                    pypi_0    pypi
huggingface_hub           0.22.0             pyhd8ed1ab_0    conda-forge
hvplot                    0.9.2           py311haa95532_0
hyperframe                5.2.0                    pypi_0    pypi
hyperlink                 21.0.0             pyhd3eb1b0_0
icc_rt                    2022.1.0             h6049295_2
icu                       73.1                 h6c2663c_0
idna                      2.10                     pypi_0    pypi
imagecodecs               2023.1.23       py311he6ff3c7_0
imageio                   2.33.1          py311haa95532_0
imagesize                 1.4.1           py311haa95532_0
imbalanced-learn          0.11.0          py311haa95532_1
importlib-metadata        7.0.1           py311haa95532_0
importlib_metadata        7.0.1                hd3eb1b0_0
incremental               22.10.0            pyhd3eb1b0_0
inflection                0.5.1           py311haa95532_0
iniconfig                 1.1.1              pyhd3eb1b0_0
intake                    0.6.8           py311haa95532_0
intel-openmp              2023.1.0         h59b6b97_46320
intervaltree              3.1.0              pyhd3eb1b0_0
ipykernel                 6.28.0          py311haa95532_0
ipython                   8.20.0          py311haa95532_0
ipython_genutils          0.2.0              pyhd3eb1b0_1
ipywidgets                7.6.5              pyhd3eb1b0_2
isort                     5.9.3              pyhd3eb1b0_0
itemadapter               0.3.0              pyhd3eb1b0_0
itemloaders               1.1.0           py311haa95532_0
itsdangerous              2.0.1              pyhd3eb1b0_0
jaraco.classes            3.2.1              pyhd3eb1b0_0
jedi                      0.18.1          py311haa95532_1
jellyfish                 1.0.1           py311h36a85e1_0
jinja2                    3.1.3           py311haa95532_0
jmespath                  1.0.1           py311haa95532_0
joblib                    1.2.0           py311haa95532_0
jpeg                      9e                   h2bbff1b_1
jq                        1.6                  haa95532_1
json5                     0.9.6              pyhd3eb1b0_0
jsonpatch                 1.33                     pypi_0    pypi
jsonpointer               2.1                pyhd3eb1b0_0
jsonschema                4.19.2          py311haa95532_0
jsonschema-specifications 2023.7.1        py311haa95532_0
jupyter                   1.0.0           py311haa95532_9
jupyter-lsp               2.2.0           py311haa95532_0
jupyter_client            8.6.0           py311haa95532_0
jupyter_console           6.6.3           py311haa95532_0
jupyter_core              5.5.0           py311haa95532_0
jupyter_events            0.8.0           py311haa95532_0
jupyter_server            2.10.0          py311haa95532_0
jupyter_server_terminals  0.4.4           py311haa95532_1
jupyterlab                4.0.11          py311haa95532_0
jupyterlab-variableinspector 3.1.0           py311haa95532_0
jupyterlab_pygments       0.1.2                      py_0
jupyterlab_server         2.25.1          py311haa95532_0
jupyterlab_widgets        3.0.9           py311haa95532_0
keyring                   23.13.1         py311haa95532_0
kiwisolver                1.4.4           py311hd77b12b_0
krb5                      1.20.1               h5b6d351_0
langchain                 0.1.14                   pypi_0    pypi
langchain-anthropic       0.1.6                    pypi_0    pypi
langchain-cli             0.0.21                   pypi_0    pypi
langchain-cohere          0.1.0                    pypi_0    pypi
langchain-community       0.0.31                   pypi_0    pypi
langchain-core            0.1.40                   pypi_0    pypi
langchain-experimental    0.0.56                   pypi_0    pypi
langchain-openai          0.1.1                    pypi_0    pypi
langchain-text-splitters  0.0.1                    pypi_0    pypi
langgraph                 0.0.32                   pypi_0    pypi
langserve                 0.0.51                   pypi_0    pypi
langsmith                 0.1.40                   pypi_0    pypi
lazy-object-proxy         1.6.0           py311h2bbff1b_0
lazy_loader               0.3             py311haa95532_0
lcms2                     2.12                 h83e58a3_0
lerc                      3.0                  hd77b12b_0
libaec                    1.0.4                h33f27b4_1
libarchive                3.6.2                hb62f4d4_2
libavif                   0.11.1               h2bbff1b_0
libboost                  1.82.0               h3399ecb_2
libbrotlicommon           1.0.9                h2bbff1b_7
libbrotlidec              1.0.9                h2bbff1b_7
libbrotlienc              1.0.9                h2bbff1b_7
libclang                  14.0.6          default_hb5a9fac_1
libclang13                14.0.6          default_h8e68704_1
libcublas                 12.1.0.26                     0    nvidia
libcublas-dev             12.1.0.26                     0    nvidia
libcufft                  11.0.2.4                      0    nvidia
libcufft-dev              11.0.2.4                      0    nvidia
libcurand                 10.3.5.119                    0    nvidia
libcurand-dev             10.3.5.119                    0    nvidia
libcurl                   8.5.0                h86230a5_0
libcusolver               11.4.4.55                     0    nvidia
libcusolver-dev           11.4.4.55                     0    nvidia
libcusparse               12.0.2.55                     0    nvidia
libcusparse-dev           12.0.2.55                     0    nvidia
libdeflate                1.17                 h2bbff1b_1
libevent                  2.1.12               h56d1f94_1
libexpat                  2.6.1                h63175ca_0    conda-forge
libffi                    3.4.4                hd77b12b_0
libgd                     2.3.3                ha43c60c_1
libglib                   2.78.4               ha17d25a_0
libiconv                  1.16                 h2bbff1b_2
libjpeg-turbo             2.1.4                hcfcfb64_0    conda-forge
liblief                   0.12.3               hd77b12b_0
libmamba                  1.5.6                hcd6fe79_0
libmambapy                1.5.6           py311h77c03ed_0
libnpp                    12.0.2.50                     0    nvidia
libnpp-dev                12.0.2.50                     0    nvidia
libnvjitlink              12.1.105                      0    nvidia
libnvjitlink-dev          12.1.105                      0    nvidia
libnvjpeg                 12.1.1.14                     0    nvidia
libnvjpeg-dev             12.1.1.14                     0    nvidia
libpng                    1.6.39               h8cc25b3_0
libpq                     12.17                h906ac69_0
libprotobuf               3.20.3               h23ce68f_0
libretranslatepy          2.1.1                    pypi_0    pypi
libsodium                 1.0.18               h62dcd97_0
libsolv                   0.7.24               h23ce68f_0
libspatialindex           1.9.3                h6c2663c_0
libssh2                   1.10.0               he2ea4bf_2
libthrift                 0.15.0               h4364b78_2
libtiff                   4.5.1                hd77b12b_0
libuv                     1.47.0               hcfcfb64_0    conda-forge
libwebp-base              1.3.2                h2bbff1b_0
libxml2                   2.10.4               h0ad7f3c_1
libxslt                   1.1.37               h2bbff1b_1
libzopfli                 1.0.3                ha925a31_0
linkify-it-py             2.0.0           py311haa95532_0
llvmlite                  0.42.0          py311hf2fb9eb_0
locket                    1.0.0           py311haa95532_0
lxml                      4.9.3           py311h09808a7_0
lz4                       4.3.2           py311h2bbff1b_0
lz4-c                     1.9.4                h2bbff1b_0
lzo                       2.10                 he774522_2
m2-msys2-runtime          2.5.0.17080.65c939c               3
m2-patch                  2.7.5                         2
m2w64-libwinpthread-git   5.0.0.4634.697f757               2
markdown                  3.4.1           py311haa95532_0
markdown-it-py            2.2.0           py311haa95532_1
markupsafe                2.1.3           py311h2bbff1b_0
marshmallow               3.21.1                   pypi_0    pypi
matplotlib                3.8.0           py311haa95532_0
matplotlib-base           3.8.0           py311hf62ec03_0
matplotlib-inline         0.1.6           py311haa95532_0
mccabe                    0.7.0              pyhd3eb1b0_0
mdit-py-plugins           0.3.0           py311haa95532_0
mdurl                     0.1.0           py311haa95532_0
menuinst                  2.0.2           py311hd77b12b_0
mistune                   2.0.4           py311haa95532_0
mkl                       2023.1.0         h6b88ed4_46358
mkl-service               2.4.0           py311h2bbff1b_1
mkl_fft                   1.3.8           py311h2bbff1b_0
mkl_random                1.2.4           py311h59b6b97_0
more-itertools            10.1.0          py311haa95532_0
mpc                       1.1.0                h7edee0f_1
mpfr                      4.0.2                h62dcd97_1
mpir                      3.0.0                hec2e145_1
mpmath                    1.3.0           py311haa95532_0
msgpack-python            1.0.3           py311h59b6b97_0
msys2-conda-epoch         20160418                      1
multidict                 6.0.4           py311h2bbff1b_0
multipledispatch          0.6.0           py311haa95532_0
multiprocess              0.70.15         py311ha68e1ae_1    conda-forge
munkres                   1.1.4                      py_0
mypy                      1.8.0           py311h2bbff1b_0
mypy_extensions           1.0.0           py311haa95532_0
navigator-updater         0.4.0           py311haa95532_1
nbclient                  0.8.0           py311haa95532_0
nbconvert                 7.10.0          py311haa95532_0
nbformat                  5.9.2           py311haa95532_0
nest-asyncio              1.6.0           py311haa95532_0
networkx                  3.1             py311haa95532_0
nltk                      3.8.1           py311haa95532_0
notebook                  7.0.8           py311haa95532_0
notebook-shim             0.2.3           py311haa95532_0
numba                     0.59.0          py311hf62ec03_0
numexpr                   2.8.7           py311h1fcbade_0
numpy                     1.26.4          py311hdab7c0b_0
numpy-base                1.26.4          py311hd01c5d8_0
numpydoc                  1.5.0           py311haa95532_0
openai                    1.16.2                   pypi_0    pypi
openjpeg                  2.4.0                h4fc8c34_0
openpyxl                  3.0.10          py311h2bbff1b_0
openssl                   3.2.1                hcfcfb64_1    conda-forge
orc                       1.7.4                h623e30f_1
orjson                    3.10.0                   pypi_0    pypi
overrides                 7.4.0           py311haa95532_0
packaging                 23.2                     pypi_0    pypi
pandas                    2.1.4           py311hf62ec03_0
pandocfilters             1.5.0              pyhd3eb1b0_0
panel                     1.3.8           py311haa95532_0
pango                     1.50.7               h66df5b2_0    conda-forge
param                     2.0.2           py311haa95532_0
paramiko                  2.8.1              pyhd3eb1b0_0
parsel                    1.8.1           py311haa95532_0
parso                     0.8.3              pyhd3eb1b0_0
partd                     1.4.1           py311haa95532_0
pathlib                   1.0.1              pyhd3eb1b0_1
pathspec                  0.10.3          py311haa95532_0
patsy                     0.5.3           py311haa95532_0
pcre2                     10.42                h0ff8eda_0
pexpect                   4.8.0              pyhd3eb1b0_3
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    10.2.0          py311h2bbff1b_0
pip                       23.3.1          py311haa95532_0
pixman                    0.43.4               h63175ca_0    conda-forge
pkce                      1.0.3           py311haa95532_0
pkginfo                   1.9.6           py311haa95532_0
platformdirs              3.10.0          py311haa95532_0
plotly                    5.9.0           py311haa95532_0
pluggy                    1.0.0           py311haa95532_1
ply                       3.11            py311haa95532_0
powershell_shortcut       0.0.1                         3
prometheus_client         0.14.1          py311haa95532_0
prompt-toolkit            3.0.43          py311haa95532_0
prompt_toolkit            3.0.43               hd3eb1b0_0
protego                   0.1.16                     py_0
protobuf                  3.20.3          py311hd77b12b_0
psutil                    5.9.0           py311h2bbff1b_0
ptyprocess                0.7.0              pyhd3eb1b0_2
pure_eval                 0.2.2              pyhd3eb1b0_0
py-cpuinfo                9.0.0           py311haa95532_0
py-lief                   0.12.3          py311hd77b12b_0
pyarrow                   14.0.2          py311h847bd2a_0
pyarrow-hotfix            0.6                pyhd8ed1ab_0    conda-forge
pyasn1                    0.4.8              pyhd3eb1b0_0
pyasn1-modules            0.2.8                      py_0
pybind11-abi              4                    hd3eb1b0_1
pycodestyle               2.10.0          py311haa95532_0
pycosat                   0.6.6           py311h2bbff1b_0
pycparser                 2.21               pyhd3eb1b0_0
pyct                      0.5.0           py311haa95532_0
pycurl                    7.45.2          py311he2ea4bf_1
pydantic                  1.10.12         py311h2bbff1b_1
pydeck                    0.8.0           py311haa95532_2
pydispatcher              2.0.5           py311haa95532_2
pydocstyle                6.3.0           py311haa95532_0
pyerfa                    2.0.0           py311h2bbff1b_0
pyflakes                  3.0.1           py311haa95532_0
pygments                  2.15.1          py311haa95532_1
pyjwt                     2.4.0           py311haa95532_0
pylint                    2.16.2          py311haa95532_0
pylint-venv               2.3.0           py311haa95532_0
pyls-spyder               0.4.0              pyhd3eb1b0_0
pynacl                    1.5.0           py311h8cc25b3_0
pyodbc                    5.0.1           py311hd77b12b_0
pyopenssl                 24.0.0          py311haa95532_0
pyparsing                 3.0.9           py311haa95532_0
pyqt                      5.15.10         py311hd77b12b_0
pyqt5-sip                 12.13.0         py311h2bbff1b_0
pyqtwebengine             5.15.10         py311hd77b12b_0
pysocks                   1.7.1           py311haa95532_0
pytables                  3.9.2           py311h91a9f6a_0
pytest                    7.4.0           py311haa95532_0
python                    3.11.7               he1021f5_0
python-dateutil           2.8.2              pyhd3eb1b0_0
python-dotenv             0.21.0          py311haa95532_0
python-fastjsonschema     2.16.2          py311haa95532_0
python-graphviz           0.20.1             pyh22cad53_0    conda-forge
python-json-logger        2.0.7           py311haa95532_0
python-libarchive-c       2.9                pyhd3eb1b0_1
python-lmdb               1.4.1           py311hd77b12b_0
python-lsp-black          1.2.1           py311haa95532_0
python-lsp-jsonrpc        1.0.0              pyhd3eb1b0_0
python-lsp-server         1.7.2           py311haa95532_0
python-slugify            5.0.2              pyhd3eb1b0_0
python-snappy             0.6.1           py311hd77b12b_0
python-tzdata             2023.3             pyhd3eb1b0_0
python-xxhash             3.4.1           py311ha68e1ae_0    conda-forge
python_abi                3.11                    2_cp311    conda-forge
pytoolconfig              1.2.6           py311haa95532_0
pytorch-cuda              12.1                 hde6ce7c_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3.post1    py311haa95532_0
pyviz_comms               3.0.0           py311haa95532_0
pywavelets                1.5.0           py311hd7041d2_0
pywin32                   305             py311h2bbff1b_0
pywin32-ctypes            0.2.0           py311haa95532_1000
pywinpty                  2.0.10          py311h5da7b33_0
pyyaml                    6.0.1           py311h2bbff1b_0
pyzmq                     25.1.2          py311hd77b12b_0
qdarkstyle                3.0.2              pyhd3eb1b0_0
qstylizer                 0.2.2           py311haa95532_0
qt-main                   5.15.2              h19c9488_10
qt-webengine              5.15.9               h5bd16bc_7
qtawesome                 1.2.2           py311haa95532_0
qtconsole                 5.4.2           py311haa95532_0
qtpy                      2.4.1           py311haa95532_0
queuelib                  1.6.2           py311haa95532_0
re2                       2022.04.01           hd77b12b_0
referencing               0.30.2          py311haa95532_0
regex                     2023.10.3       py311h2bbff1b_0
reproc                    14.2.4               hd77b12b_1
reproc-cpp                14.2.4               hd77b12b_1
requests                  2.31.0          py311haa95532_1
requests-file             1.5.1              pyhd3eb1b0_0
requests-toolbelt         1.0.0           py311haa95532_0
responses                 0.18.0                   pypi_0    pypi
rfc3339-validator         0.1.4           py311haa95532_0
rfc3986                   1.5.0                    pypi_0    pypi
rfc3986-validator         0.1.1           py311haa95532_0
rich                      13.3.5          py311haa95532_0
rope                      1.7.0           py311haa95532_0
rpds-py                   0.10.6          py311h062c2fa_0
rtree                     1.0.1           py311h2eaa2aa_0
ruamel.yaml               0.17.21         py311h2bbff1b_0
ruamel_yaml               0.17.21         py311h2bbff1b_0
s3fs                      2023.10.0       py311haa95532_0
safetensors               0.4.2           py311hc37eb10_0    conda-forge
scikit-image              0.22.0          py311hb4ba03d_0
scikit-learn              1.2.2           py311hd77b12b_1
scipy                     1.11.4          py311hc1ccb85_0
scrapy                    2.8.0           py311haa95532_0
seaborn                   0.13.2                   pypi_0    pypi
semver                    2.13.0             pyhd3eb1b0_0
send2trash                1.8.2           py311haa95532_0
sentencepiece             0.2.0                    pypi_0    pypi
service_identity          18.1.0             pyhd3eb1b0_1
setuptools                68.2.2          py311haa95532_0
shellingham               1.5.4                    pypi_0    pypi
sip                       6.7.12          py311hd77b12b_0
six                       1.16.0             pyhd3eb1b0_1
smart_open                5.2.1           py311haa95532_0
smmap                     4.0.0              pyhd3eb1b0_0
snappy                    1.1.10               h6c2663c_1
sniffio                   1.3.0           py311haa95532_0
snowballstemmer           2.2.0              pyhd3eb1b0_0
sortedcontainers          2.4.0              pyhd3eb1b0_0
soupsieve                 2.5             py311haa95532_0
sphinx                    5.0.2           py311haa95532_0
sphinxcontrib-applehelp   1.0.2              pyhd3eb1b0_0
sphinxcontrib-devhelp     1.0.2              pyhd3eb1b0_0
sphinxcontrib-htmlhelp    2.0.0              pyhd3eb1b0_0
sphinxcontrib-jsmath      1.0.1              pyhd3eb1b0_0
sphinxcontrib-qthelp      1.0.3              pyhd3eb1b0_0
sphinxcontrib-serializinghtml 1.1.5              pyhd3eb1b0_0
spyder                    5.4.3           py311haa95532_1
spyder-kernels            2.4.4           py311haa95532_0
sqlalchemy                2.0.25          py311h2bbff1b_0
sqlite                    3.41.2               h2bbff1b_0
sse-starlette             1.8.2                    pypi_0    pypi
stack_data                0.2.0              pyhd3eb1b0_0
starlette                 0.37.2                   pypi_0    pypi
statsmodels               0.14.0          py311hd7041d2_0
streamlit                 1.33.0                   pypi_0    pypi
sympy                     1.12            py311haa95532_0
tabulate                  0.9.0           py311haa95532_0
tbb                       2021.8.0             h59b6b97_0
tblib                     1.7.0              pyhd3eb1b0_0
tenacity                  8.2.2           py311haa95532_0
terminado                 0.17.1          py311haa95532_0
text-unidecode            1.3                pyhd3eb1b0_0
textdistance              4.2.1              pyhd3eb1b0_0
threadpoolctl             2.2.0              pyh0d69192_0
three-merge               0.1.1              pyhd3eb1b0_0
tifffile                  2023.4.12       py311haa95532_0
tiktoken                  0.6.0                    pypi_0    pypi
tinycss2                  1.2.1           py311haa95532_0
tk                        8.6.12               h2bbff1b_0
tldextract                3.2.0              pyhd3eb1b0_0
tokenizers                0.15.2          py311h91c4a10_0    conda-forge
toml                      0.10.2             pyhd3eb1b0_0
tomlkit                   0.12.4                   pypi_0    pypi
toolz                     0.12.0          py311haa95532_0
torch                     2.2.2                    pypi_0    pypi
torchaudio                2.2.1                    pypi_0    pypi
torchvision               0.17.1                   pypi_0    pypi
tornado                   6.3.3           py311h2bbff1b_0
tqdm                      4.65.0          py311h746a85d_0
traitlets                 5.7.1           py311haa95532_0
transformers              4.39.1                   pypi_0    pypi
translate                 3.6.1                    pypi_0    pypi
truststore                0.8.0           py311haa95532_0
twisted                   23.10.0         py311haa95532_0
twisted-iocpsupport       1.0.2           py311h2bbff1b_0
typer                     0.9.4                    pypi_0    pypi
types-requests            2.31.0.20240406          pypi_0    pypi
typing-extensions         4.9.0           py311haa95532_1
typing-inspect            0.9.0                    pypi_0    pypi
typing_extensions         4.9.0           py311haa95532_1
tzdata                    2023d                h04d1e81_0
tzlocal                   2.1             py311haa95532_1
uc-micro-py               1.0.1           py311haa95532_0
ucrt                      10.0.22621.0         h57928b3_0    conda-forge
ujson                     5.4.0           py311hd77b12b_0
unidecode                 1.2.0              pyhd3eb1b0_0
urllib3                   2.0.7           py311haa95532_0
utf8proc                  2.6.1                h2bbff1b_1
uvicorn                   0.23.2                   pypi_0    pypi
validators                0.18.2             pyhd3eb1b0_0
vc                        14.2                 h21ff451_1
vc14_runtime              14.38.33130         h82b7239_18    conda-forge
vs2015_runtime            14.38.33130         hcb4865c_18    conda-forge
w3lib                     2.1.2           py311haa95532_0
watchdog                  2.1.6           py311haa95532_0
wcwidth                   0.2.5              pyhd3eb1b0_0
webencodings              0.5.1           py311haa95532_1
websocket-client          0.58.0          py311haa95532_4
werkzeug                  2.2.3           py311haa95532_0
whatthepatch              1.0.2           py311haa95532_0
wheel                     0.41.2          py311haa95532_0
widgetsnbextension        3.5.2           py311haa95532_1
win_inet_pton             1.1.0           py311haa95532_0
winpty                    0.4.3                         4
wrapt                     1.14.1          py311h2bbff1b_0
xarray                    2023.6.0        py311haa95532_0
xformers                  0.0.25.post1             pypi_0    pypi
xlwings                   0.29.1          py311haa95532_0
xxhash                    0.8.2                hcfcfb64_0    conda-forge
xyzservices               2022.9.0        py311haa95532_1
xz                        5.4.5                h8cc25b3_0
yaml                      0.2.5                he774522_0
yaml-cpp                  0.8.0                hd77b12b_0
yapf                      0.31.0             pyhd3eb1b0_0
yarl                      1.9.3           py311h2bbff1b_0
zeromq                    4.3.5                hd77b12b_0
zfp                       1.0.0                hd77b12b_0
zict                      3.0.0           py311haa95532_0
zipp                      3.17.0          py311haa95532_0
zlib                      1.2.13               h8cc25b3_0
zlib-ng                   2.0.7                h2bbff1b_0
zope                      1.0             py311haa95532_1
zope.interface            5.4.0           py311h2bbff1b_0
zstandard                 0.19.0          py311h2bbff1b_0
zstd                      1.5.5                hd43e919_0
```