
## How to install

### Install MuJoCo 2.0

```bash
sudo apt update
sudo apt install unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
wget https://www.roboti.us/download/mujoco200_linux.zip -P /tmp
unzip /tmp/mujoco200_linux.zip -d ~/.mujoco
wget https://www.roboti.us/file/mjkey.txt -P /tmp
mv /tmp/mjkey.txt ~/.mujoco/
```

### Install dependencies

```bash
conda env create -f conda_env.yml
conda activate mrn
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd ../custom_dmc2gym
pip install -e .
cd ..
pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
pip install pybullet termcolor
```


## How to run

```bash
.\traincrun.sh
.\traindooropen.sh
.\trainwalkerstand.sh
.\trainwalkerwalker.sh
```