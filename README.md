## Prerequisites
Check installation of `xhost` on your platform, it is needed for X11 port forwarding (displaying GUI).
If it is not present on your system, install it with one of the following commands:
#### Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install x11-xserver-utils
```
#### CentOS/RHEL:
```
sudo yum install xorg-x11-server-utils
```
#### Fedora:
```
sudo dnf install xorg-x11-server-utils
```
#### Alpine Linux:
```
apk add xhost
```
#### Arch Linux:
```
sudo pacman -S xorg-xhost
```

### Running `.ipynb` files
If you wish to run `.ipynb` files, you have to install `jupyter` or `jupyterlab` package via pip or other python package manager:
```
pip install jupyterlab
```

## Running the program
To build and start the program, simply run:
```
./run.sh
```
