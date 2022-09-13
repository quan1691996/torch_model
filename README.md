## Build Deep learning model from scratch

## Set up Remote Graphical Display for wsl
XServer Windows - WSL1 & WSL2:
Install X-Server Windows
https://sourceforge.net/projects/vcxsrv/

Set Display forward in WSL Distro\
Configure Display:

If you running WSL1:
```shell script
export LIBGL_ALWAYS_INDIRECT=1
export DISPLAY=localhost:0
```
If you running WSL2:
```shell script
export LIBGL_ALWAYS_INDIRECT=1
export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0
```
(If you have disabled resolv.conf use this definition: https://stackoverflow.com/a/63092879/11473934)
and then (install x11-apps):

```shell script
sudo apt update
sudo apt install x11-apps
```
### Start XLaunch on Windows
* Multiple Windows
* Start no client
* disable Native opengl
* enable Disable access control

Test it\
In wsl: enter xcalc - Calculator should open in Windows10\

If everything worked\
And you want to persist the settings in your wsl distro. Store them in your ~/.bashrc.\

```shell script
sudo nano ~/.bashrc
```

Copy the two lines (from Set Display forward in WSL Distro - Configure Display), two the end and save it.\

### Add it to autostart
* Run Dialog see Start XLaunch on Windows
* Save configuration
* Press Windows + R
* Enter: shell:startup
* Copy saved configuration: *.launch (Generated in step 2) to this folder (step 4)

Now the XServer will be started with windows startup.\

follow this https://stackoverflow.com/a/63092879