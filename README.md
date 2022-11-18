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
In wsl: enter xcalc - Calculator should open in Windows10

If everything worked\
And you want to persist the settings in your wsl distro. Store them in your ~/.bashrc.

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

Now the XServer will be started with windows startup.

follow this https://stackoverflow.com/a/63092879

## Connect USB devices for wsl
Follow this link: https://docs.microsoft.com/en-us/windows/wsl/connect-usb

in case of this WARNING when run **usbipd wsl list**
```shell script
WARNING: usbipd not found for kernel 5.10.102.1-microsoft

You may need to install the following packages for this specific kernel:
linux-tools-5.10.102.1-microsoft-standard-WSL2
linux-cloud-tools-5.10.102.1-microsoft-standard-WSL2

You may also want to install one of the following packages to keep up to date:
linux-tools-standard-WSL2
linux-cloud-tools-standard-WSL2
```

Turn on Window Powershell as administrator

Run
``` shell script
wsl -l -v
```

Check if your targeted distro is used as default/
If not, run (<Distribution Name> is your targted distro)
``` shell script
wsl --set-default <Distribution Name>
```
## Setting Camera
* Attach camera to WSL
* check if /dev/video* is available
* If it is available, change folder permission by: 
``` shell script
sudo chmod 777 /dev/video*
```
Follow this link: https://zenn.dev/pinto0309/articles/e1432253d29e30

## Note
When restart your wsl, setting usb port again to run the program. And modify permission of /dev/video*.

``` shell script
sudo chmod 777 /dev/video*
```
