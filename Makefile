CC = /usr/bin/gcc-4.4 #nvcc only supports 4.4
CFLAGS = -Wall -shared -fPIC -I/usr/include -lluaT 


cuda: gradACR.h gradACR.cu
	  nvcc -m64 -shared -arch=sm_20 -o libgradACR.so  -Xcompiler -fPIC gradACR.cu

lua: 
	/usr/bin/gcc-4.4 -Wall -shared -fPIC -o gradACRWrapper.so -I/usr/include -lluaT -L. -lgradACR gradACRWrapper.c

clean: 
	$(RM) *.so