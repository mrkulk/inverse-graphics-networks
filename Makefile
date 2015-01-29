CC = /usr/bin/gcc-4.4 #nvcc only supports 4.4
CFLAGS = -Wall -shared -fPIC -I/usr/include -lluaT 


cuda: gradACR.h gradACR.cu
	  nvcc -m64 -shared -arch=sm_20 -o libgradACR.so -ccbin /usr/bin/gcc-4.4 -Xcompiler -fPIC gradACR.cu

lua: 
	/usr/bin/gcc-4.4 -Wall -shared -fPIC -o gradACRWrapper.so -I/usr/local/cuda-6.5/include -I/usr/local/include/TH -I/usr/local/include/THC -I/usr/include -lTHC -lluaT -L. -lgradACR gradACRWrapper.c

clean: 
	$(RM) *.so
