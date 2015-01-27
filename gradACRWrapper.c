
/* Usage:

nvcc -m64 -shared -arch=sm_20 -o libgradACR.so  -Xcompiler -fPIC gradACR.cu
 
/usr/bin/gcc-4.4 -Wall -shared -fPIC -o gradACRWrapper.so -I/usr/include -lluaT -L. -lgradACR gradACRWrapper.c


*/

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */
#include <stdio.h>


#include "gradACR.h"
 

static int gradACRWrapper(lua_State *L){                
	float rtrn = lua_tonumber(L, -1);     
	printf("Top of cube(), number=%f\n",rtrn);
	
	get_gradACR_gradient();

	lua_pushnumber(L,rtrn*rtrn*rtrn);      
	return 1;                              
}

int luaopen_gradACRWrapper(lua_State *L){
	lua_register(L,"gradACRWrapper",gradACRWrapper);
	return 0;
}
