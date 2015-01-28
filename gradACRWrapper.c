
/* Usage:

nvcc -m64 -shared -arch=sm_20 -o libgradACR.so  -Xcompiler -fPIC gradACR.cu
 
/usr/bin/gcc-4.4 -Wall -shared -fPIC -o gradACRWrapper.so -I/usr/include -lluaT -L. -lgradACR gradACRWrapper.c


*/

//#include <lua.h>                               /* Always include this */
//#include <lauxlib.h>                           /* Always include this */
//#include <lualib.h>                            /* Always include this */
#include <stdio.h>
#include <luaT.h>
#include <THC/THC.h>
#include <TH/TH.h>
#include <TH/THTensor.h>
#include "gradACR.h"
#include <math.h>


#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

// static int gradACRWrapper(lua_State *L){                
// 	float rtrn = lua_tonumber(L, -1);     
// 	printf("Top of cube(), number=%f\n",rtrn);
	
// 	get_gradACR_gradient();

// 	lua_pushnumber(L,rtrn*rtrn*rtrn);      

// 	return 1;                              
// }

static void stackDump (lua_State *L) {
  printf("-----------------\n");
  int i;
  int top = lua_gettop(L);
  for (i = 1; i <= top; i++) {  /* repeat for each level */
    int t = lua_type(L, i);
    switch (t) {

      case LUA_TSTRING:  /* strings */
        printf("`%s'", lua_tostring(L, i));
        break;

      case LUA_TBOOLEAN:  /* booleans */
        printf(lua_toboolean(L, i) ? "true" : "false");
        break;

      case LUA_TNUMBER:  /* numbers */
        printf("%g", lua_tonumber(L, i));
        break;

      default:  /* other values */
        printf("%s", lua_typename(L, t));
        break;

    }
    printf("  ");  /* put a separator */
  }
  printf("\n");  /* end the listing */
  printf("-----------------\n");
}

static int gradACRWrapper(lua_State *L){                
	//stackDump(L);

	static const void* torch_DoubleTensor_id = NULL;
  	torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");
	
	int narg = lua_gettop(L);
	THDoubleTensor *arg1 = NULL;
	int arg2 = 0;
	if(narg == 1 && (arg1 = luaT_toudata(L, 1, torch_DoubleTensor_id)))
	{
	}
	else
		luaL_error(L, "expected arguments: DoubleTensor");
	
	printf("sz:%f\n", arg1->storage->data[1]);

	//arg2 = 9;
	//lua_pushnumber(L, (lua_Number)arg2);
	
	//lua_push
	stackDump(L);
	return 1;  

}

int luaopen_gradACRWrapper(lua_State *L){
	lua_register(L,"gradACRWrapper",gradACRWrapper);
	return 0;
}
