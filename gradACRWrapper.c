
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

	int imwidth = lua_tonumber(L, 1);
	int tdim = lua_tonumber(L, 2);
	int bsize = lua_tonumber(L, 3);

	static const void* torch_DoubleTensor_id = NULL;
  	torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

	//THDoubleTensor *arg1 = NULL;
	//arg1 = luaT_toudata(L, 1, torch_DoubleTensor_id);
	//printf("sz:%f\n", arg1->storage->data[1]);
	//arg1->storage->data[1] = 6699;
	//arg2 = 9;
	//lua_pushnumber(L, (lua_Number)arg2);

  	//stackDump(L);

  	THDoubleTensor *toutput = luaT_toudata(L, 4, torch_DoubleTensor_id);
  	double *output = toutput->storage->data;

	THDoubleTensor *tpose = luaT_toudata(L, 5, torch_DoubleTensor_id);
  	double *pose = tpose->storage->data;

	THDoubleTensor *ttemplate = luaT_toudata(L, 6, torch_DoubleTensor_id);
  	double *_template = ttemplate->storage->data;

	THDoubleTensor *tgradOutput = luaT_toudata(L, 7, torch_DoubleTensor_id);
  	double *gradOutput = tgradOutput->storage->data;

	THDoubleTensor *tgradAll = luaT_toudata(L, 8, torch_DoubleTensor_id);
  	double *gradAll = tgradAll->storage->data;

  	double *gradTemplate = gradAll;
  	double *gradPose = gradAll + bsize*tdim*tdim; 

	int output_size = sizeof(double) * bsize * imwidth * imwidth ;
	int pose_size =  sizeof(double) * bsize * 3 * 3;
	int template_size = sizeof(double) * bsize * tdim * tdim ;
	int gradOutput_size = sizeof(double) * bsize * imwidth * imwidth;
	int gradTemplate_size = sizeof(double) * bsize * tdim * tdim;
	int gradPose_size = sizeof(double) * bsize * 3 * 3;

	get_gradACR_gradient( imwidth, tdim, bsize, output, pose, 
					 _template,  gradOutput,  gradTemplate,  gradPose);	
	return 1;  

}

int luaopen_gradACRWrapper(lua_State *L){
	lua_register(L,"gradACRWrapper",gradACRWrapper);
	return 0;
}
