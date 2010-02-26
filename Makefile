DEPTH = ../../../../..

include $(DEPTH)/make/openclsdkdefs.mk 

####
#
#  Targets
#
####

OPENCL			= 1
SAMPLE_EXE		= 1
INSTALL_TO_PUBLIC       = 1
EXE_TARGET 		= DiagSpMV
EXE_TARGET_INSTALL   	= DiagSpMV

####
#
#  C/CPP files
#
####

FILES 	= DiagSpMV Stencil
CLFILES	= DiagSpMV.cl

LLIBS  	+= SDKUtil
INCLUDEDIRS += $(SDK_HEADERS) 


include $(DEPTH)/make/openclsdkrules.mk 

