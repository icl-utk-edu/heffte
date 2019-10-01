function(BASIC_FIND PCKG REQ_INCS REQ_LIBS)

#If environment variable ${PCKG}_DIR is specified,
# it has same effect as local ${PCKG}_DIR
if( (NOT ${PCKG}_DIR) AND DEFINED ENV{${PCKG}_DIR} )
  set( ${PCKG}_DIR "$ENV{${PCKG}_DIR}" )
  message(STATUS " ${PCKG}_DIR is set from environment: ${${PCKG}_DIR}")
endif()

if (NOT ${PCKG}_DIR)
  if(!${PCKG}_FIND_QUIETLY)
    message (WARNING " Option ${PCKG}_DIR not set.")
  endif ()
else (NOT ${PCKG}_DIR)
  message (STATUS " ${PCKG}_DIR set to ${${PCKG}_DIR}")
endif (NOT ${PCKG}_DIR)

message (STATUS " Searching for package ${PCKG}...")
set (${PCKG}_FOUND FALSE PARENT_SCOPE)

# Look for each required include file
foreach(INC_FILE ${REQ_INCS})
  message (STATUS " Searching for include file: ${INC_FILE}")
  set (INC_DIR ${INC_FILE}-NOTFOUND)
  find_path(INC_DIR ${INC_FILE}
    HINTS ${${PCKG}_DIR} ${${PCKG}_DIR}/include)
  if (EXISTS ${INC_DIR}/${INC_FILE})
    message (STATUS " Found include file ${INC_FILE} in ${INC_DIR} required by ${PCKG}")
    set (${PCKG}_INCLUDE_DIRS ${${PCKG}_INCLUDE_DIRS} ${INC_DIR} PARENT_SCOPE)
  elseif(!${PCKG}_FIND_QUIETLY)
    message (WARNING " Failed to find include file ${INC_FILE} required by ${PCKG}")
  endif ()
endforeach()
# Look for each required library
foreach(LIB_NAME ${REQ_LIBS})
  message (STATUS " Searching for library: ${LIB_NAME}")
  set (LIB ${LIB_NAME}-NOTFOUND)
  find_library(LIB NAMES "lib${LIB_NAME}.a" "${LIB_NAME}"
    HINTS ${${PCKG}_DIR} ${${PCKG}_DIR}/lib)
  if (EXISTS ${LIB})
    message (STATUS " Found library at ${LIB} required by ${PCKG}")
    set (${PCKG}_LIBRARIES ${${PCKG}_LIBRARIES} ${LIB} PARENT_SCOPE)
    set (${PCKG}_FOUND TRUE PARENT_SCOPE)
    set (${PCKG}_FOUND TRUE )
  else ()
    set (${PCKG}_FOUND FALSE PARENT_SCOPE)
    set (${PCKG}_FOUND FALSE )
  endif ()
endforeach()

# If we made it this far, then we call the package "FOUND"
if(${PCKG}_FOUND)
  message (STATUS "All required include files and libraries found.")
else()
  if(!${PCKG}_FIND_QUIETLY)
    message ("WARNING! ${PCKG} not found.")
  else()
    message ("${PCKG} not found (Optional, Not Required).")
  endif()
endif()

endfunction(BASIC_FIND)
