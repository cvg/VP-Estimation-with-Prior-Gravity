# Macro to add source files to COLMAP library.
macro(UNCALIBRATED_VP_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(UNCALIBRATED_VP_SOURCES ${UNCALIBRATED_VP_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(UNCALIBRATED_VP_ADD_SOURCES)

