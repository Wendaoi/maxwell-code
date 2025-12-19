#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "versions.h"

namespace maxlab
{
/**
* @brief If version of headers does not match static library, this function will exit the code.
* If you encounter this error, it means you have mixed up your header and *.a files.
*/
inline void checkVersions()
{
    if (strcmp(MAXLAB_PUBLIC_API_VERSION, getStaticLibraryVersion()) != 0)
    {
        fprintf(stderr, "The version in the headers of the library does not match the version of the compiled code.\n"
                        "-> Header version:\t\t%s\n"
                        "-> Static-library version:\t%s\n", MAXLAB_PUBLIC_API_VERSION, getStaticLibraryVersion());
        exit(1);
    }
}
}