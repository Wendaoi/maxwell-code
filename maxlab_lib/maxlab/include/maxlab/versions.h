/**
 * @file versions.h
 */

#pragma once

#define MAXLAB_PUBLIC_API_VERSION "1.1.0"
#define MAXLAB_GIT_COMMIT_HASH "79e629e0"

namespace maxlab
{
/**
 * @brief Get the version of the library that was compiled into the library itself.
 * @returns Pointer to an array of characters. Format is "X.Y.Z".
 */
const char *getStaticLibraryVersion();
}
