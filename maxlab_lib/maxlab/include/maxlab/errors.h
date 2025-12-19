/**
 * @file errors.h
 */

#pragma once

namespace maxlab
{
/**
 * @brief Enum indicating all possible error codes.
 */
enum Status : int
{
    MAXLAB_OK,                      ///< All good
    MAXLAB_API_FAIL,                ///< API response was unexpected
    MAXLAB_NO_SERVER_CONNECTION,    ///< No connection to ``mxwserver``
    MAXLAB_LICENSE_INVALID,         ///< No valid license
    MAXLAB_API_NO_RESPONSE,         ///< No response from API
    MAXLAB_STREAM_ALREADY_OPENED,   ///< Trying to open an already open stream for the second time
    MAXLAB_INCOMPATIBLE_FILTERING,  ///< Not all filtering options are compatible with all streams
    MAXLAB_INVALID_INPUT,           ///< Arguments passed to function are invalid
    MAXLAB_STREAM_NOT_OPENED,       ///< Stream is not opened
    MAXLAB_NO_FRAME,                ///< No data in the stream
};

/**
 * @brief Function to translate error code into human readable text.
 * @param status Error code to be translated.
 * @return Null-terminated character array.
 */
const char *statusToText(Status status);


/**
 * @brief Function to verify that \p status is \ref maxlab::Status::MAXLAB_OK.
 * @param status
 *
 * This function checks if \p status is ok and otherwise prints the error message to ``stdcerr`` and exits the
 * execution with error code ``1``.
 */
void verifyStatus(Status status);
} // namespace maxlab