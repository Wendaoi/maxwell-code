/**
 * @file api_comm.h
 */

#pragma once

/**
 * @brief MaxLab C++ API namespace.
 * @author Ivan Ilak
 * @date 08.09.2023
 */

namespace maxlab
{
enum Status : int;  // see maxlab/errors.h

/**
 * @brief Structure to hold information about the response of the API call
 *
 * The necessary memory for the array was allocated on the heap and memory management is done by the library.
 * Every \ref maxlab::Response instance needs to free-d by a call to \ref maxlab::freeResponse!
 */
struct Response
{
    char * content = nullptr;   ///< The content of the message.
    Status status;              ///< Status of the API call
};


/**
 * @brief Function to trigger sending of sequence from ``mxwserver`` to the device.
 * @param sequenceName The name of the predefined ``maxlab.sequence.Sequence`` with the Python API.
 * @return Information if call was successful and if not what kind of error happened. Please consult the definition
 * of \ref maxlab::Status for more infos.
 */
Status sendSequence(const char *sequenceName);

/**
 * @brief Function to send raw API call to the ``mxwserver``.
 * @param message The raw API string as ``const char*``.
 * @return Instance of the \ref maxlab::Response struct. Please consult the definition of \ref maxlab::Response for
 * more infos.
 */
Response sendRaw(const char * message);

/**
 * @brief Function to deallocate memory of the response-object.
 * @param response The to be deleted object.
 * @return Status if the deallocation worked correctly.
 */
Status freeResponse(Response * response);

} // namespace maxlab