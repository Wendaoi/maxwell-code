/**
 * @file data_streamer.h
 */

/**
 * \addtogroup maxlab
 *  @{
 */

#pragma once

#include <stdint.h>

#include "errors.h"

/**
 * @brief General namespace of the MaxLab C++ API.
 */
namespace maxlab
{
struct SpikeEvent;

/**
 * @brief Enum that represents the possible types of filters that can be selected.
 */
enum class FilterType
{
    FIR = 0,    ///< Default filter in Scope. Latency is roughly ``2048 / sampling rate`` ms.
    IIR = 1,    ///< Latency is roughly ``10 / sampling rate`` ms.
};

/**
 * @brief Information about the state of a data frame.
 *
 * In particular, the frame-number, the well-id and if the frame is corrupted or not.
 */
struct FrameInfo
{
    uint64_t frame_number = 0; ///< Frame number
    uint8_t well_id = 0;       ///< Well Id on MultiWell devices
    bool corrupted = false;    ///< If frame is corrupted or not. If corrupted, no guarantees on data contained in it.
};

/**
 * @brief Structure to store the information obtained from the filtered data stream.
 *
 * The filtered data stream contains all detected spikes at each individual frame.
 *
 * Note that there will at most be a 1024 spikes (i.e. the number of readout channels), but there might be less.
 * You thus need to make sure to never access memory that is further offset from the ``spikeEvents`` pointer than
 * ``spikeCount``.
 *
 * Example
 * @code
 *  ...
 *  FilteredFrameData frame;
 *  DataStreamerFiltered_receiveNextFrame(&frame);
 *  frame.spikeEvents[frame.spikeCount - 1]    // Valid access
 *  frame.spikeEvents[frame.spikeCount]        // Invalid access and undefined behaviour
 * @endcode
 */
struct FilteredFrameData
{
    uint64_t spikeCount;                ///< How many spikes were detected, i.e. the length of the ``spikeEvents`` array
    const SpikeEvent * spikeEvents;     ///< Pointer to the spikes
    FrameInfo frameInfo;                ///< Information about current frame even if no spikes were detected
};

/**
 * @brief Structure to store the information obtained from the raw data stream.
 *
 * Note that there are always exactly 1024 amplitudes (i.e. the number of readout channels).
 *
 * Example
 * @code
 *  ...
 *  RawFrameData frame;
 *  DataStreamerRaw_receiveNextFrame(&frame);
 *  frame.amplitudes[frame.amplitudeCount - 1]    // Valid access
 *  frame.amplitudes[frame.amplitudeCount]        // Invalid access and undefined behaviour
 * @endcode
 */
struct RawFrameData
{
    FrameInfo frameInfo;                            ///< For more infos see \ref maxlab::FrameInfo.
    const static uint64_t amplitudeCount = 1024;    ///< Fixed size of the amplitudes pointer
    const float * amplitudes;                       ///< Pointer to the amplitudes array
};

/**
 * @brief Crates the connection to the filtered data stream.
 * @param filterType Specify the type of filtering you want. For more information, see \ref maxlab::FilterType.
 * @return Status of the initialization of the stream.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_LICENSE_INVALID``
 *  - ``maxlab::Status::MAXLAB_STREAM_ALREADY_OPENED``
 *  - ``maxlab::Status::MAXLAB_INCOMPATIBLE_FILTERING``
 *  - ``maxlab::Status::MAXLAB_NO_SERVER_CONNECTION``
 *
 * It is recommended to use \ref maxlab::verifyStatus to check the return of this function.
 */
Status DataStreamerFiltered_open(FilterType filterType);

/**
 * @brief Closes the connection to the filtered data stream and restores the state of the mxwserver to the default.
 * @return Status of the closure of the stream.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_NO_SERVER_CONNECTION``
 *  - ``maxlab::Status::MAXLAB_API_NO_RESPONSE``
 *
 * Note that you should ensure to always call this function at end of you experiment and check with \ref
 * maxlab::verifyStatus if the closing worked correctly. If it did not, make sure to restart ``mxwserver`` before
 * starting any new runs of doing any experiments in ``scope``.
 */
Status DataStreamerFiltered_close();

/**
 * @brief Receive next frame from filtered data stream if there is one.
 * @param frameData Buffer in which data can be written into.
 * @return Status of the receiving operation.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_INVALID_INPUT``
 *  - ``maxlab::Status::MAXLAB_STREAM_NOT_OPENED``
 *  - ``maxlab::Status::MAXLAB_NO_FRAME``
 *
 * This function is currently **blocking**, i.e. it waits to receive data for 1 millisecond, if it takes longer
 * we return \ref maxlab::Status::MAXLAB_NO_FRAME, otherwise \ref maxlab::Status::MAXLAB_OK.
 *
 * If you receive \ref maxlab::Status::MAXLAB_NO_FRAME it means that something is wrong (dropped frames,
 * corrupted frames, etc.).
 */
Status DataStreamerFiltered_receiveNextFrame(FilteredFrameData * frameData);

/**
 * @brief Allows the change of filter-type while keeping the data stream open.
 * @param filterType Filter type, for more information see \ref maxlab::FilterType.
 * @return Status of the operation.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_STREAM_NOT_OPENED``
 *  - ``maxlab::Status::MAXLAB_INCOMPATIBLE_FILTERING``
 */
Status DataStreamerFiltered_setFilterType(FilterType filterType);

/**
 * @brief Get currently set filter type.
 * @return Filter type, for more infos, see \ref maxlab::FilterType.
 */
FilterType DataStreamerFiltered_getFilterType();

/**
 * @brief Crates the connection to the raw data stream.
 * @return Status of the initialization of the stream.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_LICENSE_INVALID``
 *  - ``maxlab::Status::MAXLAB_STREAM_ALREADY_OPENED``
 *  - ``maxlab::Status::MAXLAB_INCOMPATIBLE_FILTERING``
 *  - ``maxlab::Status::MAXLAB_NO_SERVER_CONNECTION``
 *
 * It is recommended to use \ref maxlab::verifyStatus to check the return of this function.
 */
Status DataStreamerRaw_open();

/**
 * @brief Closes the connection to the raw data stream and restores the state of the mxwserver to the default.
 * @return Status of the closure of the stream.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_NO_SERVER_CONNECTION``
 *  - ``maxlab::Status::MAXLAB_API_NO_RESPONSE``
 *
 * Note that you should ensure to always call this function at end of you experiment and check with \ref
 * maxlab::verifyStatus if the closing worked correctly. If it did not, make sure to restart ``mxwserver`` before
 * starting any new runs of doing any experiments in ``scope``.
 */
Status DataStreamerRaw_close();

/**
 * @brief Receive next frame from filtered data stream if there is one.
 * @param frameData Buffer in which data can be written into.
 * @return Status of the receiving operation.
 * Possible values are:
 *  - ``maxlab::Status::MAXLAB_OK``
 *  - ``maxlab::Status::MAXLAB_INVALID_INPUT``
 *  - ``maxlab::Status::MAXLAB_STREAM_NOT_OPENED``
 *  - ``maxlab::Status::MAXLAB_NO_FRAME``
 *
 * This function is currently **blocking**, i.e. it waits to receive data for 1 millisecond, if it takes longer
 * we return \ref maxlab::Status::MAXLAB_NO_FRAME, otherwise \ref maxlab::Status::MAXLAB_OK.
 *
 * If you receive \ref maxlab::Status::MAXLAB_NO_FRAME it means that something is wrong (dropped frames,
 * corrupted frames, etc.).
 */
Status DataStreamerRaw_receiveNextFrame(RawFrameData * frameData);

} // namespace maxlab

/*! @} */
