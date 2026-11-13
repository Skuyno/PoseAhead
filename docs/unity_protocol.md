# PoseAhead UDP Protocol v2

One pose is one IPv4 UDP datagram in the current implementation. All integers and IEEE-754 floats are
little-endian. The hard packet limit is 1200 bytes; v2 packets are 331-635 bytes
depending on optional transforms. Receivers must never concatenate datagrams.

## Header

| Offset | Type | Field |
|---:|---|---|
| 0 | 4 bytes | ASCII magic `PAUD` |
| 4 | uint8 | protocol version, `2` |
| 5 | uint8 | flags |
| 6 | uint8 | joint count, `17` |
| 7 | uint8 | algorithm ID |
| 8 | uint64 | ordered sender session ID |
| 16 | uint64 | frame ID |
| 24 | uint64 | source timestamp ns |
| 32 | uint64 | produced timestamp ns |
| 40 | uint64 | send timestamp ns |
| 48 | int16 | target prediction offset ms |
| 50 | uint16 | reserved, must be zero |

Flags: bit 0 root position, bit 1 root rotation, bit 2 joint rotations, bit 3
predictability, bit 4 estimated monocular root. Bit 4 requires bit 0 and adds no
payload bytes; C# exposes it as `RootPositionEstimated`. Older receivers reject
the new flag and must be updated. No flag is a ground-truth certificate.
Unknown flag bits make the packet incompatible and must be
rejected.

## Payload order

Fields are tightly packed without alignment padding:

1. optional predictability `float32`, constrained to `[0, 1]`;
2. optional root position, 3 x `float32` XYZ;
3. optional root rotation, 4 x `float32` XYZW;
4. required joint positions, 17 x 3 x `float32` in canonical joint order;
5. optional local joint rotations, 17 x 4 x `float32` XYZW;
6. required joint confidence, 17 x `float32`;
7. required joint validity, 3 bytes, least-significant bit first;
8. CRC-32 of every preceding byte, `uint32`.

Default MeTRAbs packets contain its monocular estimated root and set the
estimated-root flag (343 bytes); see `metrabs.md` for assumptions and Unity
container placement. Position-only packets with trusted root translation are 343 bytes; every optional
field together uses 635 bytes. The seven unused high bits of the final validity
byte must be zero. Optional rotations must be unit quaternions with absolute
norm tolerance `1e-4`; malformed rotations are rejected, not normalized silently.

## Algorithm IDs

| ID | Algorithm |
|---:|---|
| 0 | raw observation |
| 1 | hold |
| 2 | linear extrapolation |
| 3 | double exponential smoothing |
| 4 | Kalman |
| 5 | EMA |
| 6 | One-Euro |
| 7 | Kalman constant-velocity extrapolation |
| 10-13 | learned predictors for 33/66/100/150 ms |
| 20 | adaptive selector |

IDs 1-3 retain their original meanings for compatibility; they are not aliases
for EMA or One-Euro. IDs 5-7 identify the implemented Python baseline processors.
Older v2 receivers reject these new IDs and
must be updated before those methods are sent. The packet layout is unchanged.

The experimental adaptive processor uses ID 20 only for positive compensation,
and ID 0 with offset zero for raw fallback. Its offset is the selected nominal
33/66/100/150-ms forecast horizon. A position blend with the current observation
is partial compensation, not a measured reduction in latency or an estimate at
alpha times that horizon. Alpha and causal decisions are preserved in Python
processed-recording metadata, not in the datagram. The threshold margin is not a
calibrated probability, so this processor leaves optional predictability absent.
Optional smooth transitions can mix several current-anchor forecasts; then the
nominal offset is the largest horizon with a positive weight. It is not necessarily
the horizon currently requested by the policy. UDP cannot reconstruct this mixture;
full requested decisions and forecast weights remain in the processed NPZ logs.
See [adaptive compensation](adaptive_compensation.md).

With explicit `fallback_method: kalman_cv`, adaptive alpha means neural weight
against Kalman-CV at the same future target, not against current Raw. Kalman-only
outputs use ID 7 with their actual configured horizon; positive neural mixtures
use ID 20 at that same horizon. Tracking-loss Raw uses ID 0/offset zero. The root
is the Kalman baseline's translation. Weight/target details remain in Python logs,
not additional wire fields; zero neural weight is not zero prediction horizon.

## Timing and receiver policy

The Unity receiver binds only to IPv4 loopback (`127.0.0.1:5005` by default),
not LAN/VPN interfaces, and rejects non-loopback senders. Remote streaming is
not enabled by the current receiver. This is not authentication: local processes
are trusted and CRC only detects corruption. `IsReceiving` and `LastError` expose
receiver health; unexpected socket failures are not silently hidden.

Sender timestamps share the sender's monotonic clock. They cannot be subtracted
from Unity timestamps until clock offset is calibrated. Unity keeps only the
newest accepted `(session_id, frame_id)` and rejects malformed packets, CRC failures, unsupported
versions, duplicates, and older frames **before** replacing the capacity-one slot. UDP loss is expected and
must never cause an unbounded retry or queue.

`session_id` is the sender's positive `perf_counter_ns()` value at sender
construction. Python process restarts on the same Windows host/boot retain that
clock domain, so a later session accepts reset frame IDs immediately. Packets
from older sessions stay rejected. This protocol assumes one sender host per
receiver. After a sender OS reboot or switching hosts, explicitly restart the
receiver to reset its clock epoch; session clocks from different hosts must not
be compared. No receiver-to-sender clock subtraction is used for ordering.

A pose expires 0.5 seconds after local receipt by default. Expiry clears the
displayable pose but retains the ordering high-water mark, so delayed packets do
not revive old state. Receiver enable/disable resets the displayable pose; enable
starts a fresh buffer. The local timeout cannot measure time spent in the network
before receipt without clock synchronization.

V2 adds the session field; v1 is explicitly rejected, not silently reinterpreted.
Update Python and C# together. The retained v1 fixture is used only to test
rejection. Canonical `PAFR` storage serialization remains at version 1.

The project is pinned to Unity 6.3 LTS `6000.3.23f1`, changeset
`09d2ecc7fb28`. Decoder and ordered buffer are free of `UnityEngine` dependencies.
The full-field golden fixture is also generated identically by Python and decoded
by C#. All 16 optional-field combinations, rotation validation, new baseline IDs,
nonzero padding, and truncated packets with valid CRCs have regression coverage.
The .NET test compiles all transport sources, including the actual MonoBehaviour
receiver with minimal Unity type stubs. It tests golden bytes, corruption,
ordering, expiry, session switching, and loopback reception. Unity Editor scene
and lifecycle integration still require a separate editor test. Position-based
Humanoid mapping, a diagnostic scene generator and paced Python replay are now
implemented; see [retargeting and remaining native checks](unity_retargeting.md).
