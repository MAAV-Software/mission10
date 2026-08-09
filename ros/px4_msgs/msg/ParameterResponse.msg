# Result and read-back value for one ParameterRequest.

uint64 timestamp
uint32 request_id

uint8 RESULT_SUCCESS = 0
uint8 RESULT_NOT_FOUND = 1
uint8 RESULT_TYPE_MISMATCH = 2
uint8 RESULT_VEHICLE_ARMED = 3
uint8 RESULT_SET_FAILED = 4
uint8 RESULT_INVALID_OPERATION = 5
uint8 RESULT_INVALID_NAME = 6
uint8 result

uint8 VALUE_TYPE_UNKNOWN = 0
uint8 VALUE_TYPE_INT32 = 1
uint8 VALUE_TYPE_FLOAT = 2
uint8 parameter_type

char[17] name # Canonical PX4 parameter name, null terminated
int32 int_value
float32 float_value

uint8 ORB_QUEUE_LENGTH = 8
