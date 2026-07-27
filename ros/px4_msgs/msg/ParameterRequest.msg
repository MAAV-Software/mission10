# Read or write one PX4 parameter through the uXRCE-DDS bridge.

uint64 timestamp
uint32 request_id # Opaque caller-selected value echoed by ParameterResponse

uint8 OPERATION_READ = 0
uint8 OPERATION_WRITE = 1
uint8 operation

uint8 VALUE_TYPE_UNKNOWN = 0
uint8 VALUE_TYPE_INT32 = 1
uint8 VALUE_TYPE_FLOAT = 2
uint8 parameter_type # Required for writes; ignored for reads

char[17] name # Canonical PX4 parameter name, null terminated
int32 int_value
float32 float_value

uint8 ORB_QUEUE_LENGTH = 8
