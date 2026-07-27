#!/usr/bin/env python3
"""Read or write one PX4 parameter over the uXRCE-DDS link."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from px4_msgs.msg import ParameterRequest, ParameterResponse
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


RESULT_NAMES = {
    ParameterResponse.RESULT_SUCCESS: "success",
    ParameterResponse.RESULT_NOT_FOUND: "parameter not found",
    ParameterResponse.RESULT_TYPE_MISMATCH: "parameter type mismatch",
    ParameterResponse.RESULT_VEHICLE_ARMED: "vehicle is armed",
    ParameterResponse.RESULT_SET_FAILED: "PX4 rejected the value",
    ParameterResponse.RESULT_INVALID_OPERATION: "invalid operation",
    ParameterResponse.RESULT_INVALID_NAME: "invalid parameter name",
}

TYPE_NAMES = {
    ParameterResponse.VALUE_TYPE_INT32: "INT32",
    ParameterResponse.VALUE_TYPE_FLOAT: "FLOAT",
}


def topic(namespace: str, suffix: str) -> str:
    prefix = f"/{namespace.strip('/')}" if namespace.strip("/") else ""
    return f"{prefix}/fmu/{suffix}"


def encoded_name(name: str) -> list[int]:
    try:
        raw = name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("PX4 parameter names must be ASCII") from exc
    if not raw or len(raw) > 16:
        raise ValueError("PX4 parameter names must contain 1 to 16 characters")
    return list(raw + bytes(17 - len(raw)))


class ParameterClient(Node):
    def __init__(self, namespace: str) -> None:
        super().__init__("px4_parameter_client")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
        )
        self.publisher = self.create_publisher(
            ParameterRequest, topic(namespace, "in/parameter_request"), qos
        )
        self.subscription = self.create_subscription(
            ParameterResponse,
            topic(namespace, "out/parameter_response"),
            self._response_cb,
            qos,
        )
        self._next_id = time.monotonic_ns() & 0xFFFFFFFF
        self._response: ParameterResponse | None = None
        self._wanted_id: int | None = None

    def _response_cb(self, response: ParameterResponse) -> None:
        if response.request_id == self._wanted_id:
            self._response = response

    def exchange(
        self,
        name: str,
        operation: int,
        *,
        parameter_type: int = ParameterRequest.VALUE_TYPE_UNKNOWN,
        int_value: int = 0,
        float_value: float = 0.0,
        timeout_s: float,
    ) -> ParameterResponse:
        self._next_id = (self._next_id + 1) & 0xFFFFFFFF
        request = ParameterRequest()
        request.timestamp = self.get_clock().now().nanoseconds // 1000
        request.request_id = self._next_id
        request.operation = operation
        request.parameter_type = parameter_type
        request.name = encoded_name(name)
        request.int_value = int_value
        request.float_value = float_value

        self._wanted_id = request.request_id
        self._response = None
        deadline = time.monotonic() + timeout_s
        next_publish = 0.0

        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.publisher.publish(request)
                next_publish = now + 0.25
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - now)))
            if self._response is not None:
                return self._response

        raise TimeoutError(
            "no PX4 parameter response; verify the new firmware and DDS link"
        )


def checked(response: ParameterResponse) -> ParameterResponse:
    if response.result != ParameterResponse.RESULT_SUCCESS:
        reason = RESULT_NAMES.get(response.result, f"unknown result {response.result}")
        raise RuntimeError(reason)
    return response


def response_value(response: ParameterResponse) -> int | float:
    if response.parameter_type == ParameterResponse.VALUE_TYPE_INT32:
        return response.int_value
    if response.parameter_type == ParameterResponse.VALUE_TYPE_FLOAT:
        return response.float_value
    raise RuntimeError(f"unsupported PX4 parameter type {response.parameter_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default="",
        help="vehicle namespace without slashes, for example drone4",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    commands = parser.add_subparsers(dest="command", required=True)

    get_parser = commands.add_parser("get", help="read one parameter")
    get_parser.add_argument("name")

    set_parser = commands.add_parser("set", help="write and read back one parameter")
    set_parser.add_argument("name")
    set_parser.add_argument("value")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")

    rclpy.init()
    client = ParameterClient(args.namespace)
    try:
        current = checked(
            client.exchange(
                args.name,
                ParameterRequest.OPERATION_READ,
                timeout_s=args.timeout,
            )
        )

        if args.command == "get":
            print(
                f"{args.name} = {response_value(current)} "
                f"({TYPE_NAMES[current.parameter_type]})"
            )
            return 0

        if current.parameter_type == ParameterResponse.VALUE_TYPE_INT32:
            value = int(args.value, 0)
            written = client.exchange(
                args.name,
                ParameterRequest.OPERATION_WRITE,
                parameter_type=ParameterRequest.VALUE_TYPE_INT32,
                int_value=value,
                timeout_s=args.timeout,
            )
        elif current.parameter_type == ParameterResponse.VALUE_TYPE_FLOAT:
            value = float(args.value)
            written = client.exchange(
                args.name,
                ParameterRequest.OPERATION_WRITE,
                parameter_type=ParameterRequest.VALUE_TYPE_FLOAT,
                float_value=value,
                timeout_s=args.timeout,
            )
        else:
            raise RuntimeError(f"unsupported PX4 parameter type {current.parameter_type}")

        checked(written)
        print(
            f"{args.name} = {response_value(written)} "
            f"({TYPE_NAMES[written.parameter_type]}, written and read back)"
        )
        return 0
    finally:
        client.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
