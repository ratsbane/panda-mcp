"""Entry point for `python -m franka_rt`."""

import argparse
import logging

from .server import FrankaRTServer


def main():
    parser = argparse.ArgumentParser(description="franka-rt ZMQ server")
    parser.add_argument(
        "--hostname", default="192.168.0.253",
        help="Franka Panda IP address (default: 192.168.0.253)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    server = FrankaRTServer(hostname=args.hostname)
    server.run()


if __name__ == "__main__":
    main()
