import base64
import struct


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def png_pixels(png: bytes) -> int:
    width, height = struct.unpack(">II", png[16:24])
    return width * height
