import argparse
import numpy as np

import grpc

import minecraft_pb2_grpc

from minecraft_pb2 import *
from src.helpers import generate_random_string, get_dir

parser = argparse.ArgumentParser(description='Generate an npz map from a minecraft server running.')

parser.add_argument('-s', '--server',
                    default="localhost:5001",
                    type=str)

parser.add_argument('-c', '--center_position',
                    default=[0, 28, 0],
                    nargs="*",
                    type=int)

parser.add_argument('-r', '--radius',
                    default=[28, 28, 28],
                    nargs="*",
                    type=int)

parser.add_argument('-o', '--output_dir',
                    default='./outputs/worlds',
                    type=str)

parser.add_argument('-w', '--world',
                    type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    assert all(x > 0 for x in args.radius), "Radius be positive"
    assert len(args.radius) == 3, "only 3D coordinates are allowed for radius"
    assert len(args.center_position) == 3, "only 3D coordinates are allowed for center"
    assert args.center_position[1] - args.radius[1] >= 0, "Underground not supported"

    # connect to server
    channel = grpc.insecure_channel(args.server)
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

    # read the cube area
    x, y, z = args.center_position
    dx, dy, dz = args.radius
    blocks = client.readCube(Cube(
        min=Point(x=x - dx, y=y - dy, z=z - dz),
        max=Point(x=x + dx, y=y + dy, z=z + dz)
    ))

    world_slice = np.full((dx*2 + 1, dy*2 + 1, dz*2 + 1), AIR)

    # loop over all the data structure
    for block in blocks.ListFields()[0][1]:
        world_slice[
            block.position.x + dx,
            block.position.y,
            block.position.z + dz,
        ] = block.type

    save_dir = get_dir(args.output_dir)
    np.save(f"{save_dir}/{args.world}_{generate_random_string()}", world_slice)
    print(f"World slice saved at {save_dir}")
