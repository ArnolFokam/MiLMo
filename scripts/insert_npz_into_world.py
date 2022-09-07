import argparse
from typing import List

import numpy as np

import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *


parser = argparse.ArgumentParser(description='Generate a world in minecraft server from npz.')

parser.add_argument('-s', '--server',
                    default="localhost:5001",
                    type=str)

parser.add_argument('-c', '--center_position',
                    default=[0, 28, 0],
                    type=List[int])

parser.add_argument('-r', '--radius',
                    default=[28, 28, 28],
                    type=List[int])

parser.add_argument('-n', '--npz_dir',
                    type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    # connect to server
    channel = grpc.insecure_channel(args.server)
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

    # read the cube area
    x, y, z = args.center_position
    dx, dy, dz = args.radius
    world_slice = np.load(args.npz_dir)

    blocks_list = []
    for i in range(world_slice.shape[0]):
        for k in range(world_slice.shape[2]):
            for j in range(world_slice.shape[1]):
                blocks_list.append(
                    Block(position=Point(x=x + i - dx, y=j, z=z + k - dz),
                          type=int(world_slice[i, j, k]), orientation=NORTH),
                )

    if len(blocks_list) != 0:
        client.spawnBlocks(Blocks(blocks=blocks_list))

    print(f"World generated")
