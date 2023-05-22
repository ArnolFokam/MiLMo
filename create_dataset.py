import time
import argparse
import numpy as np

import grpc
import logging
import src.plugins.minecraft_pb2_grpc as minecraft_pb2_grpc

from src.plugins.minecraft_pb2 import *
from src.helpers import generate_random_string, get_dir

parser = argparse.ArgumentParser(description='Generate an npz map from a minecraft server running.')

parser.add_argument('-s', '--server',
                    default="localhost:5001",
                    type=str)

parser.add_argument('-o', '--output_dir',
                    default='./data/worlds',
                    type=str)

parser.add_argument('-w', '--world',
                    default="shmar",
                    type=str)

parser.add_argument('-lx', '--limit_x',
                    default=[-98, -52],
                    type=list)

parser.add_argument('-ly', '--limit_y',
                    default=[15, 92],
                    type=list)

parser.add_argument('-lz', '--limit_z',
                    default=[94, 135],
                    type=list)


parser.add_argument('-mbt', '--min_block_type', default=4, type=int)


def get_world_slice(client, args):
    # read the cube area
    dx = abs(args.limit_x[0] - args.limit_x[1]) + 1 # x size of the block
    dy =  abs(args.limit_y[0] - args.limit_y[1]) + 1 # y size of the block
    dz = abs(args.limit_z[0] - args.limit_z[1]) + 1 # z size of the block
    
    world_slice = np.full((dy, dx, dz), AIR)
    
    for y in range(world_slice.shape[0]):
        blocks = client.readCube(Cube(
            min=Point(x=min(args.limit_x), y=min(args.limit_y) + y, z=min(args.limit_z)),
            max=Point(x=max(args.limit_x), y=min(args.limit_y) + y, z=max(args.limit_z))
        ))

        # loop over all the data structures
        for block in blocks.ListFields()[0][1]:
            world_slice[
                world_slice.shape[0] - (block.position.y - min(args.limit_y) + 1),
                block.position.x - min(args.limit_x),
                block.position.z - min(args.limit_z),
            ] = block.type
            
    return world_slice    

if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting world slice to npz conversion")
    
    args = parser.parse_args()

    # connect to server
    logging.info(f"Connecting to server {args.server}")
    channel = grpc.insecure_channel(args.server)
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)
    
    # get the world slice from the limit
    world_slice = get_world_slice(client, args)
    
    # logging.info(f"World slice read from server")
    name = f"{'_'.join(map(str, args.limit_x))}_{'_'.join(map(str, args.limit_y))}_{'_'.join(map(str, args.limit_z))}"
    full_path = f"{get_dir(args.output_dir)}/{args.world}_{name}_{generate_random_string()}"
    np.save(full_path, world_slice)
    logging.info(f"World slice saved at {full_path}.npy")
    logging.info(f"Took {time.time() - start} seconds")
