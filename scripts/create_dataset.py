import argparse
import random
import numpy as np

import grpc
import tqdm
import logging
import src.plugins.minecraft_pb2_grpc as minecraft_pb2_grpc

from src.plugins.minecraft_pb2 import *
from src.helpers import generate_random_string, get_dir

parser = argparse.ArgumentParser(description='Generate an npz map from a minecraft server running.')

parser.add_argument('-s', '--server',
                    default="localhost:5001",
                    type=str)

parser.add_argument('-bs', '--block_size',
                    default=6,
                    type=int)

parser.add_argument('-o', '--output_dir',
                    default='./data/worlds',
                    type=str)

parser.add_argument('-w', '--world',
                    default="shmar",
                    type=str)

parser.add_argument('-n', '--num_blocks',
                    default=20000000,
                    type=int)

parser.add_argument('-mbt', '--min_block_type', default=4, type=int)


def get_world_slice(client, args):
    # read the cube area
    x, y, z = random.randint(-1000, 1000), random.randint(args.block_size, 128 - args.block_size), random.randint(-1000, 1000)
    dx, dy, dz = args.block_size // 2, args.block_size // 2, args.block_size // 2
    blocks = client.readCube(Cube(
        min=Point(x=x - dx, y=y - dy, z=z - dz),
        max=Point(x=x + (args.block_size - dx - 1), y=y + (args.block_size - dy - 1), z=z + (args.block_size - dz - 1))
    ))
    
    world_slice = np.full((args.block_size, args.block_size, args.block_size), AIR)

    # loop over all the data structures
    for block in blocks.ListFields()[0][1]:
        
        world_slice[
            block.position.x - x + dx,
            block.position.y - y + dy,
            block.position.z - z +  dz,
        ] = block.type
        
    return world_slice    

if __name__ == "__main__":
    logging.info("Starting world slice to npz conversion")
    
    args = parser.parse_args()
    
    assert args.min_block_type >= 1, "min_block_type must be at least 1"

    # connect to server
    logging.info(f"Connecting to server {args.server}")
    channel = grpc.insecure_channel(args.server)
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


    for _ in tqdm.tqdm(range(args.num_blocks)):
        
        world_slice = get_world_slice(client, args)
        
        while np.unique(world_slice).shape[0] < args.min_block_type:
            world_slice = get_world_slice(client, args)
        
        # logging.info(f"World slice read from server")
        save_dir = get_dir(f"{args.output_dir}/{args.world}")
        np.save(f"{save_dir}/{generate_random_string()}", world_slice)
        logging.info(f"World slice saved at {save_dir}")
