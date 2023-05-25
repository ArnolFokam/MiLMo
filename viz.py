import argparse
import glob
import os
from typing import List

import numpy as np

import grpc

import src.plugins.minecraft_pb2_grpc as minecraft_pb2_grpc
from src.plugins.minecraft_pb2 import *


parser = argparse.ArgumentParser(description='Generate a world in minecraft server from npz.')

parser.add_argument('-s', '--server',
                    default="localhost:5001",
                    type=str)

parser.add_argument('-c', '--center_position',
                    default=[0, 4, 0],
                    type=List[int])

parser.add_argument('-d', '--saved_blocks_dir',
                    type=str)

# TODO: make air to remove everything

def create_blocks(world_slice, x, y, z):
    blocks_list = []
    for j in range(world_slice.shape[0]):
        for i in range(world_slice.shape[1]):
            for k in range(world_slice.shape[2]):
                blocks_list.append(
                    Block(position=Point(x=x + i, y=y + j, z=z + k),
                        type=int(world_slice[j, i, k]), orientation=NORTH),
                )

    if len(blocks_list) != 0:
        client.spawnBlocks(Blocks(blocks=blocks_list))


if __name__ == "__main__":
    args = parser.parse_args()

    # connect to server
    channel = grpc.insecure_channel(args.server)
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

    # read the cube area
    cx, cy, cz = args.center_position
    samples = np.asarray([np.load(x) for x in glob.glob(os.path.join(args.saved_blocks_dir, "samples", "*.npy"))])
    generations = np.asarray([np.load(x) for x in glob.glob(os.path.join(args.saved_blocks_dir, "generations", "*.npy"))])
    
    assert generations.shape == samples.shape
    
    # seperation between the true samples and the generated
    samples_to_generations_seperation = 10
    
    # x-intra instance seperation
    x_instance_seperation = 5
    
    # z-intra instance seperation
    z_instance_seperation = 5
    
    rows = int(np.ceil(np.sqrt(len(samples))))
    cols = int(np.ceil(np.sqrt(len(samples))))
    
    # total x of the array of generation
    width = rows * (x_instance_seperation * 2 + samples.shape[2])
    
    # total z of the array of generation
    height = cols * (z_instance_seperation * 2 + samples.shape[3])
    
    i, j = 0, 0
    for sample, generation in zip(samples, generations):
        # find the correct coordinates in the world for the samples
        sx = cx + i * (x_instance_seperation * 2 + samples.shape[2])
        sz = cz + j * (z_instance_seperation * 2 + samples.shape[3])
        
        # find the correct corrdin# blocks_list = []
        gx = sx + width + samples_to_generations_seperation
        gz = sz
        
        # generate the samples and generations
        samples_blocks = create_blocks(sample, sx, cy, sz)
        generations_blocks = create_blocks(generation, gx, cy, gz)
        
        # update the coordinates
        j += 1
        if j == cols:
            j = 0
            i += 1

    print(f"World generated")
