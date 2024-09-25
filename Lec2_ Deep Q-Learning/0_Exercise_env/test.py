"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"The current working directory is: {current_working_directory}")

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = "C:/1 - eigenes/Lehrauftrag - BHT/Vorlesungsreihe WS23-24/Lec2_ Deep Q-Learning/src/f4"
os.chdir(new_working_directory)

# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")

# -------------------------------------------



import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(opt.saved_path))
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction) #[0]

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)
