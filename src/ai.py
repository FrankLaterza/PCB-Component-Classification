import argparse
import sam
import clfy
import os
import torch

def print_pytorch_config():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

def train(epoch):
    print(f"running train with epoch: {epoch}")
    clfy.train_model(epoch)
    return

def classify(file_path):
    print(f"running classification on file: {file_path}")
    print(clfy.predict_image(file_path))

def segment(file_path):
    print(f"running separation on file: {file_path}")
    sam.run_sam(file_path)

def full(file_path):
    print(f"running full on file: {file_path}")

    # segment(file_path)
    directory = "./obj"
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        classify(full_path)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI created to classify components on a PCB."
    )
    parser.add_argument(
        '-train',
        type=int,
        help="use to train dataset"
    )
    parser.add_argument(
        '-classify',
        type=str,
        help="path to the file for the classification process."
    )
    parser.add_argument(
        '-segment',
        type=str,
        help="path to the file for the segment process."
    )
    parser.add_argument(
        '-full',
        type=str,
        help="path to the file for the full process."
    )
    return parser.parse_args()


# main entry point
if __name__ == '__main__':
    args = parse_arguments()

    print_pytorch_config()

    if args.train:
        train(args.train)

    if args.classify:
        classify(args.classify)

    if args.segment:
        segment(args.segment)

    if args.full:
        full(args.full)

    if not (args.full or args.segment or args.classify or args.train):
        print("no action specified. use -train, -full FILE, -separate PATH, or -classify PATH.")