import argparse
import sam
import clfy
import os

def train(epoch):
    print(f"running train with epoch: {epoch}")
    clfy.train_model(epoch)
    return

def separate(file_path):
    print(f"running separation on file: {file_path}")
    sam.run_sam(file_path)

def classify(file_path):
    print(f"running classification on file: {file_path}")
    print(clfy.predict_image(file_path))

def full(file_path):
    print(f"running full on file: {file_path}")

    # separate(file_path)
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
        '-separate',
        type=str,
        help="path to the file for the separation process."
    )
    parser.add_argument(
        '-classify',
        type=str,
        help="path to the file for the classification process."
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

    if args.train:
        train(args.train)

    if args.separate:
        separate(args.separate)

    if args.classify:
        classify(args.classify)

    if args.full:
        full(args.full)

    if not (args.full or args.separate or args.classify):
        print("no action specified. use -train, -full FILE, -separate PATH, or -classify PATH.")