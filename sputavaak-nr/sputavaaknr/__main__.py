import argparse
from sputavaaknr.denoise import denoise
import os

def main():
    parser = argparse.ArgumentParser(description="Example script with CLI parameters")

    # Define parameters
    parser.add_argument("-i", "--input", default="input.mp3", help="Path to input mp3 file")
    parser.add_argument("-o", "--output", default="output.mp3", help="Path to output file")
    parser.add_argument("-b", "--bitrate", default="128k", help="Bitrate for encoding (default: 128k)")

    args = parser.parse_args()

    try:
        assert os.path.exists(args.input), f"Input file does not exist: \"{args.input}\""
        assert args.output != args.input, f"input and output: \"{args.input}\" must be different"
    except AssertionError as e:
        print(f"Error: {e}")
        exit(1)

    # Access parameters
    print(f"Input: {args.input}  -> {args.output} (bitrate: {args.bitrate})")
    denoise(args.input, args.output, args.bitrate)

if __name__ == "__main__":
    main()

