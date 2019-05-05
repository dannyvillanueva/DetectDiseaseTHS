from ths.utils.synomymos import OverSampleTweetsZeroFile
import sys


def main(input_file_name, output_file_name):
    over_sample = OverSampleTweetsZeroFile(input_file_name=input_file_name, output_file_name=output_file_name)
    over_sample.oversample()

if __name__ == "__main__":
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    main(input_file_name=input_file_name, output_file_name=output_file_name)