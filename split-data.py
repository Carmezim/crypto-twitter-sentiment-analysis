"""
Script to split data into train and test CSV files

Example usage:
    $python3 split_data.py dataset.csv
    $python3 split_data.py dataset.csv <split percentage float>
"""
import sys
from preprocess import write_status


def split(filename, split_percentage=0.1):
    print("Splitting data in %s%% train and %s%% test" % (1 - split_percentage,
                                                    split_percentage))
    save_train = open("%s-train.csv" % filename, "w", encoding="utf-8")
    save_test = open("%s-test.csv" % filename, "w", encoding="utf-8")
    with open("%s.csv" % filename, "r", encoding="utf-8") as csv:
        lines = csv.readlines()
        total = len(lines)
        split_index = int(total - split_percentage * total)
        #train, test = lines[:split_index], lines[split_index:]
        for i, line in enumerate(lines):
            if i < split_index + 1:
                save_train.write(line)
            else:
                save_test.write(line)
            write_status(i + 1, total)
        save_test.close()
        save_train.close()
        print("\nData successfully split and saved in %s-train.csv and "
                                        "%s-test.csv" % (filename, filename))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 split-data.py <CSV>')
        exit()
    csv_file_name = sys.argv[1].split(".")[0]
    print(csv_file_name)
    if len(sys.argv) == 3:
        if sys.argv[2]:
            split_percentage = float(sys.argv[2])
            split(csv_file_name, split_percentage)
    else:
        split(csv_file_name)
