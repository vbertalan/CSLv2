def get_max_line_length_and_average(file_path):
    max_length = 0
    total_tokens = 0
    num_lines = 0

    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            length = len(tokens)
            total_tokens += length
            num_lines += 1
            if length > max_length:
                max_length = length

    average_length = total_tokens / num_lines if num_lines > 0 else 0
    return max_length, average_length


if __name__ == "__main__":
    input_file = "all_sequences.txt"  # replace with your actual file path
    max_len, avg_len = get_max_line_length_and_average(input_file)
    print(f"The longest line has {max_len} tokens.")
    print(f"The average line length is {avg_len:.2f} tokens.")
