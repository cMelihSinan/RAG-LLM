import re

input_file_path = "/Users/melihsinancubukcuoglu/Desktop/access_log"
output_file_path = "/Users/melihsinancubukcuoglu/Desktop/output.txt"

def parse_log_line(line):
    pattern = r'(\S+) - - \[(.*?)\] "(.*?)"'
    match = re.search(pattern, line)

    if match:
        ip_address = match.group(1)
        timestamp = match.group(2)
        request = match.group(3)
        page = request.split(" ")[1] if len(request.split(" ")) > 1 else "-"
        formatted_output = f"IP Address: {ip_address}, Timestamp: {timestamp}, Page: {page}"
        return formatted_output

    return None

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        formatted_line = parse_log_line(line)
        if formatted_line:
            outfile.write(formatted_line + '\n')
