from colorama import Fore, Style, init
import json
init()

def print_colored_text(text, color):
    """
    打印带颜色的文本。

    :param text: 要打印的文本（字符串）。
    :param color: 字体颜色（字符串，可选值：'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'）。
    """
    color_map = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
    }
    if color not in color_map:
        raise ValueError(f"不支持的颜色: {color}。可选颜色: {list(color_map.keys())}")
    return color_map[color] + text + Style.RESET_ALL

# 读取jsonl文件
def read_jsonl_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 写入jsonl文件
def write_jsonl_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n') 

if __name__ == '__main__':
    colored_text = print_colored_text('Hello, World', 'red')
    print(colored_text)