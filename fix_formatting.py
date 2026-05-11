import os
import re


def fix_trailing_whitespace(filepath):
    """删除行尾空格"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 删除行尾空格，但保留换行符
    lines = [re.sub(r"[ \t]+$", "", line.rstrip("\n")) + "\n" for line in lines]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Fixed trailing whitespace: {filepath}")


def add_newline_at_end(filepath):
    """添加文件末尾换行符"""
    with open(filepath, "rb") as f:
        content = f.read()

    if content and not content.endswith(b"\n"):
        with open(filepath, "ab") as f:
            f.write(b"\n")
        print(f"Added newline at end: {filepath}")


# 需要修复的文件列表
files = [
    ".github/workflows/cd.yml",
    ".github/workflows/ci.yml",
    "README.md",
    "CONTRIBUTING.md",
    ".vscode/settings.json",
    "pyproject.toml",
    "pytest.ini",
    "examples/models/imdb_dataset/README.md",
    "examples/models/bert-base-uncased/tokenizer.json",
    ".gitignore"
]

for f in files:
    if os.path.exists(f):
        fix_trailing_whitespace(f)
        add_newline_at_end(f)

print("Done!")
