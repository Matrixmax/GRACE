import logging
import re
import shutil
import tempfile
import time
from pathlib import Path

from query.run import get_files
from run.runner import Runner
from sparrow_schema.schema import sparrow

file_list = dict()


# lib库与抽取器分开,逻辑上可以有一个库只有lib库无抽取器。
def open_lib():
    lib = ["java", "javascript", "go", "xml", "python", "sql", "cfamily", "properties", "swift", "arkts"]
    return lib


def get_rebuild_lang(args):
    lib_list = list()
    for lang in args.language:
        if lang == "all":
            lib_list += open_lib()
            continue
        lib_list.append(lang)
    lib_list = list(set(lib_list))
    return lib_list


def extract_first_number(string):
    pattern = r"\d+(?=.*\|)"
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None


def line_replace(output, line, base):
    actual_line = int(line) - base
    return output.replace(line, " " * (len(line) - len(str(actual_line))) + str(actual_line), 1)


def log_mapping(output):
    result = output
    if "-->" in output:
        match = re.search(r'-->\s*([^:]+):(\d+)', output)
        if match:
            file_name = match.group(1)
            line = match.group(2)
            num = 0
            for file in file_list:
                num += file["lines"]
                if int(line) <= num:
                    result = output.replace(file_name, str(file["name"]), 1)
                    result = line_replace(result, line, num - file["lines"])
                    break
    else:
        line = extract_first_number(output)
        if line is not None:
            num = 0
            for file in file_list:
                num += file["lines"]
                if int(line) <= num:
                    result = line_replace(output, line, num - file["lines"])
                    break
    return result


# 日志输出修改:目前将lib和一进行编译，此处out通过行号将错误信息对应回原文件
def log_info(output):
    if output:
        output = log_mapping(output)
        logging.info(output.strip())


def log_error(output):
    if output:
        output = log_mapping(output)
        logging.error(output.strip())


def rebuild_lang(lib, verbose):
    global file_list
    lib_path = sparrow.lib_script / "coref" / lib
    if not Path(lib_path).exists():
        logging.warning("lib %s not exist, please check", str(lib_path))
        return
    file_list = list()
    gdl_list = get_files(lib_path, ".gs")
    gdl_list += get_files(lib_path, ".gdl")
    with tempfile.NamedTemporaryFile(suffix='.gdl', mode="w+") as output_file:
        output_file.writelines("// script\n")
        file_list.append({"name": "begin", "lines": 1})
        num = 1
        for file_name in gdl_list:
            with open(file_name, "r") as file:
                lines = file.readlines()
                if len(lines) == 0:
                    continue
                last_element = lines[-1]
                if not last_element.endswith('\n'):
                    last_element += '\n'
                    lines[-1] = last_element
                output_file.writelines(lines)
                file_list.append({"name": file_name, "lines": len(lines)})
                num += len(lines)
        output_file.seek(0)
        cmd = list()
        cmd += [str(sparrow.godel_script), "--semantic-only", str(output_file.name)]
        if verbose:
            cmd += ["--verbose"]
        tmp = Runner(cmd, 3600)
        start_time = time.time()
        return_code = tmp.subrun(log_info, log_error)
        if return_code != 0:
            logging.error("%s lib compile error, please check it yourself", lib_path)
            return -1
        shutil.copy2(output_file.name, sparrow.lib_script / "coref" / (lib + ".gdl"))
        logging.info("%s lib compile success time: %.2fs", lib_path, time.time() - start_time)


def rebuild_lib(args):
    lib_list = get_rebuild_lang(args)
    status = 0
    for lib in lib_list:
        if rebuild_lang(lib, args.verbose) == -1:
            status = -1
    if status == -1:
        raise RuntimeError("lib compile fail")
