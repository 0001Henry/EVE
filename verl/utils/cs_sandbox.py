import json
import os
import re
import contextlib
import ast
import sys
import io
import stat # 新增：用于处理文件权限

# import contextlib # Duplicate import
import random
import string
import shutil
import PIL
from PIL import Image, UnidentifiedImageError, ImageColor, ImageDraw, ImageEnhance, ImageOps, ImageMath, ImageFilter # Pillow for image processing
import numpy # For numerical operations, often used with cv2
import math # For mathematical operations
import multiprocessing  # For timeout control
import timeout_decorator    # For more efficient timeout control
import glob
import importlib
import types
import pickle
import textwrap # 导入 textwrap 模块

import autopep8

# Attempt to import cv2, but make it optional if not strictly needed by all sandbox uses
try:
    import cv2
except ImportError:
    cv2 = None # Keep track if cv2 is not available
    print("Warning: OpenCV (cv2) not found. cv2-dependent sandboxed code will fail.", file=sys.stderr)


# --- Enhanced Security: Prohibit dangerous system calls ---
DANGEROUS_PATTERNS = [
    r'\bsys\.',
    # r'\bshutil\.', # 我们将通过代理对象来控制，而不是完全禁止
    r'\bsocket\.',
    r'\bsubprocess\.',
    r'\bexec\(', r'\beval\(',
    r'\bcompile\(',
    r'\b__import__\(',
    # --- 新增/强化：明确禁止已知的删除函数调用 ---
    r'\bos\.(remove|unlink|rmdir)\b',
    r'\bshutil\.rmtree\b',
    r'\bshutil\.move\b',      # 禁止 shutil.move
    r'\bos\.(rename|renames)\b', # 禁止 os.rename 和 os.renames
    # ... (其他危险模式保持不变) ...
]


def check_dangerous_code(code_string):
    """
    Performs a basic static analysis to detect dangerous patterns in the code.
    Returns True if dangerous patterns are found, False otherwise.
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code_string, re.IGNORECASE):
            return True
    return False

EXEC_TIME_LIMIT=180
TEMP_PROCESSED_IMAGES_DIR = os.environ.get("TEMP_PROCESSED_IMAGES_DIR", "./temp_processed_images/")

class ReadOnlyPath:
    """
    Context manager that temporarily makes a file read-only during code execution.
    This prevents sandboxed code from modifying original input files.
    """
    def __init__(self, path):
        self.path = path if isinstance(path, str) else None
        self.original_permissions = None

    def __enter__(self):
        """进入上下文时，保存原始权限并将文件设为只读。"""
        # 仅当路径是有效文件时才继续
        if self.path and os.path.isfile(self.path):
            try:
                # 获取当前权限模式
                self.original_permissions = os.stat(self.path).st_mode
                # 移除用户、组和其他人的写权限
                read_only_permissions = self.original_permissions & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                if self.original_permissions != read_only_permissions:
                    os.chmod(self.path, read_only_permissions)
            except OSError as e:
                # 如果我们没有权限更改文件的权限（例如，不是文件所有者），则打印警告
                print(f"Warning: Could not make '{self.path}' read-only: {e}", file=sys.stderr)
                self.original_permissions = None # 确保我们不会在退出时尝试恢复权限
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时，恢复文件的原始权限。"""
        # 仅当原始权限被成功保存时才恢复
        if self.path and self.original_permissions is not None and os.path.isfile(self.path):
            try:
                os.chmod(self.path, self.original_permissions)
            except OSError as e:
                # 如果恢复失败，则打印警告
                print(f"Warning: Could not restore original permissions for '{self.path}': {e}", file=sys.stderr)


def align_first_line_to_second(code_string: str) -> str:
    """
    检查代码字符串，找到第一个和第二个非空行。如果它们的缩进不同，
    就将第一个非空行的缩进修改为与第二个非空行一致。

    Args:
        code_string: 输入的多行代码字符串。

    Returns:
        修复了缩进的代码字符串；如果代码行数不足，则返回原字符串。
    """
    lines = code_string.splitlines()

    first_line_info = None
    second_line_info = None

    # 1. 遍历所有行，查找第一个和第二个非空行及其信息（行号和内容）
    for index, line_content in enumerate(lines):
        # 如果行strip()后不为空，说明是有效行
        if line_content.strip():
            # 如果还没找到第一个，则记录当前行为第一个
            if first_line_info is None:
                first_line_info = {'index': index, 'content': line_content}
            # 如果已找到第一个但未找到第二个，则记录当前行为第二个
            elif second_line_info is None:
                second_line_info = {'index': index, 'content': line_content}
                # 两个都找到了，可以停止搜索
                break
    
    
    # 2. 检查是否找到了足够的行来进行比较和修复
    # 如果代码中少于两个非空行，则无法操作，直接返回原样

    if not first_line_info or not second_line_info:
        return code_string
    # 3. 提取两行的内容和缩进
    
    first_line_content = first_line_info['content']
    second_line_content = second_line_info['content']

    # 计算各自的缩进
    # len(line) - len(line.lstrip(' ')) 是计算前导空格的可靠方法
    first_line_indent = ' ' * (len(first_line_content) - len(first_line_content.lstrip(' ')))
    second_line_indent = ' ' * (len(second_line_content) - len(second_line_content.lstrip(' ')))
    # 4. 如果缩进不一致，则进行修复
    if first_line_indent != second_line_indent:
        # 获取第一个非空行的原始索引
        original_index = first_line_info['index']
        # 移除该行的所有前导空格
        stripped_content = first_line_content.lstrip(' ')
        # 将第二行的缩进应用到第一行，并在原始位置更新
        lines[original_index] = second_line_indent + stripped_content
        
    # 5. 将可能被修改过的行列表重新组合成一个字符串并返回
    return "\n".join(lines)


def get_image_paths(temp_output_dir):
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(temp_output_dir, f'*.{ext}')
        image_paths.extend(glob.glob(pattern))

    return image_paths

VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
ORIGINAL_OUTPUT_PREFIX = os.environ.get('ORIGINAL_OUTPUT_PREFIX', './temp_processed_images/')

class ImagePathTransformer(ast.NodeTransformer):
    def __init__(self, replacement_path):
        self.replacement_path = replacement_path
        self.path_was_replaced = False

    def visit_Assign(self, node):
        # Process simple assignments: target_variable = "string_literal"
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_variable_name = node.targets[0].id
            
            if target_variable_name == 'image_path': # Target specific variable name
                current_path_value = None
                # Extract string value from ast.Constant (Python 3.8+) or ast.Str (older Python)
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    current_path_value = node.value.value
                elif isinstance(node.value, ast.Str): # For Python < 3.8
                    current_path_value = node.value.s
                
                if current_path_value is not None:
                    is_valid_image_on_disk = False
                    if os.path.exists(current_path_value) and \
                       os.path.isfile(current_path_value) and \
                       any(current_path_value.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS):
                        is_valid_image_on_disk = True
                    
                    if not is_valid_image_on_disk:
                        # Replace the value of the AST node
                        if hasattr(ast, 'Constant'): # Python 3.8+
                             node.value = ast.Constant(value=self.replacement_path)
                        else: # Python < 3.8
                             node.value = ast.Str(s=self.replacement_path)
                        self.path_was_replaced = True
                        
        return self.generic_visit(node) # Continue traversing

MIN_CROP_DIMENSION = 64

# import os # Duplicate import
# import ast # Duplicate import
# Ensure Pillow is available for image dimension reading
pil_available = False
# Image = None # Image is already imported from PIL at the top
try:
    from PIL import Image # This ensures Image is the class from PIL
    pil_available = True
except ImportError:
    print("Warning: PIL (Pillow) is not installed. Crop coordinate clamping will be skipped or limited.")

def ensure_temp_dir(temp_output_dir):
    """Ensures the temporary directory for processed images exists."""
    os.makedirs(temp_output_dir, exist_ok=True)
    # Also ensure the directory is writable (basic check)
    if not os.access(temp_output_dir, os.W_OK):
        raise PermissionError(f"Temporary directory {temp_output_dir} is not writable.")




def execute_code_in_sandbox(code_to_execute, input_image_path, item_id="N/A", 
                            temp_output_dir=TEMP_PROCESSED_IMAGES_DIR,
                            previous_execution_context=None):
    """
    Executes Python code in a restricted sandbox using multiprocessing for timeout control.
    """
    if check_dangerous_code(code_to_execute):
        return [], "", f"Sandbox for {item_id}: Code contains potentially dangerous system operations such as remove. Execution denied.", None

    ensure_temp_dir(temp_output_dir)
    
    with ReadOnlyPath(input_image_path):
        try:
            # 使用 multiprocessing.Manager 创建共享字典
            manager = multiprocessing.Manager()
            result_dict = manager.dict()
            
            # 创建进程执行沙盒代码
            process = multiprocessing.Process(
                target=_sandboxed_execution_wrapper,
                args=(code_to_execute, input_image_path, temp_output_dir, 
                      item_id, previous_execution_context, result_dict)
            )
            
            process.start()
            process.join(timeout=EXEC_TIME_LIMIT)
            
            if process.is_alive():
                # 超时，终止进程
                process.terminate()
                process.join()
                error_msg = f"Sandbox for {item_id}: Execution timed out after {EXEC_TIME_LIMIT} seconds."
                print(error_msg)
                return [], "", error_msg, None
            
            # 从共享字典中提取结果
            processed_paths_list = result_dict.get("processed_paths_list", [])
            full_print_output = result_dict.get("print_output", "")
            error_msg = result_dict.get("error", None)
            current_execution_context = result_dict.get("execution_context", {'globals': {}, 'locals': {}})
            
        except Exception as e:
            error_msg = f"Sandbox for {item_id}: Unexpected error: {e}"
            print(error_msg)
            return [], "", error_msg, None
    
    return processed_paths_list, full_print_output, error_msg, current_execution_context



def _sandboxed_execution_wrapper(code_to_execute, input_image_path, temp_output_dir, 
                                 item_id, previous_execution_context, result_dict):
    """
    包装函数，在子进程中执行沙盒代码，并将结果写入共享字典
    """
    try:
        # 移除 @timeout_decorator.timeout 装饰器，直接执行原函数逻辑
        result = _execute_sandbox_logic(
            code_to_execute, input_image_path, temp_output_dir, 
            item_id, previous_execution_context
        )
        # 将结果写入共享字典
        for key, value in result.items():
            result_dict[key] = value
    except Exception as e:
        result_dict["error"] = f"Sandbox execution error: {e}"


def _execute_sandbox_logic(code_to_execute, input_image_path, temp_output_dir, item_id, 
                           previous_execution_context=None):
    """
    原 _sandboxed_execution_target 的核心逻辑，移除装饰器
    """
    # ... (保持原有的所有沙盒执行逻辑不变) ...
    # 直接复制 _sandboxed_execution_target 函数体的内容到这里
    return_dict = {}
    
    # Prepare a restricted environment for exec()
    allowed_builtins = {
        "print": print, "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple, "range": range, "round": round,
        "abs": abs, "min": min, "max": max, "sum": sum, "sorted": sorted,
        "any": any, "all": all, "zip": zip, "map": map, "filter": filter,
        "True": True, "False": False, "None": None, "isinstance": isinstance,
        "issubclass": issubclass, "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "AttributeError": AttributeError, "IndexError": IndexError,
        "KeyError": KeyError, "NotImplementedError": NotImplementedError,
        "enumerate": enumerate, "pow": pow, "divmod": divmod, "bin": bin,
        "oct": oct, "hex": hex, "complex": complex, "__import__": __import__,
        "globals": globals, "locals": locals, "open": open, 
        # Modules accessible via __import__ or provided in sandbox_globals
        "os": os, "shutil": shutil, 
        "itertools": __import__('itertools'), "re": __import__('re'), "time": __import__('time'),
        "datetime": __import__('datetime'), "math": __import__('math'), "cmath": __import__('cmath'),
        "collections": __import__('collections'), "json": json, "PIL": __import__('PIL'),
        "random": random, "UnidentifiedImageError": UnidentifiedImageError, "bool": bool,
    }
    sandbox_globals = {
        "__builtins__": allowed_builtins,
        "os": os,
        "random": random,
        "string": string,
        "math": math,
        "Image": Image, 
        "UnidentifiedImageError": UnidentifiedImageError,
        "numpy": numpy, 
        "np": numpy,   # TODO: pandayin: Check this? this seems like a duplicate
        "json": json, # Make json directly available
        "re": re,     # Make re directly available
        "shutil": shutil,
        "cv2": cv2,
        "cv": cv2,  # Alias 'cv' to 'cv2' for compatibility
        "PIL": PIL,
        "ImageDraw": ImageDraw,
        "ImageOps": ImageOps,
        "ImageEnhance": ImageEnhance,
        "ImageMath": ImageMath,
        "ImageColor": ImageColor,
        "ImageFilter": ImageFilter,
        # PIL is available via __import__ or Image class
    }

    if cv2:
        sandbox_globals["cv2"] = cv2

    sandbox_globals_always = sandbox_globals
    sandbox_locals = {
        "image_path": input_image_path,
        "temp_output_dir": temp_output_dir
    }

    if previous_execution_context:
        # Restore picklable local variables from the previous step
        sandbox_locals.update(previous_execution_context.get('locals', {}))
        
        # Re-create imported modules from the previous step
        imports_to_recreate = previous_execution_context.get('globals', {})
        for alias, module_name in imports_to_recreate.items():
            try:
                # Re-import the module and add it to the current step's locals
                module_obj = importlib.import_module(module_name)
                sandbox_locals[alias] = module_obj
            except ImportError:
                print(f"Warning: Could not re-import module '{module_name}' (as '{alias}') from previous step.")

    code_to_execute = align_first_line_to_second(code_to_execute)
    if autopep8:
        try:
            dedented_code = textwrap.dedent(code_to_execute).strip()
            code_to_execute = dedented_code
            formatted_code = autopep8.fix_code(code_to_execute, options={'aggressive': 2})
            # if formatted_code.strip() != code_to_execute.strip():
            #     print(f"INFO: Attempted to auto-format code for {item_id} using autopep8.")
            code_to_execute = formatted_code
        except Exception as e:
            pass
            # print(f"WARNING: Error during autopep8 formatting for {item_id}: {e}. Proceeding with original code.", file=sys.stderr)
    else:
        # print(f"INFO: autopep8 not available, skipping auto-formatting for {item_id}.", file=sys.stderr)
        pass

    # ... (AST transformations: ImagePathTransformer, CropCoordinateTransformer, OpenCVNamespaceTransformer) ...
    # (These transformations should operate on code_to_execute before exec)
    if input_image_path and isinstance(input_image_path, str):
        try:
            tree = ast.parse(code_to_execute)
            transformer = ImagePathTransformer(input_image_path) 
            new_tree = transformer.visit(tree)
            if transformer.path_was_replaced:
                if hasattr(ast, 'unparse'): 
                    code_to_execute = ast.unparse(new_tree)
                else:
                    print("Warning: ast.unparse not available (requires Python 3.9+). Code for image_path replacement not updated. Consider installing 'astor' for older Python versions or upgrading Python.")
        except SyntaxError as e:
            print(f"Syntax error when parsing code for image_path replacement: {e}")
        except Exception as e:
            print(f"An error occurred during image_path AST transformation: {e}")

    actual_image_width, actual_image_height = None, None
    if input_image_path and os.path.isfile(input_image_path):
        try:
            with Image.open(input_image_path) as img_obj:
                actual_image_width, actual_image_height = img_obj.size
            if actual_image_width is None or actual_image_height is None :
                print(f"WARNING: Could not determine dimensions for '{input_image_path}' using PIL.")
        except Exception as e:
            print(f"WARNING: Could not read image dimensions from '{input_image_path}' using PIL: {e}")
        
    captured_stdout = io.StringIO()
    processed_path_from_code = None
    processed_paths_list = []
    error_msg = None
    full_print_output = None
    exec_scope = dict(sandbox_globals)
    exec_scope.update(sandbox_locals)

    try:
        with contextlib.redirect_stdout(captured_stdout):
            original_prefix = os.environ.get('ORIGINAL_OUTPUT_PREFIX', './temp_processed_images/')
            replacement_target_path = temp_output_dir
            if original_prefix.endswith('/') and not temp_output_dir.endswith('/'):
                replacement_target_path = temp_output_dir + '/'
            
            # Create subdirectories if implied by paths in code
            quoted_path_pattern_str = r"(['\"])(" + re.escape(original_prefix) + r"[^'\"]*)\1"
            quoted_path_pattern = re.compile(quoted_path_pattern_str)
            unique_paths_found = set()
            for match in quoted_path_pattern.finditer(code_to_execute):
                unique_paths_found.add(match.group(2)) 
            for full_path_str in unique_paths_found:
                if full_path_str.startswith(original_prefix):
                    # relative_path_suffix = full_path_str[len(original_prefix):]
                    # print('full_path_str', unique_paths_found)
                    relative_path_suffix = full_path_str[len(original_prefix):].replace(' ', '')
                    if relative_path_suffix: 
                        path_directory_part = os.path.dirname(relative_path_suffix)
                        if path_directory_part: 
                            target_subdir_to_create = os.path.join(temp_output_dir, path_directory_part)
                            os.makedirs(target_subdir_to_create, exist_ok=True)
            
            code_to_execute = code_to_execute.replace(original_prefix, replacement_target_path)
            code_to_execute = re.sub(r':\.\d{1}f}', ':.8f}', code_to_execute)
            exec(code_to_execute, exec_scope, exec_scope)


        full_print_output = captured_stdout.getvalue().strip()

        if "processed_path" in exec_scope and (not previous_execution_context or not "processed_path" in previous_execution_context.get('locals', {})):
            processed_path_from_code = exec_scope["processed_path"]
            if isinstance(processed_path_from_code, str) and \
               processed_path_from_code.startswith(temp_output_dir) and \
               os.path.isfile(processed_path_from_code):
                processed_paths_list.append(processed_path_from_code)
            elif isinstance(processed_path_from_code, str) and \
                 processed_path_from_code.startswith(temp_output_dir) and \
                 not os.path.exists(processed_path_from_code):
                error_msg = f"Sandbox for {item_id}: 'processed_path' variable set to '{processed_path_from_code}', but file does not exist."
                processed_path_from_code = None 
            elif processed_path_from_code is not None:
                error_msg = f"Sandbox for {item_id}: 'processed_path' variable was '{processed_path_from_code}', which is not a valid file path in {temp_output_dir}."
                processed_path_from_code = None
        # print(full_print_output)
        if full_print_output:
            path_search_pattern = rf"({re.escape(temp_output_dir)}[^\s\'\"]+\.(?:jpg|jpeg|png|bmp|gif|tiff))"
            possible_image_path_list = get_image_paths(temp_output_dir)
            possible_error_msg = ""
            num_parse_images = 0
            for match in re.finditer(path_search_pattern, full_print_output):
                potential_path_from_print = match.group(1)
                num_parse_images += 1
                if os.path.isfile(potential_path_from_print):
                    if potential_path_from_print not in processed_paths_list:
                        processed_paths_list.append(potential_path_from_print)
                elif not error_msg: 
                    possible_error_msg = f"Sandbox for {item_id}: Path '{potential_path_from_print}' found in print, but file does not exist or is not a file."
            
            if len(processed_paths_list) == 0:
                if num_parse_images == len(possible_image_path_list):
                    processed_paths_list = possible_image_path_list
                else:
                    path_search_pattern = rf"([^\s\'\"]+\.(?:jpg|jpeg|png|bmp|gif|tiff))"
                    list_of_all_matches = re.findall(path_search_pattern, full_print_output)
                    if len(list_of_all_matches) == len(possible_image_path_list):
                        processed_paths_list = possible_image_path_list
                    elif possible_error_msg:
                        error_msg = possible_error_msg
            # import pdb; pdb.set_trace()
            if not processed_paths_list and temp_output_dir not in code_to_execute:
                normalized_output = full_print_output.strip()
                if normalized_output and os.path.isfile(normalized_output):
                    processed_paths_list.append(normalized_output)

    except ImportError as e:
        error_msg = f"Sandbox for {item_id}: Code execution failed due to ImportError. Ensure all required modules are available and correctly named: {e}"
        if 'cv2' in str(e).lower() and not cv2:
            error_msg += f"(Note: cv2 was not available in the sandbox host environment): {e}"
    except MemoryError as e:
        error_msg = f"Sandbox for {item_id}: Code execution failed due to MemoryError. The operation likely consumed too much memory: {e}"
    except SyntaxError as e: # Catch syntax errors from exec itself
        error_msg = f"Sandbox for {item_id}: Code execution failed due to SyntaxError: {e}"
    except Exception as e:
        error_msg = f"Sandbox for {item_id}: Code execution failed: {e}"

    return_dict["processed_paths_list"] = processed_paths_list
    return_dict["print_output"] = full_print_output
    return_dict["error"] = error_msg

    if full_print_output is not None and not processed_paths_list:
        # Keep non-path print outputs valid for general-purpose safe code execution.
        # Only mark as error when output appears to be a file path that cannot be matched.
        path_like_output = bool(re.search(r"[\\/].+\.(?:jpg|jpeg|png|bmp|gif|tiff|webp)", full_print_output, re.IGNORECASE))
        if path_like_output:
            error_msg = f"Sandbox for {item_id}: Path/result output error, unable to match save path"
            return_dict["error"] = error_msg

    # picklable_variables = {}
    # imports_to_persist = {}

    # for name, value in exec_scope.items():
    #     if name == "__builtins__":
    #         continue
    #     if name in sandbox_globals_always.keys():
    #         continue
    #     if isinstance(value, types.ModuleType):
    #         imports_to_persist[name] = value.__name__
    #     else:
    #         try:
    #             picklable_variables[name] = value
    #         except (pickle.PicklingError, TypeError):
    #             print(f"Warning: Var '{name}' of type {type(value).__name__} is not picklable.", file=sys.stderr)

    # # Store the final state of globals and locals
    # return_dict["execution_context"] = {
    #     'globals': imports_to_persist,
    #     'locals': picklable_variables
    # }
    return return_dict